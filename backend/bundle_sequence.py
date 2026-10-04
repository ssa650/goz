"""Frozen ordered clip bundles, concurrent generation, and ordered assembly."""
import asyncio
from copy import deepcopy
from uuid import UUID
from pydantic import ValidationError
from .clip_settings import ClipGenerationSettings, generation_options
from .clips import ACTIVE_STATUSES
from .config import MAX_CLIPS
from .fal_adapter import FalError
from .frames import stitch_videos

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class ClipValidationError(ValueError):
    def __init__(self, errors):
        super().__init__("Fix the highlighted clips before generating.")
        self.errors = errors


class BundleSequenceService:
    def __init__(self, engine):
        self.engine = engine

    def active(self):
        return next((s for s in self.engine.sequences.values() if s.get("mode") == "bundles" and
                     (s["status"] not in TERMINAL or any(self.engine.jobs.get(c.get("jobId"), {}).get("status") in ACTIVE_STATUSES for c in s["clips"]))), None)

    def validate(self, items):
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_CLIPS:
            raise ValueError(f"Supply between 1 and {MAX_CLIPS} ordered clip bundles.")
        errors, clips, ids, orders = {}, [], set(), set()
        for position, item in enumerate(items):
            clip_id = item.get("id") if isinstance(item, dict) else None
            label = clip_id if isinstance(clip_id, str) else f"position:{position}"
            try:
                if not isinstance(item, dict):
                    raise ValueError("Supply a complete clip object.")
                UUID(clip_id)
                if clip_id in ids:
                    raise ValueError("Duplicate clip ID. Duplicate clips must have their own ID.")
                ids.add(clip_id)
                order = item.get("order")
                if type(order) is not int or order < 0 or order >= len(items) or order in orders:
                    raise ValueError("Clip order must be unique and consecutive, starting at zero.")
                orders.add(order)
                record = self.engine.clips.get(clip_id)
                settings = ClipGenerationSettings.model_validate({k:v for k,v in item.items() if k not in ("id", "order")})
                if not settings.prompt.strip():
                    raise ValueError("Prompt is empty.")
                self.engine.clips.validate_references(settings)
                self.engine.clips.validate_required_frames(record, settings)
                clips.append(dict(id=clip_id, order=order, **settings.model_dump(mode="json")))
            except (ValueError, TypeError, AttributeError, FalError) as error:
                errors[label] = "; ".join(x["msg"] for x in error.errors()) if isinstance(error, ValidationError) else str(error)
        if errors:
            raise ClipValidationError(errors)
        library_ids = {c["id"] for c in self.engine.clips.library()["clipDefinitions"]}
        if ids != library_ids:
            raise FalError("The sequence changed. Reload its clip cards before generating.", 409)
        return sorted(clips, key=lambda c:c["order"])

    async def start(self, body):
        from .engine import now
        e = self.engine
        run_id = body.get("id")
        if not isinstance(run_id, str):
            raise ValueError("Supply a unique sequence UUID.")
        run_id = str(UUID(run_id))
        if set(body) - {"id", "clips"}:
            raise ValueError("Submit one ordered clips array, with all settings inside each clip bundle.")
        async with e.lock:
            # A retry never generates another paid batch, including after edits/reload.
            if run_id in e.sequences:
                existing = e.sequences[run_id]
                if existing.get("mode") != "bundles":
                    raise FalError("This run ID belongs to another sequence.", 409)
                return self.snapshot(existing)
            e.generation_guard()
            ordered = self.validate(body.get("clips"))
            if not e.adapter.configured():
                raise FalError("Add your Fal key first.", 503)
            if e.busy():
                raise FalError("Wait for existing generations, or resolve an uncertain Fal request, before generating the sequence.", 409)
            # Commit the exact frontend order and all settings as one validated unit.
            records = {c["id"]:c for c in e.clips.library()["clipDefinitions"]}
            e.clips.library()["clipDefinitions"] = [records[c["id"]] for c in ordered]
            sequence = dict(id=run_id, mode="bundles", status="preparing", clips=deepcopy(ordered), startedAt=now(), finalVideoUrl=None)
            e.sequences[run_id] = sequence
            for clip in sequence["clips"]:
                settings = e.clips.settings(clip)
                records[clip["id"]].update(settings.model_dump(mode="json"), order=clip["order"])
                job = e.new_job(generation_options(settings), clipId=clip["id"], sequenceId=run_id,
                                sequenceIndex=clip["order"], settings=settings.model_dump(mode="json"))
                clip.update(jobId=job["id"], status="queued", generatedVideoUrl=None)
                records[clip["id"]].update(jobId=job["id"], status="queued", error=None)
                records[clip["id"]]["jobIds"].append(job["id"])
                e.update(job, status="queued")
            e.persist()
            e.sequence_tasks[run_id] = e.spawn(self.run(sequence))
            return self.snapshot(sequence)

    def snapshot(self, sequence):
        result = deepcopy(sequence)
        for clip in result["clips"]:
            job = self.engine.jobs.get(clip.get("jobId"))
            clip["result"] = self.engine.public_job(job) if job else None
        result["generationBusy"] = self.engine.busy()
        return result

    def job_updated(self, job):
        sequence = self.engine.sequences.get(job.get("sequenceId"))
        if not sequence or sequence.get("mode") != "bundles":
            return
        clip = next(c for c in sequence["clips"] if c["id"] == job["clipId"])
        clip.update(status=job["status"], error=job.get("error"),
                    generatedVideoUrl=f"/api/jobs/{job['id']}/video" if job["status"] == "completed" else None)

    async def run(self, sequence):
        from .engine import now
        e = self.engine
        gate = asyncio.Semaphore(3)
        async def generate(clip):
            job = e.jobs[clip["jobId"]]
            async with gate:
                if sequence["status"] in TERMINAL or job.get("cancelRequested"):
                    e.finish(job, status="cancelled")
                    return
                if any(j.get("requestUncertain") for j in e.jobs.values()):
                    e.finish(job, status="failed", error="A Fal submission is uncertain. No further clips were submitted; check Fal history.")
                    return
                rejected = next((e.jobs[c["jobId"]] for c in sequence["clips"]
                                 if e.jobs[c["jobId"]].get("providerStatus") in (401, 403)), None)
                if rejected:
                    e.finish(job, status="failed", error=rejected["error"])
                    return
                await e.run_job(job, {})
        try:
            e.update(sequence, status="generating")
            await asyncio.gather(*(generate(c) for c in sorted(sequence["clips"], key=lambda c:c["order"])))
            if sequence["status"] in TERMINAL:
                return
            if any(c["status"] != "completed" for c in sequence["clips"]):
                failed = [c for c in sequence["clips"] if c["status"] != "completed"]
                errors = list(dict.fromkeys(c.get("error") or "Generation did not complete." for c in failed))
                detail = errors[0] if len(errors) == 1 else " · ".join(
                    f"Clip {c['order'] + 1}: {c.get('error') or c['status']}" for c in failed[:3])
                raise ValueError(f"{detail} No final video was stitched.")
            e.update(sequence, status="stitching")
            # Re-sort immediately before media assembly; completion timing is irrelevant.
            ordered = sorted(sequence["clips"], key=lambda c:c["order"])
            inputs = [dict(clipId=c["id"], order=c["order"], duration=c["duration"],
                           videoUrl=c["generatedVideoUrl"], path=await e.media_path(e.jobs[c["jobId"]])) for c in ordered]
            sequence["stitchInputs"] = [{k:v for k,v in item.items() if k != "path"} for item in inputs]
            e.persist()
            await stitch_videos(inputs, e.media/f"sequence-{sequence['id']}.mp4")
            if sequence["status"] not in TERMINAL:
                e.update(sequence, status="completed", finishedAt=now(), finalVideoUrl=f"/api/sequences/{sequence['id']}/video")
        except Exception as error:
            if sequence["status"] not in TERMINAL:
                e.update(sequence, status="failed", error=e.error(error), finishedAt=now())
        finally:
            e.sequence_tasks.pop(sequence["id"], None)

    async def cancel(self, sequence):
        from .engine import now
        e = self.engine
        if sequence["status"] not in TERMINAL:
            stitching = sequence["status"] == "stitching"
            e.update(sequence, status="cancelled", finishedAt=now(), warning="Accepted Fal requests may still finish and be charged.")
            await asyncio.gather(*(e.cancel_job(e.jobs[c["jobId"]]) for c in sequence["clips"]))
            if stitching and (task := e.sequence_tasks.get(sequence["id"])):
                task.cancel()
        return sequence
