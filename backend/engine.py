"""Generation, ordering, persistence, cancellation, and restart recovery."""
import asyncio
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4
from .config import MODELS, POLL_SECONDS, MAX_CLIPS
from .frames import download_video, extract_last_frame, video_url, media_metadata
from .fal_adapter import FalError
from .prompts import plan
from .clip_settings import random_seed, build_h3_request, ReferenceFrame
from .adaptive import decision_trace
from .clips import ClipService
from .bundle_sequence import BundleSequenceService

JOB_TERMINAL = {"completed", "failed", "cancelled"}
SEQUENCE_TERMINAL = JOB_TERMINAL | {"interrupted"}
now = lambda: int(time.time()*1000)


def timing_ms(value):
    """Provider seconds only; missing, boolean and malformed values stay unknown."""
    return round(float(value) * 1000, 3) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


class Engine:
    def __init__(self, adapter, directory, poll_seconds=POLL_SECONDS, max_job_seconds=1200, extractor=extract_last_frame):
        self.adapter = adapter
        self.directory = Path(directory)
        self.media = self.directory/"media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.images = self.directory/"images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.image_refs = {p.stem: json.loads(p.read_text()) for p in self.images.glob("*.json")}
        self.poll_seconds, self.max_job_seconds = poll_seconds, max_job_seconds
        self.extractor = extractor
        self.jobs = self.load("history.json")
        self.sequences = self.load("sequences.json")
        self.trace_journal = decision_trace.Journal(self.directory)
        self.clips = ClipService(self)
        self.bundles = BundleSequenceService(self)
        self.lock = asyncio.Lock()
        self.tasks, self.sequence_tasks, self.uploads, self.media_locks = set(), {}, {}, {}
        self.monitored = set()
        self.stopping = False
        self.generation_guard = lambda: None
        self.adaptive_active = lambda: False
        for sequence in self.sequences.values():
            if sequence["status"] not in SEQUENCE_TERMINAL:
                sequence.update(status="interrupted", finishedAt=now(), error="Server restarted. No more clips will be submitted. Check the saved Fal request before starting another run.")
        self.persist()

    def load(self, name):
        path = self.directory/name
        if not path.exists():
            return {}
        # Fail startup rather than silently lose uncertainty/paid request records.
        return {record["id"]: record for record in json.loads(path.read_text())}

    def persist(self):
        protected_jobs = {c.get("jobId") for s in self.sequences.values() if s.get("mode") == "individual" for c in s.get("clipDefinitions", [])}
        protected_jobs.update(c.get("jobId") for s in self.sequences.values() if s.get("mode") == "individual" for c in s.get("beforeImport", []))
        protected_jobs.update(c.get("jobId") for s in self.sequences.values() if s.get("mode") == "bundles" for c in s["clips"])
        for name, records, limit in (("history.json", self.jobs, 100), ("sequences.json", self.sequences, 20)):
            while len(records) > limit:
                removable = next((key for key, record in records.items()
                    if record["status"] in SEQUENCE_TERMINAL and record.get("mode") != "individual" and record["id"] not in protected_jobs and not record.get("requestUncertain")), None)
                if removable is None:
                    break
                records.pop(removable)
            path = self.directory/name
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(list(records.values()), indent=2))
            temporary.chmod(0o600)
            os.replace(temporary, path)

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def busy(self):
        return self.adaptive_active() or any(j["status"] not in JOB_TERMINAL or j.get("requestUncertain") for j in self.jobs.values()) or any(
            s["status"] not in SEQUENCE_TERMINAL for s in self.sequences.values())

    def snapshot(self, sequence):
        if sequence.get("mode") == "bundles":
            return self.bundles.snapshot(sequence)
        result = {**sequence, "generationBusy": self.busy()}
        if sequence.get("clipDefinitions") and sequence.get("mode") != "individual":
            # Read-only compatibility for older clients. Definitions are the source.
            result["prompts"] = [c["prompt"] for c in sequence["clipDefinitions"]]
        return result

    def public_job(self, job):
        result = {**job, "serverNow": now()}
        if job.get("video"):
            result["video"] = {**job["video"], "url": f"/api/jobs/{job['id']}/video"}
        if job["status"] not in JOB_TERMINAL:
            result["totalElapsedMs"] = now()-job["startedAt"]
            result["apiElapsedMs"] = now()-job["apiStartedAt"] if job.get("apiStartedAt") else None
        return result

    def error(self, error):
        message = self.adapter.redact(str(error))
        if os.getenv("OPENAI_API_KEY"):
            message = message.replace(os.environ["OPENAI_API_KEY"], "[redacted]")
        return message[:1800] or "The request failed."

    def update(self, record, **patch):
        record.update(patch)
        decision_trace.refresh(record, getattr(self.adapter, "key", None))
        if record.get("decisionTrace"):
            self.trace_journal.schedule(record["decisionTrace"])
        if record.get("sequenceId"):
            sequence = self.sequences[record["sequenceId"]]
            if sequence["status"] not in SEQUENCE_TERMINAL:
                sequence.update(generationStatus=record["status"], requestId=record.get("requestId"), warning=record.get("connectionWarning"))
        self.clips.job_updated(record)
        self.bundles.job_updated(record)
        self.persist()

    def finish(self, job, **patch):
        self.update(job, **patch, finishedAt=now(), totalElapsedMs=now()-job["startedAt"],
                    apiElapsedMs=now()-job["apiStartedAt"] if job.get("apiStartedAt") else None)

    def new_job(self, options, **metadata):
        options = {**options}
        if options["mode"] in ("text", "frames"):
            if options.get("seed") is None:
                options["seed"] = random_seed()
            options.setdefault("promptExpansionMode", "disabled")
        job = dict(id=str(uuid4()), **options, **metadata, model=MODELS[options["mode"]], status="uploading", startedAt=now())
        self.jobs[job["id"]] = job
        if job.get("decisionTrace"):
            self.trace_journal.schedule(job["decisionTrace"])
        self.persist()
        return job

    async def upload(self, image):
        digest = hashlib.sha256(image.data).hexdigest()
        if digest not in self.uploads:
            self.uploads[digest] = asyncio.create_task(self.adapter.upload(image))
        task = self.uploads[digest]
        try:
            url = await task
            if not isinstance(url, str) or not url.strip():
                raise ValueError("Image upload returned no usable URL. Try uploading the image again.")
            if not self.adapter.demo:
                video_url(url)
            return url
        except Exception:
            self.uploads.pop(digest, None)
            raise
        finally:
            if len(self.uploads) > 100:
                oldest = next((key for key, value in self.uploads.items() if value.done()), None)
                if oldest:
                    self.uploads.pop(oldest, None)

    def can_continue(self, job):
        sequence = self.sequences.get(job.get("sequenceId"))
        return not job.get("cancelRequested") and (not sequence or sequence["status"] not in SEQUENCE_TERMINAL)

    async def run_job(self, job, images):
        try:
            mapping = {}
            if job.get("clipId"):
                for field, key in (("firstFrame", "start"), ("endFrame", "end")):
                    if job.get(field):
                        mapping[key] = await self.upload_reference(job[field])
            else:
                names = list(images)
                urls_list = await asyncio.gather(*(self.upload(image) for name in names for image in images[name]))
                cursor = 0
                for name in names:
                    values = urls_list[cursor:cursor+len(images[name])]
                    cursor += len(values)
                    mapping[name] = values if name == "characters" else values[0]
                    if name in ("start", "end"):
                        reference = self.save_reference(images[name][0])
                        reference["providerUrl"] = values[0]
                        self.persist_reference(reference)
                        job["firstFrame" if name == "start" else "endFrame"] = reference["id"]
            if not self.can_continue(job):
                self.finish(job, status="cancelled")
                return job
            prompt_started = time.perf_counter()
            if job["mode"] in ("frames", "text"):
                model, payload = build_h3_request(job, mapping)
                job["model"] = model
            else:
                from .config import build_input
                payload = build_input(job, mapping)
            # Freeze the exact provider input, including uploaded URLs, before submission.
            job["generationInput"] = payload
            self.update(job, uploadElapsedMs=now()-job["startedAt"],
                        payloadConstructionMs=round((time.perf_counter()-prompt_started)*1000, 3),
                        apiStartedAt=now(), status="submitting")
            submission_started = time.perf_counter()
            decision_trace.attempted(job, payload, now(), getattr(self.adapter, "key", None))
            if job.get("decisionTrace"):
                self.trace_journal.schedule(job["decisionTrace"])
            submitted = await self.adapter.submit(job["model"], payload)
            job["submissionMs"] = round((time.perf_counter()-submission_started)*1000, 3)
            if not submitted.get("request_id"):
                raise FalError("Fal returned no request ID. Check your Fal history.")
            self.update(job, requestId=submitted["request_id"], submittedAt=now(), status="queued")
            if job.get("cancelRequested"):
                await self.cancel_job(job)
            await self.monitor(job)
        except Exception as error:
            if getattr(error, "status", None) is not None:
                job["providerStatus"] = error.status
            if job.get("apiStartedAt") and not job.get("requestId") and getattr(error, "status", None) not in (400,401,403,422):
                job["requestUncertain"] = True
            self.finish(job, status="failed", error=self.error(error)+(" Check Fal request history; new submissions are blocked." if job.get("requestUncertain") else ""))
        return job

    async def monitor(self, job):
        if job["id"] in self.monitored:
            return
        self.monitored.add(job["id"])
        try:
            errors = 0
            while not self.stopping and job["status"] not in JOB_TERMINAL:
                if now()-job["startedAt"] > self.max_job_seconds*1000:
                    job["requestUncertain"] = True
                    raise FalError("Monitoring timed out. Check this request in Fal before running again.")
                try:
                    status = await self.adapter.status(job["model"], job["requestId"])
                    errors = 0
                except Exception as error:
                    if getattr(error, "status", None) in (400,401,403,404,422):
                        job["requestUncertain"] = getattr(error, "status", None) in (401,403,404)
                        raise
                    errors += 1
                    self.update(job, connectionWarning="Status connection interrupted; reconnecting to the same Fal request.")
                    await asyncio.sleep(min(5, self.poll_seconds * 2**min(errors, 4)))
                    continue
                if not isinstance(status, dict):
                    job["requestUncertain"] = True
                    raise FalError("Fal returned malformed queue status; inspect the saved request before retrying.")
                if status.get("status") in ("CANCELLED", "CANCELED"):
                    self.finish(job, status="cancelled")
                    return
                provider_state = status.get("status")
                if provider_state not in ("IN_QUEUE", "IN_PROGRESS", "COMPLETED"):
                    job["requestUncertain"] = True
                    raise FalError("Fal returned an unrecognized queue status; inspect the saved request before retrying.")
                if provider_state == "IN_PROGRESS" and not job.get("firstProgressObservedAt"):
                    job["firstProgressObservedAt"] = now()
                    # Polling observations are upper bounds, not provider queue timestamps.
                    job["queueObservedMs"] = now() - job.get("submittedAt", job["apiStartedAt"])
                self.update(job, status="queued" if provider_state == "IN_QUEUE" else "generating",
                            queuePosition=status.get("queue_position"), connectionWarning=None)
                if provider_state == "COMPLETED":
                    api_ready = now()-job["apiStartedAt"]
                    job["providerCompletedObservedAt"] = now()
                    job["providerObservedElapsedMs"] = now()-job.get("submittedAt", job["apiStartedAt"])
                    metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
                    job["providerMetrics"] = {k: v for k, v in metrics.items() if timing_ms(v) is not None}
                    job["providerRunnerMs"] = timing_ms(metrics.get("inference_time"))
                    if status.get("error"):
                        raise FalError(f"Fal reported a failed request: {status['error']}")
                    result_started = time.perf_counter()
                    result = await self.adapter.result(job["model"], job["requestId"])
                    job["resultRetrievalMs"] = round((time.perf_counter()-result_started)*1000, 3)
                    if not isinstance(result, dict):
                        raise ValueError("Fal returned a malformed result; expected a video object.")
                    data = result.get("data", result)
                    if not isinstance(data, dict) or not isinstance(data.get("video"), dict):
                        raise ValueError("Fal completed without a usable video object.")
                    if not isinstance(data.get("video", {}).get("url"), str):
                        raise ValueError("Fal completed without a usable video URL.")
                    if not self.adapter.demo:
                        video_url(data["video"]["url"])
                    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
                    self.finish(job, status="completed", video=data["video"],
                                timings={k: v for k, v in timings.items() if timing_ms(v) is not None},
                                providerInferenceMs=timing_ms(timings.get("inference")),
                                seed=data.get("seed") if type(data.get("seed")) is int else job.get("seed"),
                                seedSource="provider" if type(data.get("seed")) is int else "submitted" if job.get("seed") is not None else "unknown", expandedPrompt=data.get("expanded_prompt"), apiReadyMs=api_ready)
                    return
                await asyncio.sleep(self.poll_seconds)
        except Exception as error:
            self.finish(job, status="failed", error=self.error(error))
        finally:
            self.monitored.discard(job["id"])

    async def cancel_job(self, job):
        if job["status"] in JOB_TERMINAL:
            return
        self.update(job, cancelRequested=True)
        if not job.get("requestId") or job.get("cancelSent"):
            return
        self.update(job, cancelSent=True)
        try:
            await self.adapter.cancel(job["model"], job["requestId"])
        except Exception:
            self.update(job, connectionWarning="Cancellation could not be confirmed. Waiting for the submitted request.")

    async def start_sequence(self, run_id, text, mode, duration, resolution, frames):
        async with self.lock:
            if run_id in self.sequences:
                return self.sequences[run_id]
            self.generation_guard()
            if not self.adapter.configured():
                raise FalError("Add your Fal key first.", 503)
            if self.busy():
                raise FalError("Another generation is active or its submission is uncertain. Check its status before running again.", 409)
            planned = plan(text, mode, duration)
            if len(frames) != planned["frameCount"]:
                raise ValueError(f"Upload exactly {planned['frameCount']} frames for {len(planned['clips'])} clips.")
            # Convert the deprecated multipart interface into atomic bundles at
            # ingress. The runner never indexes a separate prompt/frame array.
            references = [self.save_reference(image)["id"] for image in frames]
            definitions = [dict(id=str(uuid4()), order=index, prompt=prompt, seed=random_seed(),
                                firstFrame=references[index] if mode == "keyframes" else references[0] if index == 0 else None,
                                endFrame=references[index+1] if mode == "keyframes" else None,
                                duration=duration, resolution=resolution, promptExpansionMode="disabled")
                           for index, prompt in enumerate(planned["clips"])]
            sequence = dict(id=run_id, mode=mode, clipDefinitions=definitions, scenes=planned["scenes"], duration=duration,
                            resolution=resolution, model=MODELS["frames"], status="preparing", index=0, clips=[], startedAt=now())
            self.sequences[run_id] = sequence
            self.persist()
            self.sequence_tasks[run_id] = self.spawn(self.run_sequence(sequence))
            return sequence

    async def run_sequence(self, sequence):
        try:
            for index, definition in enumerate(sequence["clipDefinitions"]):
                if sequence["status"] in SEQUENCE_TERMINAL:
                    return
                self.update(sequence, status="generating", index=index)
                from .clip_settings import generation_options
                job = self.new_job(generation_options(self.clips.settings(definition)), clipId=definition["id"],
                                   sequenceId=sequence["id"], sequenceIndex=index, frameControl=sequence["mode"])
                definition["jobId"] = job["id"]
                self.update(sequence, activeJobId=job["id"])
                await self.run_job(job, {})
                if sequence["status"] in SEQUENCE_TERMINAL:
                    return
                if job["status"] != "completed":
                    raise ValueError(job.get("error", "Generation did not complete."))
                clip = dict(index=index, jobId=job["id"], url=f"/api/jobs/{job['id']}/video", duration=job["duration"],
                            totalElapsedMs=job.get("totalElapsedMs"), uploadElapsedMs=job.get("uploadElapsedMs"),
                            apiReadyMs=job.get("apiReadyMs"), inferenceMs=(job["timings"]["inference"]*1000 if (job.get("timings") or {}).get("inference") is not None else None))
                sequence["clips"].append(clip)
                self.update(sequence, activeJobId=None, requestId=None, generationStatus=None)
                if index+1 == len(sequence["clipDefinitions"]):
                    self.update(sequence, status="completed", finishedAt=now())
                    return
                if sequence["mode"] == "chain":
                    self.update(sequence, status="extracting")
                    started = time.monotonic()
                    image = await self.extractor(await self.media_path(job))
                    sequence["clipDefinitions"][index+1]["firstFrame"] = self.save_reference(image)["id"]
                    clip["extractMs"] = (time.monotonic()-started)*1000
                    self.persist()
        except Exception as error:
            if sequence["status"] not in SEQUENCE_TERMINAL:
                self.update(sequence, status="failed", error=self.error(error), finishedAt=now())
        finally:
            self.sequence_tasks.pop(sequence["id"], None)

    async def cancel_sequence(self, sequence):
        if sequence.get("mode") == "bundles":
            return await self.bundles.cancel(sequence)
        if sequence["status"] not in SEQUENCE_TERMINAL:
            extracting = sequence["status"] == "extracting"
            self.update(sequence, status="cancelled", finishedAt=now(), warning="No further clips will start. An accepted Fal request may still finish and be charged.")
            if extracting and (task := self.sequence_tasks.get(sequence["id"])):
                task.cancel()
            if job := self.jobs.get(sequence.get("activeJobId")):
                await self.cancel_job(job)
        return sequence

    async def media_path(self, job):
        path = self.media/f"{job['id']}.mp4"
        async with self.media_locks.setdefault(job["id"], asyncio.Lock()):
            if path.exists():
                return path
            temporary = path.with_suffix(".part")
            download_started = time.perf_counter()
            self.update(job, mediaStatus='downloading', mediaDownloadStartedAt=now())
            try:
                if self.adapter.demo:
                    request_id = job["video"]["url"].removeprefix("demo:")
                    source = self.adapter.directory/f"{request_id}.mp4"
                    shutil.copyfile(source, temporary)
                    temporary.chmod(0o600)
                else:
                    await download_video(job["video"]["url"], temporary)
                download_ms = round((time.perf_counter()-download_started)*1000, 3)
                check_started = time.perf_counter()
                self.update(job, mediaStatus='validating', mediaDownloadMs=download_ms, mediaDownloadedAt=now())
                metadata = await asyncio.to_thread(media_metadata, temporary)
                if not metadata.get("size") or not metadata.get("duration", 0):
                    raise ValueError("Downloaded provider result has no playable video stream.")
                os.replace(temporary, path)
                self.update(job, mediaStatus='ready', mediaDownloadMs=download_ms,
                            mediaValidationMs=round((time.perf_counter()-check_started)*1000, 3),
                            mediaReadyAt=now(), mediaReadyElapsedMs=now()-job["startedAt"],
                            actualDuration=metadata["duration"],
                            hasAudio=bool(metadata.get("audio_codec")))
            except BaseException as error:
                temporary.unlink(missing_ok=True)
                self.update(job, mediaStatus='cancelled' if isinstance(error, asyncio.CancelledError) else 'failed',
                            mediaError=self.error(error), mediaFailedAt=now())
                raise
        return path

    def persist_reference(self, reference):
        path = self.images/f"{reference['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(reference))
        temporary.chmod(0o600)
        os.replace(temporary, path)
        self.image_refs[reference["id"]] = reference

    def save_reference(self, image, reuse=True):
        digest = hashlib.sha256(image.data).hexdigest()
        existing = next((ref for ref in self.image_refs.values() if ref.get("sha256") == digest), None)
        if existing and reuse:
            return existing
        # An explicit replacement owns a new reference, even for identical bytes.
        # Refresh a completed upload cache entry so replacing an expired URL can recover.
        cached = self.uploads.get(digest)
        if not reuse and cached and cached.done():
            self.uploads.pop(digest, None)
        reference_id = str(uuid4())
        path = self.images/f"{reference_id}.bin"
        path.write_bytes(image.data)
        path.chmod(0o600)
        reference = dict(id=reference_id, name=image.name, contentType=image.content_type,
                         size=len(image.data), previewUrl=f"/api/images/{reference_id}", providerUrl=None, sha256=digest)
        self.persist_reference(reference)
        return reference

    def reference_exists(self, reference_id):
        return reference_id in self.image_refs and (self.images/f"{reference_id}.bin").is_file()

    def reference(self, reference_id):
        if not self.reference_exists(reference_id):
            raise ValueError("The reference image is missing. Upload it again.")
        return ReferenceFrame.model_validate(self.image_refs[reference_id]).model_dump(mode="json")

    async def upload_reference(self, reference_id):
        reference = self.image_refs[reference_id]
        if reference.get("providerUrl"):
            if not self.adapter.demo:
                video_url(reference["providerUrl"])
            return reference["providerUrl"]
        from .frames import verify_image
        image = await asyncio.to_thread(verify_image, (self.images/f"{reference_id}.bin").read_bytes(), reference["name"])
        reference["providerUrl"] = await self.upload(image)
        self.persist_reference(reference)
        return reference["providerUrl"]

    async def resume(self):
        for job in self.jobs.values():
            if job["status"] in JOB_TERMINAL:
                continue
            if job.get("requestId"):
                if self.adapter.configured():
                    self.spawn(self.monitor(job))
                else:
                    self.update(job, connectionWarning="Restore the same Fal key to reconnect to this request.")
            else:
                if job.get("apiStartedAt"):
                    job["requestUncertain"] = True
                self.finish(job, status="failed", error="Server restarted before saving a Fal request ID. Check Fal history before retrying.")

    async def close(self):
        self.stopping = True
        for task in (*self.tasks, *self.uploads.values()):
            task.cancel()
        await asyncio.gather(*self.tasks, *self.uploads.values(), return_exceptions=True)
        await self.trace_journal.flush()
        await self.adapter.close()
