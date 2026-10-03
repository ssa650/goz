"""Generation, ordering, persistence, cancellation, and restart recovery."""
import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4
from .config import MODELS, POLL_SECONDS, MAX_CLIPS
from .frames import download_video, extract_last_frame, video_url
from .fal_adapter import FalError
from .prompts import plan

JOB_TERMINAL = {"completed", "failed", "cancelled"}
SEQUENCE_TERMINAL = JOB_TERMINAL | {"interrupted"}
now = lambda: int(time.time()*1000)


class Engine:
    def __init__(self, adapter, directory, poll_seconds=POLL_SECONDS, max_job_seconds=1200, extractor=extract_last_frame):
        self.adapter = adapter
        self.directory = Path(directory)
        self.media = self.directory/"media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.poll_seconds, self.max_job_seconds = poll_seconds, max_job_seconds
        self.extractor = extractor
        self.jobs = self.load("history.json")
        self.sequences = self.load("sequences.json")
        self.lock = asyncio.Lock()
        self.tasks, self.sequence_tasks, self.uploads, self.media_locks = set(), {}, {}, {}
        self.monitored = set()
        self.stopping = False
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
        for name, records, limit in (("history.json", self.jobs, 100), ("sequences.json", self.sequences, 20)):
            while len(records) > limit:
                removable = next((key for key, record in records.items()
                    if record["status"] in SEQUENCE_TERMINAL and not record.get("requestUncertain")), None)
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
        return any(j["status"] not in JOB_TERMINAL or j.get("requestUncertain") for j in self.jobs.values()) or any(
            s["status"] not in SEQUENCE_TERMINAL for s in self.sequences.values())

    def snapshot(self, sequence):
        return {**sequence, "generationBusy": self.busy()}

    def public_job(self, job):
        result = {**job, "serverNow": now()}
        if job.get("video"):
            result["video"] = {**job["video"], "url": f"/api/jobs/{job['id']}/video"}
        if job["status"] not in JOB_TERMINAL:
            result["totalElapsedMs"] = now()-job["startedAt"]
            result["apiElapsedMs"] = now()-job["apiStartedAt"] if job.get("apiStartedAt") else None
        return result

    def error(self, error):
        return self.adapter.redact(str(error))[:1800] or "The request failed."

    def update(self, record, **patch):
        record.update(patch)
        if record.get("sequenceId"):
            sequence = self.sequences[record["sequenceId"]]
            if sequence["status"] not in SEQUENCE_TERMINAL:
                sequence.update(generationStatus=record["status"], requestId=record.get("requestId"), warning=record.get("connectionWarning"))
        self.persist()

    def finish(self, job, **patch):
        self.update(job, **patch, finishedAt=now(), totalElapsedMs=now()-job["startedAt"],
                    apiElapsedMs=now()-job["apiStartedAt"] if job.get("apiStartedAt") else None)

    def new_job(self, options, **metadata):
        job = dict(id=str(uuid4()), **options, **metadata, model=MODELS[options["mode"]], status="uploading", startedAt=now())
        self.jobs[job["id"]] = job
        self.persist()
        return job

    async def upload(self, image):
        digest = hashlib.sha256(image.data).hexdigest()
        if digest not in self.uploads:
            self.uploads[digest] = asyncio.create_task(self.adapter.upload(image))
        task = self.uploads[digest]
        try:
            return await task
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
            names = list(images)
            urls_list = await asyncio.gather(*(self.upload(image) for name in names for image in images[name]))
            if not self.can_continue(job):
                self.finish(job, status="cancelled")
                return job
            mapping, cursor = {}, 0
            for name in names:
                values = urls_list[cursor:cursor+len(images[name])]
                cursor += len(values)
                mapping[name] = values if name == "characters" else values[0]
            from .config import build_input
            payload = build_input(job, mapping)
            self.update(job, uploadElapsedMs=now()-job["startedAt"], apiStartedAt=now(), status="submitting")
            submitted = await self.adapter.submit(job["model"], payload)
            if not submitted.get("request_id"):
                raise FalError("Fal returned no request ID. Check your Fal history.")
            self.update(job, requestId=submitted["request_id"], status="queued")
            if job.get("cancelRequested"):
                await self.cancel_job(job)
            await self.monitor(job)
        except Exception as error:
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
                if status.get("status") in ("CANCELLED", "CANCELED"):
                    self.finish(job, status="cancelled")
                    return
                self.update(job, status="queued" if status.get("status") == "IN_QUEUE" else "generating",
                            queuePosition=status.get("queue_position"), connectionWarning=None)
                if status.get("status") == "COMPLETED":
                    api_ready = now()-job["apiStartedAt"]
                    result = await self.adapter.result(job["model"], job["requestId"])
                    data = result.get("data", result)
                    if not isinstance(data.get("video", {}).get("url"), str):
                        raise ValueError("Fal completed without a usable video URL.")
                    if not self.adapter.demo:
                        video_url(data["video"]["url"])
                    self.finish(job, status="completed", video=data["video"], timings=data.get("timings"),
                                seed=data.get("seed"), expandedPrompt=data.get("expanded_prompt"), apiReadyMs=api_ready)
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
            if not self.adapter.configured():
                raise FalError("Add your Fal key first.", 503)
            if self.busy():
                raise FalError("Another generation is active or its submission is uncertain. Check its status before running again.", 409)
            planned = plan(text, mode, duration)
            if len(frames) != planned["frameCount"]:
                raise ValueError(f"Upload exactly {planned['frameCount']} frames for {len(planned['clips'])} clips.")
            sequence = dict(id=run_id, mode=mode, prompts=planned["clips"], scenes=planned["scenes"], duration=duration,
                            resolution=resolution, model=MODELS["frames"], status="preparing", index=0, clips=[], startedAt=now())
            self.sequences[run_id] = sequence
            self.persist()
            self.sequence_tasks[run_id] = self.spawn(self.run_sequence(sequence, frames))
            return sequence

    async def run_sequence(self, sequence, frames):
        image = frames[0]
        try:
            for index, prompt in enumerate(sequence["prompts"]):
                if sequence["status"] in SEQUENCE_TERMINAL:
                    return
                self.update(sequence, status="generating", index=index)
                images = {"start": [frames[index]], "end": [frames[index+1]]} if sequence["mode"] == "keyframes" else {"start": [image]}
                job = self.new_job(dict(mode="frames", prompt=prompt, duration=sequence["duration"], resolution=sequence["resolution"]),
                                   sequenceId=sequence["id"], sequenceIndex=index, frameControl=sequence["mode"])
                self.update(sequence, activeJobId=job["id"])
                await self.run_job(job, images)
                if sequence["status"] in SEQUENCE_TERMINAL:
                    return
                if job["status"] != "completed":
                    raise ValueError(job.get("error", "Generation did not complete."))
                clip = dict(index=index, jobId=job["id"], url=f"/api/jobs/{job['id']}/video", duration=job["duration"],
                            totalElapsedMs=job.get("totalElapsedMs"), uploadElapsedMs=job.get("uploadElapsedMs"),
                            apiReadyMs=job.get("apiReadyMs"), inferenceMs=(job["timings"]["inference"]*1000 if (job.get("timings") or {}).get("inference") is not None else None))
                sequence["clips"].append(clip)
                self.update(sequence, activeJobId=None, requestId=None, generationStatus=None)
                if index+1 == len(sequence["prompts"]):
                    self.update(sequence, status="completed", finishedAt=now())
                    return
                if sequence["mode"] == "chain":
                    self.update(sequence, status="extracting")
                    started = time.monotonic()
                    image = await self.extractor(await self.media_path(job))
                    clip["extractMs"] = (time.monotonic()-started)*1000
                    self.persist()
        except Exception as error:
            if sequence["status"] not in SEQUENCE_TERMINAL:
                self.update(sequence, status="failed", error=self.error(error), finishedAt=now())
        finally:
            self.sequence_tasks.pop(sequence["id"], None)

    async def cancel_sequence(self, sequence):
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
            try:
                if self.adapter.demo:
                    request_id = job["video"]["url"].removeprefix("demo:")
                    source = self.adapter.directory/f"{request_id}.mp4"
                    shutil.copyfile(source, temporary)
                    temporary.chmod(0o600)
                else:
                    await download_video(job["video"]["url"], temporary)
                os.replace(temporary, path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        return path

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
        await self.adapter.close()
