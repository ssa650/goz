"""One adaptive viewing session: the generate -> watch -> adapt loop.

Scene N+1 is planned from the first 3.5 seconds of presented playback.
Available evidence is frozen at that deadline; detection never gates generation.
Saved bundles keep their original prompt and frame pair. Custom
scenes start from the actual last frame of the previous one.
All signals are logged with wall-clock timestamps under
data/adaptive/<session>/.
"""
import asyncio
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from . import director, fusion, profile as profiles, tracks, decision_trace, gaze_audit, tracking_diagnostics
from .cumulative_eeg_policy import CumulativeEEGHistory, RUN_MODE as CUMULATIVE_EEG_MODE
from ..frames import ffmpeg
from ..clip_settings import generation_options

OBSERVATION_SECONDS = 3.5
MAX_SCENES = int(os.getenv("GOZ_MAX_SCENES", "4"))
TRACKER = os.getenv("GOZ_TRACKER", "fal")
DEMO_COLORS = ["0xe07a5f", "0x3d85c6", "0x81b29a", "0xf2cc8f"]


class AdaptiveSession:
    def __init__(self, engine, gaze, eeg, directory, premise, characters, duration, resolution, opening,
                 opening_video=None, timeline="", clip_bundles=None, objects=None, tracker=None, playback_mode="download", eeg_run_mode=CUMULATIVE_EEG_MODE, eeg_gaze_only=False):
        if playback_mode not in ("download", "stream"):
            raise ValueError("Choose download or stream playback.")
        self.playback_mode = playback_mode
        self.local_media_tasks = {}
        self.engine, self.gaze, self.eeg = engine, gaze, eeg
        self.id = str(uuid4())
        self.dir = Path(directory)/"adaptive"/self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        if eeg_run_mode not in ("baseline", CUMULATIVE_EEG_MODE):
            raise ValueError("Choose baseline or cumulative_prior_clips EEG policy.")
        if not isinstance(eeg_gaze_only,bool):
            raise ValueError("EEG gaze-only choice must be a boolean.")
        self.eeg_gaze_only = eeg_gaze_only  # Snapshot the setup choice once per run.
        self.eeg_run_mode = eeg_run_mode
        self.eeg_history = CumulativeEEGHistory(self.id)
        self.names = [c["name"] for c in characters]
        self.objects = objects or []
        self.target_names = self.names + [o["name"] for o in self.objects]
        self.story = dict(premise=premise, characters=characters, objects=self.objects, scenes=[])
        self.duration, self.resolution, self.opening = duration, resolution, opening
        self.opening_video, self.timeline = opening_video, timeline
        self.clip_bundles = deepcopy(clip_bundles or [])
        self.max_scenes = min(MAX_SCENES, len(self.clip_bundles)) if self.clip_bundles else MAX_SCENES
        self.profile = profiles.new_profile(self.names)
        self.clips, self.events = [], []
        self.stage, self.error, self.status = "starting", None, "running"
        self.playing, self.rect = None, None
        self.task = None
        self.started = time.time()
        self.warnings = []
        self.generation_degraded = False
        self.generation_lock = asyncio.Lock()
        self.tracker = tracks.tracking_provider(tracker)
        self.local_worker = None
        self.local_worker_close_task = None
        self.tracking_readiness = dict(state="not_started" if self.tracker in ("color","opencv","fal") and not self.engine.adapter.demo else "not_required")
        self.detection_tasks = set()
        self.detection_by_clip = {}
        self.readiness_samples = []
        self.frame_tasks = {}
        self.boundary_frames = {}
        self.adaptation_source = None
        self.dump("session.json", dict(id=self.id,startedAt=self.started,playbackMode=self.playback_mode,eegRunMode=self.eeg_run_mode,eegGazeOnly=self.eeg_gaze_only))

    def warn(self, component, error):
        message = self.engine.error(error)
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            message = message.replace(key, "[redacted]")
        self.warnings.append(dict(component=component, message=message))
        self.log("fallback", component=component, message=message)

    # -- public state ------------------------------------------------------
    def log(self, kind, **data):
        event = dict(t=time.time(), kind=kind, **data)
        self.events.append(event)
        with (self.dir/"events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def set_stage(self, stage):
        self.stage = stage
        self.log("stage", stage=stage)

    def public(self):
        clips = [{k: v for k, v in c.items() if k not in ("ticks", "timeline", "image", "frozenEvidence", "decisionTrace")} for c in self.clips]
        for clip in clips:
            if job := self.engine.jobs.get(clip.get("jobId")):
                trace = job.get("decisionTrace") or {}
                clip["decisionTrace"] = {k: trace.get(k) for k in ("submission", "promptExact", "changed")}
                clip["generationInput"] = job.get("generationInput")
                clip["generationStatus"] = job.get("status")
                clip["queuePosition"] = job.get("queuePosition")
                clip["timing"] = {k:v for k,v in job.items() if k.endswith("Ms") or k in ("providerMetrics", "timings")}
        return dict(id=self.id, status=self.status, stage=self.stage, error=self.error, names=self.target_names, playbackMode=self.playback_mode, eegRunMode=self.eeg_run_mode, eegGazeOnly=self.eeg_gaze_only,
                    story=self.story, profile=self.profile, clips=clips, playing=self.playing,
                    duration=self.duration, demo=self.engine.adapter.demo, maxScenes=self.max_scenes, tracker=self.tracker, trackingReadiness=deepcopy(self.tracking_readiness),
                    trackerAvailability=dict(yoloe=tracks.yoloe_availability()),
                    events=self.events[-20:], warnings=self.warnings[-6:],
                    latency=dict(samples=len(self.readiness_samples), readinessS=self.readiness_samples,
                                 estimateS=self.readiness_estimate()),
                    queue=dict(capacity=1, future=sum(c["index"] > (self.playing if self.playing is not None else -1) and c["status"] != "failed" for c in self.clips),
                               mode="held-frame bridge" if self.current_clip() and self.current_clip()["status"] == "watched" and self.status == "running" else "live"))

    def readiness_estimate(self):
        # Conservative observed readiness, not a claimed inference duration.
        return max(self.readiness_samples[-3:]) if self.readiness_samples else 20.0

    def analyze_at(self, clip):
        return min(OBSERVATION_SECONDS, clip["duration"])

    def source_current(self, source):
        if source is None:
            return self.status == "running"
        index, clip_id = source
        return (self.status == "running" and index < len(self.clips)
                and self.clips[index]["id"] == clip_id
                and (self.playing is None or self.playing == index))

    def prepare_boundary(self, clip, path):
        """Decode the file's actual end before playback, never a displayed frame."""
        if self.clip_bundles or clip["index"] + 1 >= self.max_scenes:
            return None
        if clip["id"] in self.boundary_frames:
            return None
        if clip["id"] not in self.frame_tasks:
            async def work():
                started = time.perf_counter()
                try:
                    image = await self.engine.extractor(path)
                    if image is None:
                        raise ValueError("Actual final-frame decoding returned no image.")
                    if self.status == "running":
                        self.boundary_frames[clip["id"]] = image
                        clip["boundaryFrameStatus"] = "ready"
                except Exception as error:
                    clip["boundaryFrameStatus"] = "failed"
                    self.warn("Last frame", error)
                finally:
                    clip["boundaryFrameMs"] = round((time.perf_counter()-started)*1000, 2)
            clip["boundaryFrameStatus"] = "processing"
            self.frame_tasks[clip["id"]] = self.engine.spawn(work())
        return self.frame_tasks[clip["id"]]

    def eeg_quality_snapshot(self):
        # Copy feed identity, never mutate acquisition/calibration. A concurrent
        # channel change produces an incompatible snapshot and abstention.
        quality = deepcopy(self.eeg.status())
        quality["calibration"] = deepcopy(getattr(self.eeg, "calibration", None) or {})
        quality["gazeOnly"] = self.eeg_gaze_only
        if self.eeg_gaze_only:
            quality.update(measuredLive=quality.get("live"),measuredConfidence=quality.get("confidence"),
                live=False,confidence=0.0,policyOverrideReason="Explicit gaze-only choice for this run")
        return quality

    def collect_viewing_eeg(self, clip, observed_at):
        if self.eeg_run_mode != CUMULATIVE_EEG_MODE or not clip.get("eegHistoryStarted"):
            return
        # Current quality cannot be applied retroactively to delayed HTTP ticks.
        if not 0 <= time.time()-observed_at <= .75:
            return
        try:
            samples = self.eeg.window(max(clip["playStartedAt"],observed_at-.75), observed_at)
            added = self.eeg_history.collect(self.id,clip["id"],samples,self.eeg_quality_snapshot(),
                observed_at=observed_at)
            if added:
                tracking_diagnostics.for_clip(self,clip).record("eeg_history_collection",
                    runMode=self.eeg_run_mode,observedAt=observed_at,rowsAdded=added)
        except Exception as error:
            self.warn("EEG cumulative history",error)

    def freeze_evidence(self, clip, end):
        freeze_started = time.perf_counter()
        start = clip.get("playStartedAt", end)
        # Bound delayed HTTP/task processing to the opening playback window.
        end = min(end, start + self.analyze_at(clip))
        ticks = [t for t in clip["ticks"] if start <= t["wall"] <= end]
        track = [f for f in clip.get("track", [])
                 if f.get("clip_id", clip["id"]) == clip["id"]
                 and f.get("session_id", self.id) in (None, self.id)
                 and f["t"] <= self.analyze_at(clip)]
        def read(feed):
            try:
                return feed.window(start, end)
            except Exception as error:
                self.warn("Observation", error)
                return []
        try:
            quality = self.eeg_quality_snapshot()
        except Exception as error:
            self.warn("EEG status", error)
            quality = {}
        clip["frozenEvidence"] = deepcopy(dict(start=start, end=end, ticks=ticks, track=track,
            gaze=read(self.gaze), eeg=read(self.eeg), quality=quality,
            detectionStatus=clip.get("detectionStatus", "unavailable"), frozenAt=time.time()))
        if self.eeg_run_mode == CUMULATIVE_EEG_MODE:
            frozen = clip["frozenEvidence"]
            frozen["eegPolicy"] = self.eeg_history.compare(frozen["eeg"],quality,(start,end))
            tracking_diagnostics.for_clip(self,clip).record("eeg_cumulative_frozen",
                runMode=self.eeg_run_mode,policy=frozen["eegPolicy"])
        clip["frozenEvidence"]["freezeWorkMs"] = round((time.perf_counter()-freeze_started)*1000,3)
        clip["frozenEvidence"]["trackingTiming"] = deepcopy(clip.get("trackingTiming", {}))
        clip["analysisStarted"] = True
        tracking_diagnostics.for_clip(self,clip).record("evidence_frozen", start=start,end=end,
            **tracking_diagnostics.publication_evidence(track), trackingTiming=clip.get("trackingTiming",{}))
        self.log("observation_frozen", clip=clip["index"], clipId=clip["id"],
                 start=start, end=end, detectionStatus=clip.get("detectionStatus"),
                 observationMs=max(0, (end-start)*1000))

    # -- live helpers (dashboard + simulator) ------------------------------
    def current_clip(self):
        return self.clips[self.playing] if self.playing is not None and self.playing < len(self.clips) else None

    def live_video_time(self):
        clip = self.current_clip()
        if not clip or not clip["ticks"]:
            return None
        tick = clip["ticks"][-1]
        if not tick["playing"] or time.time() - tick["wall"] > 1.0:
            return None
        return tick["video_t"] + (time.time() - tick["wall"]) * tick.get("playbackRate", 1)

    def live_boxes(self):
        clip, t = self.current_clip(), self.live_video_time()
        if clip is None or t is None:
            return {}, None
        return tracks.boxes_at(clip.get("track") or [], t), self.rect

    def live_gaze(self):
        clip, sample = self.current_clip(), self.gaze.latest()
        if not clip or not sample or sample["t"] < self.started or not 0 <= time.time() - sample["t"] <= .5:
            return None
        labelled = fusion.label([sample], clip["ticks"][-3:], clip.get("track") or [])
        return labelled[0] if labelled else None

    def live_target(self):
        g = self.live_gaze()
        return g["target"] if g else None

    # -- browser playback reports -------------------------------------------
    def tick(self, index, video_t, playing, rect, wall, clip_id=None, epoch=0, playback_rate=1.0, visible_rect=None, mapping=None):
        if not 0 <= index < len(self.clips):
            raise ValueError("Unknown clip.")
        if self.playing is not None and index < self.playing:
            return  # Late HTTP reports cannot rewind the ordered player.
        clip = self.clips[index]
        if clip["status"] not in ("ready", "playing", "watched"):
            return
        if clip_id is not None and clip_id != clip.get("id"):
            raise ValueError("Playback clip identity mismatch.")
        if self.status in ("stopped", "failed") or clip["status"] == "watched":
            return
        if abs(wall - time.time()) > 5 or video_t > clip["duration"] + 1:
            raise ValueError("Playback timestamp is outside this clip or the server clock.")
        if clip["ticks"] and wall <= clip["ticks"][-1]["wall"]:
            return
        if mapping is not None and not mapping.get("valid"):
            rect, visible_rect = None, None
        tick = dict(mapping=deepcopy(mapping), wall=wall, video_t=video_t, playing=playing, rect=rect, visible_rect=visible_rect,
                    sessionId=self.id, clipId=clip.get("id"), epoch=epoch, playbackRate=playback_rate)
        clip["ticks"].append(tick)
        if self.playing is not None and index > self.playing:
            # Release the continuation owner promptly. Provider monitoring stays
            # alive independently; cancellation cannot block the next scene.
            if self.adaptation_source and self.adaptation_source[0] < index:
                if self.task and not self.task.done():
                    self.task.cancel()
                for job in list(self.engine.jobs.values()):
                    if (job.get("adaptiveSession") == self.id
                            and job.get("sourceSceneIndex") is not None
                            and job["sourceSceneIndex"] < index
                            and job["status"] not in ("completed", "failed", "cancelled")):
                        self.engine.spawn(self.engine.cancel_job(job))
        if self.playing is not None and index > self.playing:
            for old in self.clips[:index]:
                self.cancel_detection(old)
        self.playing, self.rect = index, rect
        if clip["status"] == "ready" and playing:
            clip.update(status="playing", playStartedAt=clip.get("firstPresentedAt", wall - video_t / playback_rate))
            if self.eeg_run_mode == CUMULATIVE_EEG_MODE:
                self.eeg_history.start_clip(clip["id"],clip["playStartedAt"],clip["duration"])
                clip["eegHistoryStarted"] = True
            self.log("play", clip=index, clipId=clip.get("id"), decisionId=clip.get("decisionId"))
        if playing:
            self.collect_viewing_eeg(clip,wall)
        if (playing and video_t >= self.analyze_at(clip) and not clip.get("analysisStarted")
                and self.status == "running"):
            self.freeze_evidence(clip, wall)
            self.task = self.engine.spawn(self.adapt(index))

    def ended(self, index):
        if self.status in ("stopped", "failed"):
            return
        if 0 <= index < len(self.clips):
            clip = self.clips[index]
            now = time.time()
            if clip.get("ticks") and clip["ticks"][-1]["playing"] and 0 <= now-clip["ticks"][-1]["wall"] <= .75:
                self.collect_viewing_eeg(clip, min(now,clip.get("playStartedAt",now)+clip["duration"]))
            self.clips[index].setdefault("endedAt", now)
            if not self.clips[index].get("finalLogStarted"):
                self.clips[index]["finalLogStarted"] = True
                self.engine.spawn(self.finalize_signals(index))
            self.clips[index]["status"] = "watched"
            if not self.clips[index].get("analysisStarted") and self.status == "running":
                if self.playing is None or self.playing == index:
                    self.freeze_evidence(self.clips[index], self.clips[index]["endedAt"])
                    self.task = self.engine.spawn(self.adapt(index))
            elif index == self.max_scenes - 1 and self.clips[index].get("analysis"):
                self.status = "finished"
                self.set_stage(f"finished: {self.max_scenes} scenes watched")
                self.close_local_worker()
            self.cancel_detection(self.clips[index])

    async def finalize_signals(self, index):
        """Retain the entire watched scene and its delayed EEG response, in
        addition to the early evidence that already drove the next prompt."""
        clip = self.clips[index]
        ticks = list(clip["ticks"])
        ended_at = time.time()
        started_at = clip.get("playStartedAt", ended_at)
        await asyncio.sleep(fusion.EEG_LAG_S[1])
        gaze = self.gaze.window(started_at, ended_at)
        eeg = self.eeg.window(started_at, ended_at + fusion.EEG_LAG_S[1])
        self.dump(f"scene{index + 1}_complete_signals.json", dict(
            gaze=gaze, eeg=eeg, ticks=ticks,
            timeline=fusion.label(gaze, ticks, clip.get("track", [])),
            playbackEndedAt=ended_at, eegResponseUntil=ended_at + fusion.EEG_LAG_S[1],
            note="Decision analysis is saved separately at the early trigger; this includes the playback tail."))

    # -- the loop -------------------------------------------------------------
    async def start(self):
        decision = dict(id=str(uuid4()), observationWindow=None, focus=None, tension="same", dialogue="same", pacing="same", tone=None, event=False,
                        reasons=["Opening scene: no viewer data yet"])
        self.task = self.engine.spawn(self.prepare_tracking_and_begin(decision))

    async def prepare_tracking_and_begin(self, decision):
        # Startup is the only warm-up wait. Later decisions and transitions
        # reuse this process without awaiting tracking readiness or inference.
        if self.tracking_readiness["state"] == "not_started":
            from .local_tracker import prewarm_local_tracker
            provider = "color" if self.tracker == "color" else "opencv"
            def readiness(event):
                state = {"warming":"warming","warm_ready":"ready","warm_failed":"failed"}.get(event.get("diagnostic_event"))
                if state:
                    self.tracking_readiness = dict(event,state=state)
                    self.log("tracking_readiness",**deepcopy(self.tracking_readiness))
            self.set_stage("preparing local tracking")
            try:
                self.local_worker = await prewarm_local_tracker(provider,on_diagnostic=readiness)
            except asyncio.CancelledError:
                self.tracking_readiness["state"] = "cancelled"
                raise
            except Exception as error:
                self.tracking_readiness.update(state="failed",error=self.engine.error(error))
                self.warn("Local tracking warm-up",error)
        if self.status != "running":
            self.close_local_worker()
            return
        if self.opening_video:
            await self.predefined_scene(decision)
        else:
            await self.make_scene(decision, self.opening, seeds=None, changes=[])

    def close_local_worker(self):
        # Cancellation/native retirement owns its slot until cleanup finishes.
        # Request shutdown synchronously; never wait at playback boundaries.
        worker = self.local_worker
        if worker is None or self.local_worker_close_task is not None:
            return
        worker.request_close()
        for task in list(self.detection_tasks):
            task.cancel()
        self.tracking_readiness["state"] = "closed"
        self.local_worker_close_task = self.engine.spawn(worker.close())

    async def predefined_scene(self, decision):
        """Scene 1 is the uploaded episode segment; only the rest is generated."""
        try:
            import imageio_ffmpeg
            path = self.opening_video
            _, seconds = await asyncio.to_thread(imageio_ffmpeg.count_frames_and_secs, str(path))
            seconds = round(seconds, 2)
            if not 0.1 <= seconds <= 120:
                raise ValueError("Opening video must be between 0.1 and 120 seconds.")
            beats = director.parse_timeline(self.timeline, self.names, seconds)
            plan = dict(scene_title="Opening segment (predefined)", summary=self.story["premise"],
                        beats=beats or [dict(t0=0.0, t1=seconds, description="Opening", characters=list(self.names),
                                             dialogue=False, speaker=None, tags=[])],
                        video_prompt="(predefined clip, not generated)", change_note="The episode starts as written.")
            clip = dict(id=str(uuid4()), sessionId=self.id, decisionId=decision.get("id"), index=0, status="tracking", decision=decision, changes=[], plan=plan, writer="predefined",
                        duration=seconds, ticks=[], seeds=None, createdAt=time.time(), path=str(path))
            self.clips.append(clip)
            clip["decisionId"] = decision.get("id") or str(uuid4())
            clip["decisionTrace"] = decision_trace.create(self, clip)
            clip["decisionTrace"]["writer"]["outputReference"] = None
            clip["decisionTrace"]["generationStatus"] = "predefined"
            self.engine.trace_journal.schedule(clip["decisionTrace"])
            self.story["scenes"].append(dict(title=plan["scene_title"], summary=plan["summary"]))
            clip["track"] = []
            self.start_detection(clip, path, None, None, seconds)
            boundary = self.prepare_boundary(clip, path)
            if boundary:
                await asyncio.shield(boundary)
                if clip.get("boundaryFrameStatus") == "failed":
                    raise ValueError("Could not prepare the actual final frame for continuation.")
            if self.status != "running":
                return
            clip.update(url=f"/api/adaptive/clips/{self.id}/0", status="ready",
                        detected=sum(1 for f in clip["track"] if f["boxes"]))
            self.set_stage("scene 1 ready")
        except Exception as error:
            self.fail(error)

    def tracking_playback(self, clip):
        if not clip or not clip.get("ticks") or self.status != "running":
            return None
        tick = clip["ticks"][-1]
        # Convert epoch wall time to a monotonic age without changing capture time.
        return dict(media_t=tick["video_t"], at=time.monotonic()-(time.time()-tick["wall"]),
            playing=tick["playing"], current=self.current_clip() is clip,
            epoch=tick.get("epoch", 0), rate=tick.get("playbackRate", 1))

    async def track(self, path, seeds, focus, seconds=None, clip_id=None, on_progress=None):
        if self.engine.adapter.demo:
            return self.demo_track(focus, seconds or self.duration)
        from .local_tracker import detect_local
        clip = next((c for c in self.clips if c["id"] == clip_id), {})
        generation_id = clip.get("jobId") or clip.get("decisionId") or clip_id
        local, cloud = [], []
        jobs = []
        def publish():
            result = tracks.merge_tracking(local,cloud) if self.tracker == "fal" else list(local)
            if on_progress:
                on_progress(result)
            return result
        async def run_local():
            nonlocal local
            def progress(result):
                nonlocal local
                local = result
                publish()
            try:
                if self.tracker == "yoloe":
                    from .yoloe_detector import detect_yoloe
                    config = tracks.yoloe_config()
                    local = await detect_yoloe(path,self.target_names,clip_id=clip_id,session_id=self.id,
                        generation_id=generation_id,on_progress=progress,config=config)
                else:
                    if self.tracking_readiness["state"] in ("warming","failed","cancelled","closed"):
                        raise RuntimeError("Local tracking is unavailable for this run; no cold worker is started during playback.")
                    local = await detect_local(path,self.target_names,clip_id=clip_id,session_id=self.id,
                        generation_id=generation_id,on_progress=progress,
                        playback_state=lambda: self.tracking_playback(clip),
                        provider="color" if self.tracker == "color" else "opencv",warm_worker=self.local_worker)
            except Exception as error:
                self.warn("YOLOE" if self.tracker == "yoloe" else "Color/shape" if self.tracker == "color" else "OpenCV",error)
                clip["localTrackingError"] = self.engine.error(error)
        async def run_cloud():
            nonlocal cloud
            def progress(result):
                nonlocal cloud
                cloud = result
                publish()
            try:
                cloud = await tracks.detect_characters(path,self.story["characters"]+self.objects,self.engine.adapter.key,
                    directory=self.engine.directory,clip_id=clip_id,session_id=self.id,on_progress=progress,
                    observation_seconds=OBSERVATION_SECONDS)
                errors = [f for f in cloud if f.get("error")]
                if errors:
                    reasons = sorted({f.get("error_message",f["error"]) for f in errors})
                    self.warn("Florence",f"{len(errors)} frame queries failed: {'; '.join(reasons)} Unknown targets; gaze/playback continue.")
                self.dump(f"clip-{clip_id}_florence.json",cloud)
            except Exception as error:
                self.warn("Florence",error)
            finally:
                self.dump(f"clip-{clip_id}_florence.json",cloud)
        try:
            async with asyncio.timeout(90):
                if self.tracker == "people":
                    result = await tracks.track_clip(path,self.engine.directory,self.names,seeds)
                    return result
                jobs = [asyncio.create_task(run_local())]
                if self.tracker == "fal":
                    jobs.append(asyncio.create_task(run_cloud()))
                await asyncio.gather(*jobs)
            result = publish()
            self.dump(f"clip-{clip_id}_detections.json",result)
            return result
        except TimeoutError:
            self.warn("Tracking","Tracking deadline expired; available partial evidence retained.")
            return publish()
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            # This join belongs solely to the background tracking task.
            await asyncio.gather(*jobs,return_exceptions=True)

    def cancel_detection(self, clip):
        task = self.detection_by_clip.get(clip["id"])
        if task and not task.done():
            task.cancel()

    def start_detection(self, clip, path, seeds, focus, seconds):
        if self.engine.adapter.demo:
            clip["track"] = self.demo_track(focus, seconds)
            clip["detectionStatus"] = "synthetic"
            return
        self.cancel_detection(clip)
        clip["detectionStatus"] = "processing"
        clip["detectionProvider"] = self.tracker
        clip["detectionLifecycle"] = "active"
        generation_id = clip.get("jobId") or clip.get("decisionId") or clip["id"]
        clip["trackingGenerationId"] = generation_id
        diagnostic = tracking_diagnostics.for_clip(self,clip)
        diagnostic.record("tracking_started",provider=self.tracker)
        async def work():
            started = time.perf_counter()
            def publish(result):
                if (clip.get("trackingGenerationId") != generation_id
                        or self.status != "running" or not any(c is clip for c in self.clips)
                        or clip.get("endedAt") is not None
                        or (self.playing is not None and clip["index"] < self.playing)):
                    diagnostic.record("publication_rejected",reason="inactive_clip_or_generation",
                        records=len(result))
                    return
                rejected = sum(f.get("clip_id",clip["id"]) != clip["id"]
                    or f.get("session_id",self.id) not in (None,self.id)
                    or f.get("generation_id",generation_id) != generation_id for f in result)
                if rejected:
                    diagnostic.record("publication_rejected",reason="provenance_mismatch",records=rejected)
                clip["track"] = deepcopy([dict(f, generation_id=generation_id) for f in result
                    if f.get("clip_id", clip["id"]) == clip["id"]
                    and f.get("session_id", self.id) in (None, self.id)
                    and f.get("generation_id",generation_id) == generation_id])
                clip["detected"] = sum(bool(f["boxes"]) for f in clip["track"])
                diagnostic.record("publication_accepted",**tracking_diagnostics.publication_evidence(clip["track"]))
                timing = clip.setdefault("trackingTiming", {})
                if clip["track"]:
                    timing.setdefault("firstRecordMs", round((time.perf_counter()-started)*1000,3))
                    if clip["detected"]:
                        timing.setdefault("firstNamedMs", round((time.perf_counter()-started)*1000,3))
                    timing["throughMediaS"] = max(f["t"] for f in clip["track"])
                    latest = max(clip["track"], key=lambda f:f["t"])
                    timing["scheduling"] = deepcopy(latest.get("scheduling", {}))
            try:
                result = await self.track(path, seeds, focus, seconds, clip["id"], publish)
            except asyncio.CancelledError:
                if clip.get("trackingGenerationId") == generation_id:
                    clip["detectionLifecycle"] = "cancelled"
                    self.dump(f"clip-{clip['id']}_detections.json",clip.get("track",[]))
                raise
            if (clip.get("trackingGenerationId") != generation_id
                    or self.status != "running" or clip.get("endedAt") is not None or not any(c is clip for c in self.clips)
                    or (self.playing is not None and clip["index"] < self.playing)):
                return
            publish(result)
            clip.update(detectionLifecycle="complete",
                        detectionStatus="ready" if result else "unavailable", detectionMs=round((time.perf_counter()-started)*1000, 1))
            self.log("detections", clip=clip["index"], clipId=clip["id"], elapsedMs=clip["detectionMs"])
        task = self.engine.spawn(work())
        self.detection_tasks.add(task)
        self.detection_by_clip[clip["id"]] = task
        task.add_done_callback(self.detection_tasks.discard)

    async def adapt(self, index):
        try:
            clip = self.clips[index]
            if self.status != "running" or clip.get("adaptationClaimed"):
                return
            clip["adaptationClaimed"] = True
            source = (index, clip["id"])
            self.adaptation_source = source
            if not self.source_current(source):
                return
            if not clip.get("frozenEvidence"):
                self.freeze_evidence(clip, clip.get("endedAt", time.time()))
            frozen = clip["frozenEvidence"]
            decision_started = time.perf_counter()
            self.set_stage(f"analyzing scene {index + 1}")
            t0, t1 = frozen["start"], frozen["end"]
            samples, eeg, track, ticks = frozen["gaze"], frozen["eeg"], frozen["track"], frozen["ticks"]
            timeline = fusion.label(samples, ticks, track)
            confidence = frozen["quality"].get("confidence", 1.0 if self.eeg.source == "sim" else 0.0)
            analysis = fusion.analyze(timeline, eeg, track, self.names, clip.get("plan"), confidence,
                                      eeg_quality=frozen["quality"], eeg_window=(t0,t1))
            if self.eeg_run_mode == CUMULATIVE_EEG_MODE:
                analysis["eeg_policy"] = deepcopy(frozen["eegPolicy"])
            analysis["eeg_run_mode"] = self.eeg_run_mode
            analysis["detection_status_at_deadline"] = frozen["detectionStatus"]
            analysis["gaze_audit"] = gaze_audit.summarize(samples, ticks, timeline)
            # Story objects have dwell but never stand in for another character
            # when establishing a comparative character preference.
            object_names = [o["name"] for o in self.objects]
            analysis["objects"] = fusion.analyze(timeline, eeg, track, object_names,
                                                   clip.get("plan"), confidence,
                                                   eeg_quality=frozen["quality"], eeg_window=(t0,t1))["characters"] if object_names else {}
            if self.eeg.source == "mindmonitor":
                analysis.update(eeg_method="alpha-beta-relative-baseline", eeg_unit="index")
            clip.update(analysis=analysis, timeline=timeline)
            tracking_diagnostics.for_clip(self,clip).record("frozen_gaze_analysis",
                validGazeSeconds=analysis.get("valid_gaze_s"), confidence=analysis.get("gaze_confidence"),
                comparisonSeconds=analysis.get("comparison_s"), characters=analysis.get("characters"),
                gazeAudit=analysis.get("gaze_audit"))
            self.dump(f"scene{index + 1}_signals.json", dict(gaze=samples, eeg=eeg, ticks=ticks, track=track,
                                                              timeline=timeline, analysis=analysis))
            self.profile, changes = profiles.update(self.profile, analysis)
            clip["profileChanges"] = changes
            self.dump("profile.json", self.profile)
            self.log("profile", clip=index, changes=changes)
            if len(self.clips) >= self.max_scenes:
                self.set_stage(f"final scene: {self.max_scenes}-scene limit reached")
                if clip["status"] == "watched":
                    self.status = "finished"
                    self.close_local_worker()
                return
            self.story["viewer_analysis"] = analysis
            # Keep a local decision ready if the remote controller fails.
            decision = profiles.decide(self.profile, analysis)
            decision.update(id=str(uuid4()), observationWindow=dict(sessionId=self.id, clipId=clip["id"],
                start=t0, end=t1, mediaStart=min((e["video_t"] for e in timeline), default=None),
                mediaEnd=max((e["video_t"] for e in timeline), default=None), validSeconds=analysis.get("valid_gaze_s", 0)))
            decision["decisionMs"] = round((time.perf_counter()-decision_started)*1000, 2)
            continuation_started = time.perf_counter()
            extraction_started = time.perf_counter()
            image = None
            if not self.clip_bundles:
                local = self.local_media_tasks.get(clip["id"])
                if local:
                    await asyncio.shield(local)
                if not self.source_current(source):
                    return
                if not clip.get("path"):
                    raise ValueError("The local video is unavailable for actual final-frame continuation.")
                boundary = self.prepare_boundary(clip, clip["path"])
                if boundary:
                    await asyncio.shield(boundary)
                if not self.source_current(source):
                    return
                if clip.get("boundaryFrameStatus") == "failed":
                    raise ValueError("Could not prepare the actual final frame for continuation.")
                image = self.boundary_frames.get(clip["id"])
            extraction_ms = round((time.perf_counter()-extraction_started)*1000, 2)
            await self.make_scene(decision, image, seeds=None, changes=changes,
                                  frame_extraction_ms=extraction_ms, continuation_started=continuation_started,
                                  source=source)

        except Exception as error:
            self.fail(error)

    async def prepare_local_media(self, clip, job, decision, duration, continuation_started):
        """Independent local validation/tracking/boundary work for a streaming clip."""
        clip["localMediaStatus"] = "downloading"
        try:
            path = await self.engine.media_path(job)
            if self.status in ("stopped", "failed"):
                return None
            if self.engine.adapter.demo:
                await self.demo_render(path, decision.get("focus"), duration)
            clip.update(path=str(path),localMediaStatus="validated",mediaReadiness="fully_validated",fullyValidatedAt=time.time())
            self.log("full_media_ready",clip=clip["index"],clipId=clip["id"],jobId=job["id"],
                     playbackMode=self.playback_mode,fullyValidatedAt=clip["fullyValidatedAt"],
                     timing={k:v for k,v in job.items() if k.endswith("Ms")})
            if self.status == "running" and clip["status"] != "watched":
                self.start_detection(clip, path, None, decision.get("focus"), duration)
            boundary = self.prepare_boundary(clip, path)
            if boundary:
                await asyncio.shield(boundary)
                if clip.get("boundaryFrameStatus") == "failed":
                    raise ValueError("Could not prepare the actual final frame for continuation.")
            if self.status != "running":
                return None
            clip["boundaryReadyAt"] = time.time() if clip.get("boundaryFrameStatus") == "ready" else None
            self.engine.update(job,continuationReadyAt=time.time(),
                               continuationReadyMs=round((time.perf_counter()-continuation_started)*1000,2))
            return path
        except asyncio.CancelledError:
            clip["localMediaStatus"] = "cancelled"
            raise
        except Exception:
            clip["localMediaStatus"] = "failed"
            raise

    async def make_scene(self, decision, image, seeds, changes, *, frame_extraction_ms=0, continuation_started=None, source=None):
        if continuation_started is None:
            continuation_started = time.perf_counter()
        if self.generation_lock.locked():
            return  # One owner per continuation; duplicate triggers do not enqueue.
        async with self.generation_lock:
            await self._make_scene(decision, image, seeds, changes,
                                   frame_extraction_ms=frame_extraction_ms, continuation_started=continuation_started, source=source)

    async def _make_scene(self, decision, image, seeds, changes, *, frame_extraction_ms, continuation_started, source=None):
        index = len(self.clips)
        clip = None
        try:
            if not self.source_current(source) or index >= self.max_scenes:
                return
            if index and self.clips[-1]["status"] not in ("playing", "watched"):
                return
            bundle = self.clip_bundles[index] if self.clip_bundles else None
            duration = bundle["duration"] if bundle else self.duration
            self.story["next_prompt"] = bundle["prompt"] if bundle else director.continuation(self.story)
            self.story["next_summary"] = bundle["prompt"][:600] if bundle else self.story["current_event"]
            self.story["frame_constraints"] = {"first_frame": bool(image or (bundle or {}).get("firstFrame")), "end_frame": bool((bundle or {}).get("endFrame"))}
            self.set_stage(f"composing adaptation for scene {index + 1}")
            composed_at = time.perf_counter()
            source_decision = deepcopy(decision)
            try:
                if self.engine.adapter.demo:
                    plan, writer = director.template(self.story, decision, duration, self.names), "template"
                else:
                    plan, writer = await director.write_scene(self.story, self.profile, decision, duration)
            except Exception as error:
                self.warn("Controller", error)
                plan, writer = director.template(self.story, decision, duration, self.names), "template-fallback"
            decision = plan["decision"]
            decision.update(id=source_decision.get("id") or str(uuid4()), observationWindow=source_decision.get("observationWindow"))
            composition_ms = round((time.perf_counter()-composed_at)*1000, 2)
            if not self.source_current(source):
                return
            clip = dict(id=str(uuid4()), sessionId=self.id, decisionId=decision["id"], observationWindow=decision.get("observationWindow"),
                        compositionMs=composition_ms, frameExtractionMs=frame_extraction_ms,
                        decisionMs=source_decision.get("decisionMs", 0), index=index, status="generating", decision=decision, changes=changes, plan=plan,
                        writer=writer, duration=duration, ticks=[], seeds=seeds, createdAt=time.time(),
                        bundle=deepcopy(bundle), basePrompt=plan["base_prompt"])
            clip["decisionTrace"] = decision_trace.create(self, clip, source)
            self.clips.append(clip)
            self.engine.trace_journal.schedule(clip["decisionTrace"])
            self.engine.generation_guard()
            self.log("decision", clip=index, decision=decision, writer=writer, change_note=plan["change_note"])
            self.set_stage(f"generating scene {index + 1}")
            if bundle:
                options = generation_options(self.engine.clips.settings(bundle))
                options["prompt"] = plan["video_prompt"]
                images = {}  # Stored first/end frames win; never replace them with another clip's frame.
            else:
                options = dict(mode="frames" if image else "text", prompt=plan["video_prompt"], duration=duration, resolution=self.resolution)
                images = {"start": [image]} if image else {}
            job = self.engine.new_job(options, decisionTrace=clip["decisionTrace"], adaptiveSession=self.id, sceneIndex=index, playbackMode=self.playback_mode,
                                      clipId=bundle["id"] if bundle else None,
                                      sourceClipId=bundle["id"] if bundle else None,
                                      basePrompt=plan["base_prompt"], engagementDecision=decision, decisionId=decision["id"],
                                      observationWindow=decision.get("observationWindow"), adaptiveClipId=clip["id"],
                                      decisionModel=writer, decisionRequestId=plan.get("decisionRequestId"),
                                      compositionMs=composition_ms, frameExtractionMs=frame_extraction_ms,
                                      sourceSceneIndex=source[0] if source else None,
                                      sourceAdaptiveClipId=source[1] if source else None)
            clip["jobId"] = job["id"]
            window = decision.get("observationWindow")
            if window:
                self.engine.update(job, observationMs=max(0, (window["end"]-window["start"])*1000),
                    observationToGenerationMs=max(0, (time.time()-window["start"])*1000))
            await asyncio.shield(self.engine.spawn(self.engine.run_job(job, images)))
            if not self.source_current(source):
                return
            if job["status"] != "completed":
                raise ValueError(job.get("error", "Generation did not complete."))
            clip["track"] = []
            delivery = "download"
            path = None
            if self.playback_mode == "stream" and not self.engine.adapter.demo:
                from ..progressive_media import probe_video
                clip["localMediaStatus"] = "downloading"
                local = self.engine.spawn(self.prepare_local_media(clip,job,decision,duration,continuation_started))
                self.local_media_tasks[clip["id"]] = local
                def local_finished(task):
                    if task.cancelled(): return
                    error = task.exception()
                    if error and self.status == "running":
                        self.warn("Local media", error)
                        self.fail(error)
                local.add_done_callback(local_finished)
                probe_started = time.perf_counter()
                try:
                    clip["streamMetadata"] = await probe_video(job["video"]["url"])
                    delivery = "stream"
                    clip["streamReadyAt"] = time.time()
                    self.engine.update(job,streamReadyAt=clip["streamReadyAt"],
                                       streamProbeMs=round((time.perf_counter()-probe_started)*1000,3))
                except Exception as error:
                    clip["streamFallbackReason"] = self.engine.error(error)
                    self.log("stream_unavailable",clip=index,clipId=clip["id"],reason=clip["streamFallbackReason"])
                    path = await asyncio.shield(local)
            else:
                path = await self.prepare_local_media(clip,job,decision,duration,continuation_started)
            if not self.source_current(source):
                return
            continuation_ms = round((time.perf_counter()-continuation_started)*1000, 2)
            timing = dict(playbackReadyMs=continuation_ms,**({"continuationReadyMs":continuation_ms} if delivery == "download" else {}))
            if window:
                timing["observationToReadyMs"] = max(0, (time.time()-window["start"])*1000)
                if job.get("apiStartedAt") is not None:
                    timing["observationToSubmitMs"] = max(0, job["apiStartedAt"]-window["start"]*1000)
            self.engine.update(job, playbackReadyAt=time.time(), playbackDelivery=delivery, **(
                {"continuationReadyAt":time.time()} if delivery == "download" else {}), **timing)
            clip.update(url=f"/api/adaptive/clips/{self.id}/{index}/stream" if delivery == "stream" else f"/api/jobs/{job['id']}/video",
                        fallbackUrl=f"/api/adaptive/clips/{self.id}/{index}/validated" if delivery == "stream" else None,
                        playbackDelivery=delivery,mediaReadiness="progressive_available" if delivery == "stream" and clip.get("localMediaStatus") != "validated" else "fully_validated",
                        detectionStatus=clip.get("detectionStatus","awaiting_local_media"),status="ready",
                        playbackReadyMs=continuation_ms, continuationReadyMs=job.get("continuationReadyMs"), generatedS=round(continuation_ms/1000, 3), readyAt=time.time(),
                        generationInput=job.get("generationInput"),
                        timing={k:v for k,v in job.items() if k.endswith("Ms") or k in ("providerMetrics", "timings")},
                        detected=sum(1 for f in clip["track"] if f["boxes"]))
            self.readiness_samples.append(continuation_ms/1000)
            self.log("stream_ready" if delivery == "stream" else "media_ready", clip=index, clipId=clip["id"], decisionId=clip["decisionId"],
                     playbackMode=self.playback_mode,delivery=delivery,
                     readinessS=continuation_ms/1000, readinessScope="completed provider MP4 + early metadata/range gate; local validation pending" if delivery == "stream" else "decision-boundary wait + composition + validated media + next actual-end-frame preparation",
                     timing=clip["timing"])
            self.story["scenes"].append(dict(title=plan["scene_title"], summary=plan["summary"]))
            self.dump("story.json", self.story)
            self.set_stage(f"scene {index + 1} ready")
        except Exception as error:
            if not self.source_current(source):
                return
            if clip is not None and await self.fallback_clip(clip, error):
                return
            self.fail(error)

    async def fallback_clip(self, clip, error):
        """Keep the last displayed frame; never advance the story with a replay."""
        self.warn("Generation", error)
        if clip.get("decisionTrace"):
            clip["decisionTrace"]["generationStatus"] = "failed"
            self.engine.trace_journal.schedule(clip["decisionTrace"])
        clip.update(status="failed", fallback=True, track=[],
                    fallbackReason="Generation failed. Holding the previous frame; this adaptation was not generated.")
        self.fail(error)
        return True

    def demo_boxes(self, focus):
        """Demo mode only: one box per character (focus = big, centred)."""
        n = len(self.target_names)
        boxes = {}
        for i, name in enumerate(self.target_names):
            if name == focus:
                boxes[name] = [0.33, 0.12, 0.67, 0.98]
            else:
                slot = (i + 0.5) / n
                w = 0.18 if focus else 0.24
                boxes[name] = [max(0.02, slot - w / 2), 0.35 if focus else 0.2, min(0.98, slot + w / 2), 0.95]
        return boxes

    def demo_track(self, focus, seconds):
        boxes = self.demo_boxes(focus)
        return [dict(t=round(i / tracks.SAMPLE_FPS, 3), boxes=boxes) for i in range(int(seconds * tracks.SAMPLE_FPS) + 1)]

    async def demo_render(self, path, focus, duration=None):
        filters = ",".join(f"drawbox=x=iw*{b[0]}:y=ih*{b[1]}:w=iw*{b[2] - b[0]}:h=ih*{b[3] - b[1]}:color={DEMO_COLORS[i % 4]}:t=fill"
                           for i, b in enumerate(self.demo_boxes(focus).values()))
        temporary = Path(str(path) + ".demo.mp4")
        await ffmpeg("-f", "lavfi", "-i", "color=c=0x1d2330:s=640x360:r=24", "-t", str(duration or self.duration), "-vf", filters,
                     "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary)
        os.replace(temporary, path)

    def fail(self, error):
        if self.status == "stopped":
            return
        self.error = self.engine.error(error)
        self.status = "failed"
        self.set_stage("stopped")
        self.close_local_worker()

    def stop(self):
        if self.status == "running":
            self.status = "stopped"
            self.set_stage("stopped by viewer")
            self.close_local_worker()
            if self.task and not self.task.done():
                self.task.cancel()
            for clip in self.clips:
                if clip.get("localMediaStatus") == "downloading": clip["localMediaStatus"] = "cancelled"
            for task in list(self.detection_tasks) + list(self.frame_tasks.values()) + list(self.local_media_tasks.values()):
                task.cancel()
            for clip in self.clips:
                job = self.engine.jobs.get(clip.get("jobId"))
                if job and job["status"] not in ("completed", "failed", "cancelled"):
                    self.engine.spawn(self.engine.cancel_job(job))

    def dump(self, name, data):
        path = self.dir/name
        path.write_text(json.dumps(data))
        path.chmod(0o600)
