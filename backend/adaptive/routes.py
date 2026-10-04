"""HTTP surface of the adaptive loop (/api/adaptive/*) + sensor lifecycle."""
import asyncio
from copy import deepcopy
import json
import math
import os
import threading
import time

from uuid import uuid4

from fastapi import Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.datastructures import UploadFile

from .sensors import EegFeed, GazeFeed, Simulator, run_muse, GAZE_PORT
from .mindmonitor import MindMonitorFeed, run_mindmonitor
from .session import AdaptiveSession, MAX_SCENES
from . import tracks
from .eeg_calibration import calibration_status
from ..fal_adapter import FalError
from ..sensor_setup import SensorSetup

MAX_CHARACTERS = 4
MAX_OPENING_BYTES = 100 * 1024 * 1024
# The setup form has 10 text fields; allow the optional scene limit and EEG alias
# with bounded headroom. Only the two supported opening file fields are allowed.
MAX_SESSION_FIELDS = 16
MAX_SESSION_FILES = 2


class Sensors:
    def __init__(self, directory=None, demo=False):
        self.eeg_mode = os.getenv("GOZ_EEG", "mindmonitor")
        self.gaze, self.eeg = GazeFeed(), MindMonitorFeed() if self.eeg_mode == "mindmonitor" else EegFeed()
        self.stop_event = threading.Event()
        self.sim_task, self.session = None, None
        self.gaze_mode = os.getenv("GOZ_GAZE", "gazekit")
        self.gaze_error = None
        self.muse_thread = None
        self.muse_reader_stop = None
        self.setup = SensorSetup(self, directory or "data", demo)

    def start_muse_reader(self):
        if self.eeg_mode == "muse" and (not self.muse_thread or not self.muse_thread.is_alive()):
            self.muse_reader_stop = threading.Event()
            self.muse_thread = threading.Thread(target=run_muse, args=(self.eeg, self.muse_reader_stop), daemon=True)
            self.muse_thread.start()
        elif self.eeg_mode == "mindmonitor" and (not self.muse_thread or not self.muse_thread.is_alive()):
            reader = run_mindmonitor if self.eeg_mode == "mindmonitor" else run_muse
            self.muse_thread = threading.Thread(target=reader, args=(self.eeg, self.stop_event), daemon=True)
            self.muse_thread.start()

    async def start(self):
        if self.gaze_mode == "gazekit":
            try:
                await self.gaze.listen(int(os.getenv("GOZ_GAZE_PORT", GAZE_PORT)))
                self.gaze.source = "waiting for `gazekit stream`"
            except OSError as error:
                self.gaze_error = f"UDP port busy: {error}"
        if self.eeg_mode == "mindmonitor":
            self.start_muse_reader()
        simulate_gaze, simulate_eeg = self.gaze_mode == "sim", self.eeg_mode == "sim"
        if simulate_gaze or simulate_eeg:
            favourite = int(os.getenv("GOZ_SIM_FAVORITE", "1"))
            bias = float(os.getenv("GOZ_SIM_BIAS", "0.7"))
            proxy = self

            class Live:
                def live_boxes(self):
                    return proxy.session.live_boxes() if proxy.session else ({}, None)

                def live_target(self):
                    return proxy.session.live_target() if proxy.session else None

            self.sim_task = asyncio.create_task(Simulator(Live(), self.gaze, self.eeg, favourite, bias).run(simulate_gaze, simulate_eeg))
        await self.setup.start()

    async def connect_muse(self):
        if self.eeg_mode != "muse":
            raise ValueError("Manual Muse connection is available when GOZ_EEG=muse.")
        await self.setup.connect_muse()

    async def disconnect_muse(self):
        if self.eeg_mode == "muse":
            await self.setup.disconnect_muse()
            if self.muse_reader_stop:
                self.muse_reader_stop.set()
            if self.muse_thread:
                await asyncio.to_thread(self.muse_thread.join, 2)
                self.muse_thread = None

    async def close(self):
        self.stop_event.set()
        if self.session:
            self.session.stop()
        await self.setup.close()
        self.gaze.close()
        if self.muse_reader_stop:
            self.muse_reader_stop.set()
        if self.sim_task:
            self.sim_task.cancel()
            await asyncio.gather(self.sim_task, return_exceptions=True)
        if self.muse_thread:
            await asyncio.to_thread(self.muse_thread.join, 5)


def number(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError("Invalid number.")
    return float(value)


def register(app, json_body, images, multipart, duration_value, resolution_value):
    def sensors(request):
        return request.app.state.sensors

    def session(request):
        s = sensors(request).session
        if s is None:
            raise ValueError("No adaptive session.")
        return s

    def guided_calibration(sn):
        quality = deepcopy(sn.eeg.status())
        quality["calibration"] = deepcopy(getattr(sn.eeg,"calibration",None) or {})
        status = calibration_status(quality)
        status["gazeOnly"] = bool(getattr(sn,"eeg_gaze_only",False))
        return status

    @app.get("/api/adaptive/muse/calibration")
    async def muse_calibration_status(request: Request):
        return guided_calibration(sensors(request))

    @app.post("/api/adaptive/muse/calibration")
    async def muse_calibration_action(request: Request):
        body = await json_body(request)
        action = body.get("action")
        if action not in ("start_or_recalibrate","gaze_only"):
            raise ValueError("Choose start_or_recalibrate or gaze_only.")
        sn,e = sensors(request),request.app.state.engine
        async with e.lock:
            if e.busy() or sn.session and sn.session.status == "running":
                raise FalError("Stop or finish the current run before changing EEG calibration or gaze-only mode.",409)
            if action == "gaze_only":
                sn.eeg_gaze_only = True
                return guided_calibration(sn)
            quality = sn.eeg.status()
            if quality.get("source") != "muse":
                raise FalError("Guided 60-second calibration requires direct Muse EEG. Gaze-only remains available.",409)
            if quality.get("live") is not True or not quality.get("deviceId"):
                raise FalError("Connect Muse and wait for live samples before starting calibration.",409)
            sn.eeg_gaze_only = False
            requested = getattr(sn,"eeg_guided_calibration_started_at",None)
            if requested is not None and requested == sn.eeg.calibration_started and not quality.get("calibrated"):
                status = guided_calibration(sn)
                status["alreadyInProgress"] = True
                return status
            sn.eeg.begin_calibration(60.0)
            sn.eeg_guided_calibration_started_at = sn.eeg.calibration_started
            return guided_calibration(sn)

    @app.get("/api/adaptive/decision-traces")
    async def decision_traces(request: Request, session_id: str | None = None):
        journal = request.app.state.engine.trace_journal
        records = await asyncio.to_thread(journal.read, session_id)
        return dict(records=records, warning=journal.error)

    @app.get("/api/adaptive/tracking-diagnostics")
    async def tracking_diagnostics_read(request: Request, session_id: str):
        from . import tracking_diagnostics
        records = await asyncio.to_thread(tracking_diagnostics.read, request.app.state.engine.directory, session_id)
        return dict(sessionId=session_id,records=records)

    @app.post("/api/adaptive/overlay-diagnostic")
    async def overlay_diagnostic(request: Request):
        from . import tracking_diagnostics
        body = await json_body(request)
        s = session(request)
        if body.get("session_id") != s.id:
            raise ValueError("Overlay diagnostic belongs to an older session.")
        clip = next((c for c in s.clips if c["id"] == body.get("clip_id")),None)
        if clip is None:
            raise ValueError("Overlay clip identity mismatch.")
        wall = number(body.get("wall"),0,1e13)/1000
        if abs(wall-time.time()) > 5:
            raise ValueError("Overlay diagnostic clock is stale.")
        now = time.monotonic()
        if now-clip.get("lastOverlayDiagnosticAt",-10) < .8:
            return dict(ok=True,rateLimited=True)
        evidence = tracking_diagnostics.overlay_evidence(body.get("diagnostic"),s.target_names)
        clip["lastOverlayDiagnosticAt"] = now
        tracking_diagnostics.for_clip(s,clip).record("overlay_report", browserWall=wall,**evidence)
        return dict(ok=True)

    @app.post("/api/adaptive/sessions", status_code=202)
    async def start(request: Request):
        e, sn = request.app.state.engine, sensors(request)
        sn.setup.require_ready()
        form = await multipart(request, {"start", "opening"},
                               max_files=MAX_SESSION_FILES, max_fields=MAX_SESSION_FIELDS)
        try:
            premise = str(form.get("premise", "")).strip()
            if not premise or len(premise) > 2000:
                raise ValueError("Describe the story premise (up to 2,000 characters).")
            characters = json.loads(str(form.get("characters", "[]")))
            if (not isinstance(characters, list) or not 2 <= len(characters) <= MAX_CHARACTERS
                    or any(not isinstance(c, dict) or not str(c.get("name", "")).strip() for c in characters)):
                raise ValueError("Name 2–4 characters.")
            characters = [dict(name=str(c["name"]).strip()[:40], description=str(c.get("description", "")).strip()[:200])
                          for c in characters]
            if len({c["name"] for c in characters}) != len(characters):
                raise ValueError("Character names must be unique.")
            objects = json.loads(str(form.get("objects", "[]")))
            if (not isinstance(objects, list) or len(objects) > 2 or any(not isinstance(o, dict)
                    or not isinstance(o.get("name"), str) or not o["name"].strip() for o in objects)):
                raise ValueError("Use at most two named story objects.")
            objects = [dict(name=o["name"].strip()[:40], description=str(o.get("description", ""))[:200]) for o in objects]
            names = [c["name"] for c in characters + objects]
            if len(set(names)) != len(names):
                raise ValueError("Character and object names must be unique.")
            from .tracks import tracking_provider
            tracker = tracking_provider(str(form["tracker"]) if "tracker" in form else None)
            if tracker == "yoloe":
                availability = tracks.yoloe_availability()
                if not availability["available"]:
                    raise ValueError("YOLOE unavailable: " + availability["reason"])
            eeg_run_mode = str(form.get("eegRunMode", form.get("eeg_run_mode", "cumulative_prior_clips")))
            if eeg_run_mode not in ("cumulative_prior_clips", "baseline"):
                raise ValueError("Choose prior played clips or calibrated baseline EEG comparison.")
            playback_mode = str(form.get("playback_mode", "download"))
            if playback_mode not in ("download", "stream"):
                raise ValueError("Choose download or stream playback.")
            timeline = str(form.get("timeline", ""))[:20000]
            use_sequence = str(form.get("use_saved_sequence", "0")) == "1"
            bundles = None
            if use_sequence:
                records = e.clips.library()["clipDefinitions"]
                bundles = e.bundles.validate([dict(id=c["id"], order=i,
                                                **e.clips.settings(c).model_dump(mode="json"))
                                             for i, c in enumerate(records)])
            video = form.get("opening")
            if bundles and form.get("start"):
                raise ValueError("Saved clips use their own frame pairs. Upload an opening video, or uncheck the saved-sequence option to use a custom opening frame.")
            frames = [] if video else await images(form, "start", 1)
            if not frames and not video and not bundles:
                raise ValueError("Upload the opening episode clip (or an opening frame).")
            opening_video = None
            if video:
                if not isinstance(video, UploadFile) or not (video.content_type or "").startswith("video/"):
                    raise ValueError("The opening clip must be a video file (MP4, MOV or WebM).")
                data = await video.read(MAX_OPENING_BYTES + 1)
                if not data or len(data) > MAX_OPENING_BYTES:
                    raise ValueError("The opening clip must be at most 100 MB.")
                uploads = e.directory/"adaptive"/"uploads"
                uploads.mkdir(parents=True, exist_ok=True)
                opening_video = uploads/f"{uuid4()}.mp4"
                opening_video.write_bytes(data)
                opening_video.chmod(0o600)
            available_limit = min(MAX_SCENES, len(bundles)) if bundles else MAX_SCENES
            scene_limit = int(number(int(str(form.get("scene_limit", available_limit))), 1, available_limit))
            async with e.lock:
                if not e.adapter.configured():
                    raise ValueError("Add your Fal key first.")
                if e.busy() or (sn.session and sn.session.status == "running"):
                    raise ValueError("A generation is already running. Stop it first.")
                sn.session = AdaptiveSession(e, sn.gaze, sn.eeg, e.directory, premise, characters,
                                             duration_value(form.get("duration", 10)),
                                             resolution_value(form.get("resolution", "480P")),
                                             frames[0] if frames else None, opening_video, timeline, bundles, objects, tracker=tracker, playback_mode=playback_mode,
                                             eeg_run_mode=eeg_run_mode,eeg_gaze_only=bool(getattr(sn,"eeg_gaze_only",False)))
                sn.session.max_scenes = scene_limit
                await sn.session.start()
            return sn.session.public()
        finally:
            await form.close()

    @app.get("/api/adaptive/clips/{session_id}/{index}")
    async def clip_file(request: Request, session_id: str, index: int):
        s = session(request)
        if s.id != session_id or not 0 <= index < len(s.clips) or not s.clips[index].get("path"):
            raise ValueError("Clip not found.")
        return FileResponse(s.clips[index]["path"], media_type="video/mp4")

    @app.get("/api/adaptive/clips/{session_id}/{index}/validated")
    async def validated_clip(request: Request, session_id: str, index: int):
        s = session(request)
        if s.id != session_id or s.status == "stopped" or not 0 <= index < len(s.clips):
            raise ValueError("Clip not found.")
        clip = s.clips[index]
        local = s.local_media_tasks.get(clip["id"])
        if local:
            await asyncio.shield(local)
        if s.status == "stopped" or clip.get("localMediaStatus") != "validated" or not clip.get("path"):
            raise ValueError("Validated fallback media is unavailable.")
        return FileResponse(clip["path"],media_type="video/mp4")

    @app.get("/api/adaptive/clips/{session_id}/{index}/stream")
    async def stream_clip(request: Request, session_id: str, index: int):
        from .. import progressive_media
        s = session(request)
        if s.id != session_id or s.status == "stopped" or not 0 <= index < len(s.clips):
            raise ValueError("Clip not found.")
        clip = s.clips[index]
        job = s.engine.jobs.get(clip.get("jobId"))
        metadata = clip.get("streamMetadata")
        if clip.get("playbackDelivery") != "stream" or not job or job["status"] != "completed" or not metadata:
            raise ValueError("Completed stream is unavailable.")
        total = metadata["bytes"]
        byte_range = request.headers.get("range")
        start,end = progressive_media.range_bounds(byte_range,total)
        client,response = await progressive_media.open_video(job["video"]["url"],
            f"bytes={start}-{end}" if byte_range else None,transport=getattr(s.engine,"stream_transport",None))
        try:
            progressive_media.validate_range(response,start,end,total,requested=byte_range is not None)
        except BaseException:
            await response.aclose(); await client.aclose()
            raise
        async def content():
            read = 0
            try:
                async with asyncio.timeout(120):
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if s.status == "stopped": return
                        read += len(chunk)
                        if read > end-start+1:
                            raise ValueError("Completed stream exceeded its expected byte range.")
                        yield chunk
                    if read != end-start+1:
                        raise ValueError("Completed stream ended before its expected byte range.")
            except Exception as error:
                count = clip.get("streamTransferErrors",0)
                if count < 4:
                    clip["streamTransferErrors"] = count+1
                    s.log("stream_transfer_failed",clip=index,clipId=clip["id"],errorType=type(error).__name__)
                raise
            finally:
                await response.aclose(); await client.aclose()
        headers={"Accept-Ranges":"bytes","Content-Length":str(end-start+1)}
        if response.status_code == 206: headers["Content-Range"]=f"bytes {start}-{end}/{total}"
        return StreamingResponse(content(),status_code=response.status_code,headers=headers,media_type="video/mp4")

    @app.post("/api/adaptive/tick")
    async def tick(request: Request):
        body = await json_body(request)
        if body.get("session_id") != session(request).id:
            raise ValueError("Playback belongs to an older session. Reload the page.")
        rect = body.get("rect")
        if rect is not None:
            rect = {k: number(rect.get(k), -1e5, 1e5) for k in ("x", "y", "w", "h")}
        visible = body.get("visible_rect")
        if visible is not None:
            visible = {k: number(visible.get(k), -1e5, 1e5) for k in ("x", "y", "w", "h")}
        mapping = body.get("mapping")
        if mapping is not None:
            if not isinstance(mapping, dict) or type(mapping.get("valid")) is not bool:
                raise ValueError("Mapping validity must be a boolean.")
            mapping = dict(valid=mapping["valid"], **{k: str(mapping[k])[:120] if mapping.get(k) is not None else None
                for k in ("method", "coordinateSpace", "reason")})
            for key in ("scale", "devicePixelRatio", "screenWidth", "screenHeight"):
                value = body["mapping"].get(key)
                mapping[key] = number(value, 0, 1e5) if value is not None else None
        if type(body.get("playing")) is not bool:
            raise ValueError("Playing must be a boolean.")
        session(request).tick(int(number(body.get("clip"), 0, 100)), number(body.get("video_t"), 0, 600),
                              body["playing"], rect, number(body.get("wall"), 0, 1e13) / 1000,
                              clip_id=body.get("clip_id"), epoch=int(number(body.get("epoch", 0), 0, 1e10)),
                              playback_rate=number(body.get("playback_rate", 1), .1, 4), visible_rect=visible,
                              mapping=mapping)
        return {"ok": True}

    @app.post("/api/adaptive/ended")
    async def ended(request: Request):
        body = await json_body(request)
        if body.get("session_id") != session(request).id:
            raise ValueError("Playback belongs to an older session. Reload the page.")
        session(request).ended(int(number(body.get("clip"), 0, 100)))
        return {"ok": True}

    @app.post("/api/adaptive/stop")
    async def stop(request: Request):
        s = session(request)
        body = await json_body(request)
        if body.get("session_id") != s.id:
            raise ValueError("Stop belongs to an older session.")
        s.stop()
        return s.public()

    @app.post("/api/adaptive/playback-event")
    async def playback_event(request: Request):
        body = await json_body(request)
        s = session(request)
        if body.get("session_id") != s.id or s.status == "stopped":
            raise ValueError("Playback event belongs to an inactive session.")
        index = int(number(body.get("clip"), 0, 100))
        if index >= len(s.clips) or body.get("clip_id") != s.clips[index]["id"]:
            raise ValueError("Playback clip identity mismatch.")
        kind = body.get("kind")
        if kind not in ("ready", "playing", "ended", "waiting", "error", "diagnostic"):
            raise ValueError("Invalid playback event.")
        wall = number(body.get("wall"), 0, 1e13)/1000
        if abs(wall-time.time()) > 5:
            raise ValueError("Playback event clock is stale.")
        clip = s.clips[index]
        fields = dict(clip=index, clipId=clip["id"], decisionId=clip.get("decisionId"), wall=wall)
        if kind == "diagnostic":
            diagnostic = body.get("diagnostic")
            if not isinstance(diagnostic, dict):
                raise ValueError("Invalid transition diagnostic.")
            phase = diagnostic.get("phase")
            if phase not in ("preload", "loadeddata", "canplay", "ended", "state", "play_requested", "playing", "play_rejected", "media_error", "waiting", "fallback_requested", "fallback_playing"):
                raise ValueError("Invalid transition phase.")
            def integer(value, lo, hi):
                result = number(value, lo, hi)
                if result != int(result):
                    raise ValueError("Invalid transition index.")
                return int(result)
            destination_index = integer(diagnostic.get("index"), 0, len(s.clips))
            source_index = diagnostic.get("previousIndex")
            source_index = integer(source_index, 0, len(s.clips)-1) if source_index is not None else None
            source = s.clips[source_index] if source_index is not None else None
            target = s.clips[destination_index] if destination_index < len(s.clips) else None
            source_id, target_id = source.get("id") if source else None, target.get("id") if target else None
            if diagnostic.get("sourceClipId") != source_id or diagnostic.get("targetClipId") not in (None, target_id):
                raise ValueError("Transition clip identity mismatch.")
            if index not in (source_index, destination_index):
                raise ValueError("Transition anchor identity mismatch.")
            epoch = integer(diagnostic.get("epoch"), 0, 1_000_000)
            reason = diagnostic.get("reason")
            if reason is not None and reason not in ("playing", "loading", "buffering", "finished", "cancelled", "stopped", "gesture", "error"):
                raise ValueError("Invalid transition state.")
            values = {}
            for key, upper in (("elapsedMs", 86_400_000), ("holdMs", 86_400_000), ("videoTime", 3600),
                               ("readyState", 4), ("networkState", 3), ("mediaErrorCode", 4)):
                value = diagnostic.get(key)
                values[key] = number(value, 0, upper) if value is not None else None
            error_name = diagnostic.get("errorName")
            if error_name is not None and (not isinstance(error_name, str) or len(error_name) > 80 or not error_name.isidentifier()):
                raise ValueError("Invalid media error name.")
            # Bound both memory and persistent rows across client resets/retries.
            budgets = getattr(s, "_transition_diagnostics", None)
            if budgets is None:
                s._transition_diagnostics = budgets = {}
            seen = budgets.setdefault(destination_index, set())
            key = (epoch, phase, reason)
            if key in seen or len(seen) >= 24:
                return {"ok": True, "recorded": False}
            seen.add(key)
            s.log("browser_transition", **fields, phase=phase, reason=reason, epoch=epoch,
                  sourceIndex=source_index, sourceClipId=source_id,
                  destinationIndex=destination_index, destinationClipId=target_id,
                  clientDestinationKnown=diagnostic.get("targetClipId") is not None,
                  errorName=error_name, **values)
            return {"ok": True, "recorded": True}
        if kind == "playing" and not clip.get("firstPresentedAt"):
            clip["firstPresentedAt"] = wall
            if job := s.engine.jobs.get(clip.get("jobId")):
                s.engine.update(job,firstPresentedAt=wall,firstPresentationElapsedMs=max(0,wall*1000-job["startedAt"]),
                                firstPresentationDelivery=clip.get("playbackDelivery","download"))
            prev = s.clips[index-1] if index else None
            clip["transitionMs"] = max(0, (wall-prev["endedAt"])*1000) if prev and prev.get("endedAt") else None
            fields["transitionMs"] = clip["transitionMs"]
            window = clip.get("observationWindow")
            clip["feedbackDelayS"] = wall-window["end"] if window else None
        elif kind == "ended":
            clip["endedAt"] = min(clip.get("endedAt", wall), wall)
            if index+1 < len(s.clips):
                successor = s.clips[index+1]
                if successor.get("firstPresentedAt"):
                    successor["transitionMs"] = max(0, (successor["firstPresentedAt"]-clip["endedAt"])*1000)
            s.ended(index)
        s.log("browser_"+kind, **fields)
        return {"ok": True}

    @app.get("/api/adaptive/state")
    async def state(request: Request):
        sn = sensors(request)
        s = sn.session
        now = time.time()
        eeg = [dict(t=round(t - now, 2), z=round(z, 2), artifact=a) for (t, _, z, a) in sn.eeg.window(now - 30, now)]
        recent = sn.gaze.window(now - 30, now)
        blinks = sum(1 for a, b in zip(recent, recent[1:]) if b.get("blink") and not a.get("blink"))
        latest = sn.gaze.latest()
        response = None
        if s and s.current_clip():
            clip = s.current_clip()
            # Limit dashboard work to recent data; final analysis uses the full scene.
            t0 = max(now - 15, clip.get("playStartedAt", now))
            from . import fusion
            timeline = fusion.label(sn.gaze.window(t0, now), clip["ticks"][-160:], clip.get("track", []))
            response = fusion.analyze(timeline, sn.eeg.window(t0, now), clip.get("track", []), s.target_names,
                                      eeg_confidence=sn.eeg.status().get("confidence", 0))
        return dict(
            session=s.public() if s else None,
            trackerAvailability=dict(yoloe=tracks.yoloe_availability()),
            response=response,
            setup=sn.setup.snapshot(),
            gaze=dict(**sn.gaze.status(), error=sn.gaze_error, point=s.live_gaze() if s else None,
                      blinks_per_min=round(2 * blinks, 1) if recent else None,
                      yaw=latest.get("yaw") if latest else None, pitch=latest.get("pitch") if latest else None),
            eeg=dict(**sn.eeg.status(), series=eeg[-120:]),
            simulation=sn.gaze_mode == "sim" or sn.eeg_mode == "sim",
            serverNow=now)

    @app.post("/api/adaptive/muse/connect")
    async def connect_muse(request: Request):
        await sensors(request).connect_muse()
        return sensors(request).setup.snapshot()

    @app.post("/api/adaptive/muse/disconnect")
    async def disconnect_muse(request: Request):
        await sensors(request).disconnect_muse()
        return sensors(request).setup.snapshot()
