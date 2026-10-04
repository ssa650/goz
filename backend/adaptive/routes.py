"""HTTP surface of the adaptive loop (/api/adaptive/*) + sensor lifecycle."""
import asyncio
import json
import math
import os
import threading
import time

from uuid import uuid4

from fastapi import Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from .sensors import EegFeed, GazeFeed, Simulator, run_muse, GAZE_PORT
from .mindmonitor import MindMonitorFeed, run_mindmonitor
from .session import AdaptiveSession, MAX_SCENES
from . import tracks
from ..sensor_setup import SensorSetup

MAX_CHARACTERS = 4
MAX_OPENING_BYTES = 100 * 1024 * 1024


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

    @app.get("/api/adaptive/decision-traces")
    async def decision_traces(request: Request, session_id: str | None = None):
        journal = request.app.state.engine.trace_journal
        records = await asyncio.to_thread(journal.read, session_id)
        return dict(records=records, warning=journal.error)

    @app.post("/api/adaptive/sessions", status_code=202)
    async def start(request: Request):
        e, sn = request.app.state.engine, sensors(request)
        sn.setup.require_ready()
        form = await multipart(request, {"start", "opening"})
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
                                             frames[0] if frames else None, opening_video, timeline, bundles, objects, tracker=tracker)
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
        if kind not in ("ready", "playing", "ended", "waiting", "error"):
            raise ValueError("Invalid playback event.")
        wall = number(body.get("wall"), 0, 1e13)/1000
        if abs(wall-time.time()) > 5:
            raise ValueError("Playback event clock is stale.")
        clip = s.clips[index]
        fields = dict(clip=index, clipId=clip["id"], decisionId=clip.get("decisionId"), wall=wall)
        if kind == "playing" and not clip.get("firstPresentedAt"):
            clip["firstPresentedAt"] = wall
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
