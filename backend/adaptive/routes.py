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
from .session import AdaptiveSession
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
        self.setup = SensorSetup(self, directory or "data", demo)

    def start_muse_reader(self):
        if self.eeg_mode in ("muse", "mindmonitor") and (not self.muse_thread or not self.muse_thread.is_alive()):
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
        if not self.setup.required or self.eeg_mode == "mindmonitor":
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

    async def close(self):
        self.stop_event.set()
        await self.setup.close()
        self.gaze.close()
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
            async with e.lock:
                if not e.adapter.configured():
                    raise ValueError("Add your Fal key first.")
                if e.busy() or (sn.session and sn.session.status == "running"):
                    raise ValueError("A generation is already running. Stop it first.")
                sn.session = AdaptiveSession(e, sn.gaze, sn.eeg, e.directory, premise, characters,
                                             duration_value(form.get("duration", 10)),
                                             resolution_value(form.get("resolution", "480P")),
                                             frames[0] if frames else None, opening_video, timeline, bundles)
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
        rect = body.get("rect")
        if rect is not None:
            rect = {k: number(rect.get(k), -1e5, 1e5) for k in ("x", "y", "w", "h")}
        session(request).tick(int(number(body.get("clip"), 0, 100)), number(body.get("video_t"), 0, 600),
                              bool(body.get("playing")), rect, number(body.get("wall"), 0, 1e13) / 1000)
        return {"ok": True}

    @app.post("/api/adaptive/ended")
    async def ended(request: Request):
        session(request).ended(int(number((await json_body(request)).get("clip"), 0, 100)))
        return {"ok": True}

    @app.post("/api/adaptive/stop")
    async def stop(request: Request):
        s = session(request)
        s.stop()
        return s.public()

    @app.get("/api/adaptive/state")
    async def state(request: Request):
        sn = sensors(request)
        s = sn.session
        now = time.time()
        eeg = [dict(t=round(t - now, 2), z=round(z, 2), artifact=a) for (t, _, z, a) in sn.eeg.window(now - 30, now)]
        recent = sn.gaze.window(now - 30, now)
        blinks = sum(1 for a, b in zip(recent, recent[1:]) if b.get("blink") and not a.get("blink"))
        latest = sn.gaze.latest()
        return dict(
            session=s.public() if s else None,
            setup=sn.setup.snapshot(),
            gaze=dict(**sn.gaze.status(), error=sn.gaze_error, point=s.live_gaze() if s else None,
                      blinks_per_min=round(2 * blinks, 1) if recent else None,
                      yaw=latest.get("yaw") if latest else None, pitch=latest.get("pitch") if latest else None),
            eeg=dict(**sn.eeg.status(), series=eeg[-120:]),
            serverNow=now)
