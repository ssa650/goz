"""FastAPI serves the JavaScript player and owns every provider interaction."""
import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
from uuid import UUID
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from .config import (MODELS, DURATION, DURATIONS, MAX_IMAGE_BYTES, MAX_CHARACTERS,
                     MAX_CLIPS, MAX_PROMPT_BYTES, POLL_SECONDS)
from .engine import Engine, JOB_TERMINAL
from .fal_adapter import FalAdapter, FalError
from .frames import verify_image
from .prompts import plan

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT/".env")


def duration_value(value):
    if isinstance(value, bool):
        raise ValueError("Choose a whole-number duration from 5 to 15 seconds.")
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        raise ValueError("Choose a whole-number duration from 5 to 15 seconds.") from None
    if number not in DURATIONS:
        raise ValueError("Choose a whole-number duration from 5 to 15 seconds.")
    return number


def resolution_value(value):
    if value not in ("480P", "768P", "1080P"):
        raise ValueError("Choose a supported resolution.")
    return value


async def json_body(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_PROMPT_BYTES+12000:
            raise FalError("Request body is too large.", 413)
        chunks.append(chunk)
    data = b"".join(chunks)
    if len(data) > MAX_PROMPT_BYTES+12000:
        raise FalError("Request body is too large.", 413)
    try:
        value = json.loads(data)
    except ValueError:
        raise ValueError("Supply a valid JSON object.") from None
    if not isinstance(value, dict):
        raise ValueError("Supply a valid JSON object.")
    return value


async def multipart(request, allowed):
    form = await request.form(max_files=13, max_fields=8, max_part_size=MAX_PROMPT_BYTES)
    for name, value in form.multi_items():
        if isinstance(value, UploadFile) and name not in allowed:
            await form.close()
            raise ValueError("Unexpected upload field.")
    return form


async def images(form, field, maximum):
    files = form.getlist(field)
    if len(files) > maximum:
        raise ValueError(f"Too many {field} images.")
    result = []
    for file in files:
        if not isinstance(file, UploadFile):
            raise ValueError("Supply image files.")
        data = await file.read(MAX_IMAGE_BYTES+1)
        result.append(await asyncio.to_thread(verify_image, data, file.filename or "frame.png"))
    return result


def create_app(engine=None):
    @asynccontextmanager
    async def lifespan(app):
        if engine is None:
            directory = Path(os.getenv("GOZ_DATA_DIR", str(ROOT/"data")))
            if os.getenv("GOZ_DEMO", "0") == "1":
                from .demo import DemoAdapter
                adapter = DemoAdapter(directory/"demo")
            else:
                adapter = FalAdapter(os.getenv("FAL_KEY", ""))
            app.state.engine = Engine(adapter, directory)
        else:
            app.state.engine = engine
        await app.state.engine.resume()
        yield
        await app.state.engine.close()

    app = FastAPI(title="GOZ", lifespan=lifespan)

    @app.middleware("http")
    async def local_access(request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in ("127.0.0.1", "localhost", "testserver") or (origin and origin != f"http://{request.headers.get('host')}"):
            return JSONResponse({"error": "Only the local app can make this request."}, status_code=403)
        try:
            if int(request.headers.get("content-length", "0")) > 13*MAX_IMAGE_BYTES+MAX_PROMPT_BYTES+12000:
                return JSONResponse({"error": "Upload batch is too large."}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "Invalid content length."}, status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(FalError)
    async def provider_error(request, error):
        return JSONResponse({"error": request.app.state.engine.error(error)}, status_code=error.status or 502)

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        return JSONResponse({"error": request.app.state.engine.error(error)}, status_code=500)

    @app.get("/api/config")
    async def config(request: Request):
        e = request.app.state.engine
        return dict(configured=e.adapter.configured(), demo=e.adapter.demo, backend="python", models=MODELS,
                    duration=DURATION, durations=DURATIONS, maxImageBytes=MAX_IMAGE_BYTES,
                    maxCharacters=MAX_CHARACTERS, maxClips=MAX_CLIPS, pollMs=int(e.poll_seconds*1000), sequenceRunner=True)

    @app.post("/api/key")
    async def key(request: Request):
        e = request.app.state.engine
        value = (await json_body(request)).get("key")
        if not isinstance(value, str) or not value.strip() or len(value) > 1000:
            raise ValueError("Enter your Fal API key.")
        async with e.lock:
            if e.adapter.configured() and e.busy():
                raise FalError("Wait for the active generation before changing the key.", 409)
            e.adapter.set_key(value.strip())
            e.uploads.clear()
            await e.resume()
        return {"configured": True}

    @app.post("/api/plan")
    async def preview(request: Request):
        body = await json_body(request)
        return plan(body.get("text", ""), body.get("mode", "keyframes"), duration_value(body.get("duration", DURATION)))

    @app.get("/api/sequences")
    async def sequences(request: Request):
        e = request.app.state.engine
        return [e.snapshot(s) for s in reversed(list(e.sequences.values()))]

    def get_sequence(e, run_id):
        if run_id not in e.sequences:
            raise FalError("Sequence not found.", 404)
        return e.sequences[run_id]

    @app.get("/api/sequences/{run_id}")
    async def sequence(request: Request, run_id: str):
        e = request.app.state.engine
        return e.snapshot(get_sequence(e, run_id))

    @app.post("/api/sequences", status_code=202)
    async def start(request: Request):
        e = request.app.state.engine
        form = await multipart(request, {"start", "frames", "prompt_file"})
        try:
            run_id = str(form.get("id", ""))
            try:
                UUID(run_id)
            except ValueError:
                raise ValueError("A valid run ID is required.") from None
            if run_id in e.sequences:
                return e.snapshot(e.sequences[run_id])
            mode = str(form.get("mode", "chain"))
            duration = duration_value(form.get("duration", DURATION))
            resolution = resolution_value(form.get("resolution", "480P"))
            if form.get("prompt_file"):
                prompt_file = form["prompt_file"]
                if not isinstance(prompt_file, UploadFile):
                    raise ValueError("Supply a prompt file.")
                raw = await prompt_file.read(MAX_PROMPT_BYTES+1)
                if len(raw) > MAX_PROMPT_BYTES:
                    raise ValueError("Prompt file must be at most 100 KB.")
                text = raw.decode("utf-8-sig")
            else:
                text = str(form.get("prompts", ""))
            if mode == "keyframes" and form.getlist("start") or mode == "chain" and form.getlist("frames"):
                raise ValueError("Supply only the frames for the selected mode.")
            frames = await images(form, "frames" if mode == "keyframes" else "start", 13 if mode == "keyframes" else 1)
            result = await e.start_sequence(run_id, text, mode, duration, resolution, frames)
            return e.snapshot(result)
        finally:
            await form.close()

    @app.post("/api/sequences/{run_id}/cancel")
    async def cancel(request: Request, run_id: str):
        e = request.app.state.engine
        return e.snapshot(await e.cancel_sequence(get_sequence(e, run_id)))

    @app.get("/api/jobs")
    async def jobs(request: Request):
        e = request.app.state.engine
        return [e.public_job(j) for j in reversed(list(e.jobs.values()))]

    def get_job(e, job_id):
        if job_id not in e.jobs:
            raise FalError("Generation not found.", 404)
        return e.jobs[job_id]

    @app.post("/api/jobs", status_code=202)
    async def legacy_job(request: Request):
        e = request.app.state.engine
        form = await multipart(request, {"start", "end", "characters"})
        try:
            mode, prompt = form.get("mode"), form.get("prompt")
            if mode not in MODELS:
                raise ValueError("Choose a valid generation mode.")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
                raise ValueError("Write a prompt between 1 and 8,000 characters.")
            names = json.loads(form.get("names", "[]"))
            if not isinstance(names, list) or len(names) > MAX_CHARACTERS or any(not isinstance(n, str) or len(n)>80 for n in names):
                raise ValueError("Character names must be at most 80 characters each.")
            aspect = form.get("aspectRatio", "16:9")
            if aspect not in ("16:9", "9:16", "1:1", "21:9", "4:3", "3:4", "adaptive"):
                raise ValueError("Choose a supported aspect ratio.")
            options = dict(mode=mode, prompt=prompt.strip(), names=names, aspectRatio=aspect,
                           duration=duration_value(form.get("duration", DURATION)), resolution=resolution_value(form.get("resolution", "480P")))
            uploaded = {name: await images(form, name, MAX_CHARACTERS if name == "characters" else 1) for name in ("start", "end", "characters")}
            if mode != "characters" and (not uploaded["start"] or not uploaded["end"]):
                raise ValueError("Upload both starting and ending frames.")
            if mode != "frames" and not uploaded["characters"]:
                raise ValueError("Add at least one character image.")
            if mode == "frames" and uploaded["characters"] or mode == "characters" and (uploaded["start"] or uploaded["end"]):
                raise ValueError("Use combined mode to send frames with character references.")
            async with e.lock:
                if not e.adapter.configured():
                    raise FalError("Add your Fal API key first.", 503)
                if e.busy():
                    raise FalError("A generation is already running or uncertain.", 409)
                job = e.new_job(options, frameControl="exact" if mode == "frames" else "prompt-guided" if mode == "combined" else "none")
                e.spawn(e.run_job(job, {k:v for k,v in uploaded.items() if v}))
            return e.public_job(job)
        finally:
            await form.close()

    @app.post("/api/jobs/{job_id}/measurement")
    async def measurement(request: Request, job_id: str):
        e = request.app.state.engine
        job = get_job(e, job_id)
        ms = (await json_body(request)).get("clickToResultMs")
        if job["status"] not in JOB_TERMINAL or type(ms) not in (int, float) or not math.isfinite(ms) or not 0 <= ms <= 7*86400000:
            raise ValueError("Invalid browser timing or unfinished generation.")
        e.update(job, clickToResultMs=ms)
        return {"saved": True}

    @app.get("/api/jobs/{job_id}/events")
    async def events(request: Request, job_id: str):
        e = request.app.state.engine
        job = get_job(e, job_id)
        async def stream():
            while not await request.is_disconnected():
                yield "data: "+json.dumps(e.public_job(job))+"\n\n"
                if job["status"] in JOB_TERMINAL:
                    return
                await asyncio.sleep(e.poll_seconds)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}/video")
    @app.get("/api/jobs/{job_id}/download")
    async def video(request: Request, job_id: str):
        e = request.app.state.engine
        job = get_job(e, job_id)
        if job["status"] != "completed" or not job.get("video"):
            raise FalError("Video not found.", 404)
        path = await e.media_path(job)
        return FileResponse(path, media_type="video/mp4", filename=f"goz-{job_id}.mp4" if request.url.path.endswith("/download") else None)

    app.mount("/", StaticFiles(directory=ROOT/"frontend", html=True), name="frontend")
    return app


app = create_app()
