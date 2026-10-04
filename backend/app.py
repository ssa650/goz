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
from pydantic import ValidationError
from .config import (MODELS, DURATION, DURATIONS, MAX_IMAGE_BYTES, MAX_CHARACTERS,
                     MAX_CLIPS, MAX_PROMPT_BYTES, POLL_SECONDS)
from .engine import Engine, JOB_TERMINAL
from .fal_adapter import FalAdapter, FalError
from .frames import verify_image
from .prompts import plan
from .bundle_sequence import ClipValidationError
from .clip_import import filename_key
from .preset import ensure_default_sequence, load_default_sequence

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


async def multipart(request, allowed, max_files=13):
    form = await request.form(max_files=max_files, max_fields=8, max_part_size=MAX_PROMPT_BYTES)
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
            await ensure_default_sequence(app.state.engine)
        else:
            app.state.engine = engine
        from .adaptive.routes import Sensors
        e = app.state.engine
        app.state.sensors = Sensors(e.directory, demo=e.adapter.demo)
        e.generation_guard = app.state.sensors.setup.require_ready
        await app.state.sensors.start()
        await app.state.engine.resume()
        yield
        await app.state.sensors.close()
        await app.state.engine.close()

    app = FastAPI(title="GOZ", lifespan=lifespan)

    @app.middleware("http")
    async def local_access(request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in ("127.0.0.1", "localhost", "testserver") or (origin and origin != f"http://{request.headers.get('host')}"):
            return JSONResponse({"error": "Only the local app can make this request."}, status_code=403)
        try:
            max_files = MAX_CLIPS*2 if request.url.path == "/api/clips/frames" else 13
            if int(request.headers.get("content-length", "0")) > max_files*MAX_IMAGE_BYTES+MAX_PROMPT_BYTES+12000:
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

    @app.exception_handler(ValidationError)
    async def invalid_settings(request, error):
        messages = [f"{'.'.join(map(str, item['loc'])) or 'Settings'}: {item['msg']}" for item in error.errors()]
        return JSONResponse({"error": "; ".join(messages)}, status_code=400)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(ClipValidationError)
    async def invalid_clips(request, error):
        return JSONResponse({"error": str(error), "clipErrors": error.errors}, status_code=400)

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
                    sensorSetup=request.app.state.sensors.setup.snapshot(),
                    duration=DURATION, durations=DURATIONS, maxImageBytes=MAX_IMAGE_BYTES,
                    maxCharacters=MAX_CHARACTERS, maxClips=MAX_CLIPS, promptExpansionModes=["disabled", "balanced", "quality"], individualClips=True, canUndoImport="beforeImport" in e.sequences.get("individual-clips", {}), pollMs=int(e.poll_seconds*1000), sequenceRunner=True, presetAvailable=True)

    @app.get("/api/sensors")
    async def sensor_state(request: Request):
        return request.app.state.sensors.setup.snapshot()

    @app.post("/api/sensors/setup", status_code=202)
    async def sensor_retry(request: Request):
        e, sensors = request.app.state.engine, request.app.state.sensors
        async with e.lock:
            if e.busy() or sensors.session and sensors.session.status == "running":
                raise FalError("Stop or finish the current generation before recalibrating sensors.", 409)
            sensors.start_muse_reader()
            await sensors.setup.retry()
        return sensors.setup.snapshot()

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

    @app.get("/api/clips")
    async def clip_list(request: Request):
        return request.app.state.engine.clips.list()

    @app.post("/api/clips/preset")
    async def clip_preset(request: Request):
        e = request.app.state.engine
        async with e.lock:
            return await load_default_sequence(e)

    @app.post("/api/clips", status_code=201)
    async def clip_add(request: Request):
        e = request.app.state.engine
        body = await json_body(request)
        starter = body.pop("starter", False)
        async with e.lock:
            result = e.clips.add(body)
            if starter:
                e.clips.get(result["id"])["starter"] = True
                e.persist()
        return result

    @app.post("/api/clips/import")
    async def clip_import(request: Request):
        e = request.app.state.engine
        body = await json_body(request)
        async with e.lock:
            return e.clips.import_prompts(body.get("text", ""), body.get("mode"))

    @app.post("/api/clips/import/undo")
    async def clip_import_undo(request: Request):
        e = request.app.state.engine
        async with e.lock:
            return e.clips.undo_import()

    @app.post("/api/clips/frames")
    async def clip_frames(request: Request):
        e = request.app.state.engine
        form = await multipart(request, {"frames"}, max_files=MAX_CLIPS*2)
        try:
            files = form.getlist("frames")
            if not files or any(not isinstance(file, UploadFile) for file in files):
                raise ValueError("Select the frame images referenced by your clips JSON.")
            async with e.lock:
                plan = e.clips.frame_plan([file.filename for file in files])
                # Validate every image before changing any clip attachment.
                uploaded = []
                for file in files:
                    try:
                        uploaded.append(await asyncio.to_thread(verify_image, await file.read(MAX_IMAGE_BYTES+1), file.filename))
                    except ValueError as error:
                        raise ValueError(f"{file.filename}: {error}") from None
                references = {filename_key(image.name):e.save_reference(image, reuse=False) for image in uploaded}
                return e.clips.attach_frames(plan, references)
        finally:
            await form.close()

    @app.patch("/api/clips/{clip_id}")
    async def clip_edit(request: Request, clip_id: str):
        e = request.app.state.engine
        body = await json_body(request)
        async with e.lock:
            record = e.clips.get(clip_id)
            result = e.clips.edit(clip_id, body)
            if body:
                record.pop("starter", None)
                e.persist()
            return result

    @app.post("/api/clips/order")
    async def clip_order(request: Request):
        e = request.app.state.engine
        body = await json_body(request)
        async with e.lock:
            return e.clips.reorder(body.get("clipIds"))

    @app.delete("/api/clips/{clip_id}")
    async def clip_remove(request: Request, clip_id: str):
        e = request.app.state.engine
        async with e.lock:
            e.clips.remove(clip_id)
        return {"removed": True}

    @app.post("/api/clips/{clip_id}/duplicate", status_code=201)
    async def clip_duplicate(request: Request, clip_id: str):
        e = request.app.state.engine
        async with e.lock:
            return e.clips.duplicate(clip_id)

    @app.post("/api/clips/{clip_id}/generate", status_code=202)
    async def clip_generate(request: Request, clip_id: str):
        body = await json_body(request)
        token = body.get("token")
        if not isinstance(token, str):
            raise ValueError("Supply a unique generation token.")
        return await request.app.state.engine.clips.generate(clip_id, token)

    @app.post("/api/clips/{clip_id}/cancel")
    async def clip_cancel(request: Request, clip_id: str):
        e = request.app.state.engine
        record = e.clips.get(clip_id)
        job = e.jobs.get(record.get("jobId"))
        if job:
            await e.cancel_job(job)
        return e.clips.snapshot(record)

    @app.post("/api/images", status_code=201)
    async def image_upload(request: Request):
        e = request.app.state.engine
        form = await multipart(request, {"image"})
        try:
            uploaded = await images(form, "image", 1)
            if len(uploaded) != 1:
                raise ValueError("Upload one PNG, JPEG, or WebP image, up to 10 MB.")
            reference = e.save_reference(uploaded[0], reuse=False)
            # Keep previews local. The provider URL is obtained by the existing upload
            # service on generation, so frame selection works before adding a key.
            return e.reference(reference["id"])
        finally:
            await form.close()

    @app.get("/api/images/{reference_id}")
    async def image_preview(request: Request, reference_id: str):
        e = request.app.state.engine
        reference = e.reference(reference_id)
        return FileResponse(e.images/f"{reference_id}.bin", media_type=reference["contentType"])

    @app.post("/api/plan")
    async def preview(request: Request):
        body = await json_body(request)
        return plan(body.get("text", ""), body.get("mode", "keyframes"), duration_value(body.get("duration", DURATION)))

    @app.get("/api/sequences")
    async def sequences(request: Request):
        e = request.app.state.engine
        return [e.snapshot(s) for s in reversed(list(e.sequences.values())) if s.get("mode") != "individual"]

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
        if request.headers.get("content-type", "").split(";")[0] == "application/json":
            return await e.bundles.start(await json_body(request))
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

    async def final_path(request, run_id):
        e = request.app.state.engine
        run = get_sequence(e, run_id)
        if run.get("mode") != "bundles" or run["status"] != "completed" or not run.get("finalVideoUrl"):
            raise FalError("The final video is not ready. All clips must complete and stitch first.", 409)
        path = e.media/f"sequence-{run['id']}.mp4"
        if not path.is_file():
            raise FalError("The stitched video file is missing.", 404)
        return path

    @app.get("/api/sequences/{run_id}/video")
    async def final_video(request: Request, run_id: str):
        return FileResponse(await final_path(request, run_id), media_type="video/mp4")

    @app.get("/api/sequences/{run_id}/download")
    async def final_download(request: Request, run_id: str):
        return FileResponse(await final_path(request, run_id), media_type="video/mp4", filename="final_video.mp4")

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
            if mode not in ("frames", "characters", "combined"):
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
                e.generation_guard()
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

    from .adaptive.routes import register
    register(app, json_body, images, multipart, duration_value, resolution_value)

    app.mount("/", StaticFiles(directory=ROOT/"frontend", html=True), name="frontend")
    return app


app = create_app()
