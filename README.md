# GOZ
The **Python backend** handles Fal uploads, model payloads, queue submission/status/results, cancellation, prompt parsing and scene splitting, actual last-frame extraction, history, and MP4 downloads. The **plain JavaScript frontend** is a video player with temporary debug controls for frame uploads, prompt files/pasting, duration, resolution, run status, timings, and downloads. It calls only this local backend; it has no provider SDK or API credentials.

## Start

Python 3.11+ is required. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m backend
```

Open http://127.0.0.1:3210. On macOS, `Start GOZ.command` also sets up the environment on first launch and starts the server. No Node server or frontend build step is needed.

Add your Fal key in the debug panel (Python server memory only), or copy `.env.example` to `.env` and set `FAL_KEY`. `.env` is ignored by Git and never served. The migration intentionally starts with fresh history and does not copy credentials, source Git metadata, node_modules, or old generated media.

## Debug workflow

1. Choose **Start and end keyframes** or **Initial frame → chain**.
2. Upload PNG, JPEG, or WebP frames, up to 10 MB each. Keyframes are sorted by file name; name them `01.png`, `02.png`, etc. A sequence of N clips needs N+1 keyframes. Chain mode needs just one initial frame.
3. Upload `.txt` (one prompt per nonempty line) or `.json` (an array of strings), or paste prompts. Up to 12 scenes, 8,000 characters per scene, 100 KB per batch. The copied `prompts/bubble-studio.json` is available for debugging but is never loaded or submitted automatically.
4. Select 5–15 seconds per clip and 480p/768p/1080p. In keyframe mode, Python splits scenes with `DURATION:` and timed blocks into shorter clips, keeping dialogue in a single part. The planning count shows the resulting frame requirement. Chain mode uses one prompt per clip.
5. **Run sequence** starts paid Fal calls in live mode. Clips play muted in order as they become ready. Enable sound if desired. Cancel prevents subsequent clips and requests best-effort cancellation of an accepted request, which may still be charged.

Two video elements preload/swap clips. Playback waits for late clips rather than skipping them; real generation can take longer than playback. Frame previews, inputs, and raw backend state are debug UI and can be removed when the project has another input source.

## Local demo (no Fal calls)

```sh
GOZ_DEMO=1 GOZ_DATA_DIR=output/demo-data PORT=3211 .venv/bin/python -m backend
```

Open http://127.0.0.1:3211. The UI explicitly says **DEMO**. This produces synthetic solid-color MP4s at the chosen duration, using bundled FFmpeg. It exercises the same Python pipeline and JavaScript player without an API key or spending credits. It does not animate the uploaded images or evaluate prompt quality.

## Project layout

- `backend/app.py`: FastAPI routes and static frontend serving.
- `backend/engine.py`: generation pipeline, sequence ordering, persistence and recovery.
- `backend/fal_adapter.py`: Fal storage upload and one-shot HTTP queue requests.
- `backend/frames.py`: bounded media download and full-decode final-frame extraction.
- `backend/prompts.py`, `backend/config.py`: prompt planning and non-secret model settings.
- `frontend/`: JavaScript player, debug controls and styles; the player queue is copied from GOZ_TEST.
- `tests/`: Python integration tests and JavaScript playback tests.

## API

- `GET /api/config`: non-secret settings, key configuration status, demo flag.
- `POST /api/key`: `{ "key": "..." }`, stored only in server memory.
- `POST /api/plan`: `{ "text": "...", "mode": "keyframes", "duration": 5 }`, returns clip prompts and required frame count.
- `POST /api/sequences`: multipart `id` (UUID), `mode`, `prompts` (raw text or JSON array), `duration`, `resolution`, plus repeated `frames` or one `start`. An optional `prompt_file` can replace `prompts`. Each UUID identifies one retained run, including retries after completion or cancellation.
- `GET /api/sequences`, `GET /api/sequences/{id}`, `POST /api/sequences/{id}/cancel`.
- `GET /api/jobs/{id}/video`: Python downloads/caches the MP4 and serves it with byte-range support; browsers never fetch Fal directly.
- `GET /api/jobs/{id}/download`: saves the same MP4.

The original `/api/jobs` frame/character/combined modes, job history, SSE events, and browser timing measurement routes are also ported to Python. Combined mode uses reference images and prompt-guided compositions; it does not guarantee exact keyframes. Model fields follow the [Fal image-to-video schema](https://fal.ai/models/minimax/h3-max-turbo/image-to-video/api) and the [Fal Python client](https://fal-ai.github.io/fal/client/fal_client.html).

## Recovery and local data

The server binds to `127.0.0.1` and rejects foreign origins/hosts. Run one backend process, with one worker, per data directory. The entire `data/` folder is local and ignored by Git. Prompts, request IDs, timings, provider media URLs, and downloaded clips are persisted; keys and original uploaded frames are not. Keep downloaded clips you need; provider URLs can expire. Media cache is retained locally and can be removed with the server stopped when no longer needed.

Reload reconnects without submitting another sequence. Server restart interrupts the sequence and never automatically submits its next clip; a known active Fal request can be monitored when the same key is configured. Paid POSTs are never automatically retried. If confirmation is lost or status monitoring times out after 20 minutes, a persistent `requestUncertain` flag blocks further generations. Check Fal request history before resolving that flag in `data/history.json` with the server stopped. Corrupt history fails startup rather than silently removing this block.

## Verification

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
npm run check
npm test
```

Tests use fake adapters and synthetic media. They cover ordered chaining/keyframes, duplicate run tokens, concurrent runs, cancellation during upload/submission, reconnects, restart recovery, bounded inputs, one-shot queue submission, legacy modes, real final-frame decoding, and player order/buffering/autoplay. Browser verification uses local demo clips; paid generation and model quality have not been verified for this Python migration.
