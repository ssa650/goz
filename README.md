# GOZ

The **Python backend** handles Fal uploads, model payloads, queue submission/status/results, cancellation, prompt parsing and scene splitting, actual last-frame extraction, history, and MP4 downloads. The **plain JavaScript frontend** is a video player with Generate, Download, Regenerate, and a collapsed generation-times table. It calls only this local backend; it has no provider SDK or API credentials.

## Start

Python 3.11+ is required. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m backend
```

Open http://127.0.0.1:3210. On macOS, `Start GOZ.command` also sets up the environment on first launch and starts the server. No Node server or frontend build step is needed.

Copy `.env.example` to `.env` and set `FAL_KEY`, then restart the Python backend. `.env` is ignored by Git and never served. The migration intentionally starts with fresh history and does not copy credentials, source Git metadata, node_modules, or old generated media.

## Adaptive story (gaze + EEG → next scene)

http://127.0.0.1:3210/adaptive.html plays an ordered story with subtle adjustments from the viewer's responses. While a 5–15 s scene plays, the backend timestamps gaze, blinks, head direction (from [gazekit](../gazekit), `gazekit stream`) and Muse 2 EEG engagement, hit-tests gaze against the characters detected in that scene, and updates a viewer profile. At 70 % of the scene, `gpt-6-luna` chooses one bounded engagement action using the Responses API with strict Structured Outputs. Python applies one short, predefined cue to the next saved prompt; OpenAI never supplies a replacement video prompt. The original scene, dialogue, seed, initial/end frame asset IDs, expansion mode, duration and resolution are preserved. The dashboard shows the current gaze target, the EEG trace, which preference changed and why, and the AI's decision.

Backend startup launches Gazekit automatically and opens a guided setup panel. Run `.venv/bin/python -m backend` and open the player. The setup order is:

1. GOZ immediately launches a native **Gazekit camera-selection window**, without waiting for a browser button or terminal input. It automatically lists connected external cameras and highlights **iPhone Camera**. Press **Enter** to select the detected iPhone, or click/press the displayed camera number. This uses Apple's built-in **Continuity Camera**, with no iPhone app, IP address or phone socket server. Keep the iPhone locked, mounted near the Mac, with its rear cameras facing you. If it is missing, enable Continuity Camera in iPhone Settings → General → AirPlay & Continuity (or AirPlay & Handoff), or connect it over USB, then press **R** to refresh. The selected camera index is reused for calibration and streaming. See [Apple's setup requirements](https://support.apple.com/en-us/102546).

2. Gazekit opens its full target calibration window using that external camera. Follow the targets and press a key on the results screen. A cancelled, failed or poor result cannot advance to Muse or unlock generation.
3. Start gaze streaming with that exact validated external-camera model, perform its quick alignment, and wait for fresh valid gaze samples. Samples are tagged with the calibration attempt ID; another producer cannot unlock this setup. The camera reference is shared between calibration and streaming within this attempt.
4. **Muse now uses Mind Monitor on your phone, like Fable.** Connect the Muse 2 in Mind Monitor (GOZ does not connect to it over Bluetooth in this mode). Put the phone and Mac on the same Wi-Fi. In Mind Monitor, set the OSC destination IP to the Mac's address shown in GOZ, UDP port **5000**, and enable **OSC Stream Brainwaves → All Values** (Average Only also works). Start OSC streaming and keep the app running. Allow local-network access on the phone and incoming Python connections on the Mac if prompted. If iPhone Camera is your Continuity Camera, use another phone for Mind Monitor so the camera phone can stay locked.
5. GOZ shows contact status for TP9/AF7/AF8/TP10 and warms up with **10 usable alpha/beta readings**. It accepts **at least one good contact** and excludes bad electrodes from All Values. It uses Fable's alpha/beta ratio, EMA weight 0.2, and fixed focus/relax thresholds 1.4/2.0. This is a heuristic, not a personal calibration or a probability. The 60-second raw calibration is no longer required in this mode. Stale contact status, missing alpha/beta, or zero good contacts block new generation and show an explanation.

`GOZ_EEG=mindmonitor` is the default. `GOZ_MINDMONITOR_PORT` changes the UDP port; `GOZ_MINDMONITOR_PHONE_IP` optionally pins the sender. Otherwise the first valid contact packet pins the phone for that backend session. Restart GOZ to switch phones. OSC runs on the local network; guest Wi-Fi client isolation can prevent packets reaching the Mac. `GOZ_EEG=muse` retains the direct MuseLSL path and its 60-second raw baseline for troubleshooting. After updating an existing checkout, run `.venv/bin/python -m pip install -r requirements.txt` once to install the OSC receiver dependency.

**Generate**, **Regenerate**, individual clip generation, legacy generation and the adaptive-start API all require ready, live signals. If a signal drops or becomes noisy, new starts are blocked. Previously accepted jobs and downloads remain available. Use **Retry sensor setup** after fixing a setup failure or losing a calibration; retries preserve an owned, still-running Muse bridge and replace the gaze process. Restarting the backend requires fresh gaze calibration and fresh EEG readiness. Backend shutdown terminates only the sensor processes it launched; an existing external Muse stream remains yours.

Gazekit defaults to the sibling `../gazekit` checkout and the backend's Python interpreter. The ridge calibration path uses the NumPy, OpenCV, MediaPipe and scikit-learn dependencies already installed with `requirements.txt`; it does not need CNN training. `GOZ_GAZEKIT_DIR`, `GOZ_GAZEKIT_PYTHON`, `GOZ_MUSE_ADDRESS` and `GOZ_MUSE_NAME` override the local paths or Muse device. macOS may ask for Bluetooth/camera permissions. The official MediaPipe face model is downloaded once if absent. Recordings, the trained gaze model, validation report and Muse readiness metadata (or LSL baseline) remain inside `data/calibration/`, locally; child processes do not receive OpenAI/Fal keys.

`GOZ_REQUIRE_SENSORS=0` explicitly disables the gate for development. `GOZ_DEMO=1` also skips real hardware calibration and is labelled as a demo; neither is evidence of hardware readiness. The Muse checks are application-level signal-quality checks, not a device-native impedance test. Live connection/calibration requires the user's physical headset, camera and participation.

Open the page in a browser window at 100 % zoom (full screen is most accurate: gaze arrives in screen points and the page maps it onto the video using its window position). Keep **Use the saved ordered prompts and frame pairs** checked to freeze the existing clip bundles in their current order. No image upload is required for that mode. An optional opening episode clip plays as-is in place of Clip 1; uncheck the saved-sequence option to use a custom opening image/video and premise instead. Name the characters with a short look description ("green octopus with a long nose"), write the premise, optionally add the clip's timeline, and press **Start**.

Timeline lines mark who speaks and the genre of each moment, so EEG changes during a character's lines count toward that character (speaker timing for generated scenes is unknown unless an explicit timeline is supplied; it is never inferred from the engagement decision):

```
0-3 SpongeBob: I'm ready! #humor
3-7 Squidward: Not today. #humor
7-10 Patrick crashes through the door #action
```

- **Response to a character** = share of the time they were on screen that the viewer looked at them × (1 + EEG index in the 0.3–2 s after each look + ½ × EEG index while they speak). It is *strong* when attention ≥ 40 % and either EEG measure ≥ +0.5 (index units for OSC; standard deviations for LSL).
- **Decisions**: keep, focus one character, slightly faster/slower pacing, subtle suspense/humor, or clearer delivery of the existing dialogue. Only one action is applied. The schema forbids arbitrary prompt text; invalid, incomplete or refused decisions stop before another Fal request. No events, plot changes or new dialogue are introduced. Prompts too close to the 8,000-character limit are retained unchanged rather than truncated.
- **EEG** defaults to Mind Monitor processed alpha/beta on good contacts. The smoothed ratio maps to a focus index `(1.7 − ratio) / 0.3`, clipped to ±5, so Fable's 1.4/2.0 thresholds map to +1/−1. The adaptive fusion uses this index in the existing `z` field; it is not a standard deviation in OSC mode. With `GOZ_EEG=muse`, EEG uses β/(α+θ) over raw 2 s windows every 0.25 s and a frozen personal baseline; windows over 150 µV peak-to-peak are rejected.
- **Characters** are found per scene by Florence-2 open-vocabulary detection on Fal (`GOZ_TRACKER=fal`, default; works on cartoons): 2 frames/s × each character's look description, ~60 small paid calls per 15 s scene, labelled by character. `GOZ_TRACKER=people` uses local MediaPipe people detection instead (live action only; ~7 MB model downloaded to `data/models/`; identities seeded left to right).
- Every signal, tick, analysis and decision is saved under `data/adaptive/<session>/`.
- Fal generation usually takes longer than a scene plays, so the player holds the last frame and shows "Generating scene N…" until it is ready. `GOZ_MAX_SCENES` (default 4) caps paid generations per session. Without `OPENAI_API_KEY`, a local heuristic chooses a bounded cue and the dashboard says **template**. The opening prompt stays unchanged because no viewer evidence exists yet. The next clip's original prompt, applied prompt, decision, model and OpenAI request ID are saved with its job. `OPENAI_MODEL` defaults to `gpt-6-luna`; add `OPENAI_API_KEY` and `FAL_KEY` to `.env`, then restart the backend.

Rehearse without hardware or spending credits (synthetic clips, simulated viewer who prefers the second character):

```sh
GOZ_DEMO=1 GOZ_GAZE=sim GOZ_EEG=sim GOZ_DATA_DIR=output/demo-data PORT=3211 .venv/bin/python -m backend
```

`GOZ_GAZE` is `gazekit|sim|off`, `GOZ_EEG` is `mindmonitor|muse|sim|off`. Real gated generation requires `gazekit` and either `mindmonitor` (default) or `muse`; simulated/off modes are for explicitly configured demos or development. `GOZ_SIM_FAVORITE` (character index) and `GOZ_SIM_BIAS` tune the simulator. SIM sources are labelled on the dashboard.

## Ordered video sequence

The supplied Secret Box prompts and 13 JPGs are bundled under `presets/secret-box/`. New workspaces load all 12 ordered, 15-second clip bundles with their original seeds and correctly paired initial/end frames. There are no prompt editors, file upload controls, or API-key forms in the player. Existing saved definitions and generation history remain intact.

**Generate** submits the saved ordered bundles. **Regenerate** starts a new run with the previous run's exact prompts, seeds, frame asset IDs, expansion modes, durations and resolutions. Neither action changes the seeds. Both actions submit paid Fal requests when demo mode is off. Buttons are locked while generation/stitching is active. A lost submission acknowledgement is retried with the same saved run UUID, preventing an accidental second batch; page reload reconnects an already accepted run.

The player starts with audio enabled at full volume as soon as Clip 1 completes. Two video elements preload the next ordered clip and swap automatically, without rewinding or waiting for the final MP4. If the next clip is still generating, the player holds the last frame and resumes at that exact position when ready. Out-of-order completion never skips clips. There are no native playback controls, hover overlays, seeking, or pause actions. A separate **Fullscreen** button expands the player container, preserving fullscreen across clip transitions; press Escape to exit. Clicking Fullscreen also recovers playback if autoplay was blocked.

After all clips complete, Python sorts by `order` immediately before stitching and creates `final_video.mp4`. **Download** becomes available after assembly. FFmpeg preserves audio, supplies silence for silent clips, normalizes mixed dimensions and frame rates, and trims/pads segments to the requested duration. Final stitching does not interrupt or restart live playback.

The **Generation times** dropdown at the bottom lists all clips in sequence order, their individual status, total time (including waiting/uploads), and provider generation time (queue plus processing). Failed clip rows include readable provider errors. The main error also shows the actual cause, including exhausted Fal balance, rather than just a generic stitching failure. Account/authentication rejection stops not-yet-submitted clips; already accepted requests are preserved and monitored.

Definitions remain atomic clip objects in `data/sequences.json`; frozen runs retain prompt, seed, both asset IDs, settings, status and result. Asset bytes live in `data/images/`, provider secrets remain server-side, and the existing import/edit APIs are still available for developer use without an upload screen.

## Local demo (no Fal calls)

```sh
GOZ_DEMO=1 GOZ_DATA_DIR=output/demo-data PORT=3211 .venv/bin/python -m backend
```

Open http://127.0.0.1:3211. The UI explicitly says **DEMO**. This produces synthetic solid-color MP4s at the chosen duration, using bundled FFmpeg. It exercises the same Python pipeline and JavaScript player without an API key or spending credits. It does not animate the uploaded images or evaluate prompt quality.

## Project layout

- `backend/app.py`: FastAPI routes and static frontend serving.
- `backend/adaptive/mindmonitor.py`: Mind Monitor UDP/OSC receiver, contact selection, paired alpha/beta freshness, EMA, and Fable thresholds.
- `backend/sensor_setup.py`, `backend/gaze_worker.py`: sensor process ownership, startup calibration, readiness gating, retries and Gazekit integration. `GET /api/sensors` exposes setup status; `POST /api/sensors/setup` retries while generation is idle.
- `backend/clips.py`, `backend/clip_settings.py`: independent clip definitions, typed validation and shared H3 routing.
- `backend/bundle_sequence.py`: frozen clip batches, bounded concurrency, per-clip validation and ordered assembly.
- `backend/engine.py`: generation pipeline, history and recovery; deprecated multipart requests convert to complete clip definitions at ingress.
- `backend/fal_adapter.py`: Fal storage upload and one-shot HTTP queue requests.
- `backend/frames.py`: bounded media download and full-decode final-frame extraction and audio-preserving MP4 stitching.
- `backend/prompts.py`, `backend/config.py`: prompt planning and non-secret model settings.
- `frontend/`: JavaScript player, sequence controller and styles. `sequence-playback.js` feeds ordered live snapshots into the two-element queue without restarting playback. Player UI, controller, clip state, playback adapter and queue use JSDoc types checked with TypeScript.
- `tests/`: Python integration tests and JavaScript playback tests.

## API

Individual cards use `GET/POST /api/clips`, `PATCH/DELETE /api/clips/{id}`, and `POST /api/clips/{id}/duplicate`, `/generate`, or `/cancel`. Generation takes `{ "token": "UUID" }`; edit/create takes the card settings, with `firstFrame`/`endFrame` as stored image IDs or null. `POST /api/clips/import` accepts `{ "text": "...", "mode": "replace" }` (or `append`). Without mode, structured clip objects replace the editor; legacy prompt strings append. `POST /api/clips/import/undo` restores the prior editor. `POST /api/clips/frames` takes multipart repeated `frames` and resolves imported filename bindings atomically. `POST /api/images` takes multipart `image`; `GET /api/images/{id}` serves its validated thumbnail/source image. Arbitrary client-supplied storage URLs are not accepted.

- `GET /api/config`: non-secret settings, key configuration status, demo flag.
- `POST /api/key`: `{ "key": "..." }`, stored only in server memory.
- `POST /api/plan`: `{ "text": "...", "mode": "keyframes", "duration": 5 }`, returns clip prompts and required frame count.
- `POST /api/clips/order`: `{ "clipIds": ["UUID", "UUID"] }`, every current clip ID exactly once.
- `POST /api/sequences`: `{ "id": "UUID", "clips": [{ "id": "clip UUID", "order": 0, "prompt": "...", "seed": 483729, "firstFrame": "asset UUID or null", "endFrame": null, "promptExpansionMode": "disabled", "duration": 15, "resolution": "480P" }] }`. The frontend sends array order; the backend validates and sorts explicit order. Each run ID is idempotent. Validation failures include a `clipErrors` map keyed by clip ID.
- `GET /api/sequences/{id}/video` and `/download`: stream or save the completed stitched MP4.
- Deprecated `POST /api/sequences` multipart `id` (UUID), `mode`, `prompts` (raw text or JSON array), `duration`, `resolution`, plus repeated `frames` or one `start`. An optional `prompt_file` can replace `prompts`. Each UUID identifies one retained run, including retries after completion or cancellation.
- `GET /api/sequences`, `GET /api/sequences/{id}`, `POST /api/sequences/{id}/cancel`.
- `GET /api/jobs/{id}/video`: Python downloads/caches the MP4 and serves it with byte-range support; browsers never fetch Fal directly.
- `GET /api/jobs/{id}/download`: saves the same MP4.

The original `/api/jobs` frame/character/combined modes, job history, SSE events, and browser timing measurement routes are also ported to Python. Combined mode uses reference images and prompt-guided compositions; it does not guarantee exact keyframes. Model fields follow the [Fal image-to-video schema](https://fal.ai/models/minimax/h3-max-turbo/image-to-video/api) and the [Fal Python client](https://fal-ai.github.io/fal/client/fal_client.html).

## Recovery and local data

The server binds to `127.0.0.1` and rejects foreign origins/hosts. Run one backend process, with one worker, per data directory; independent clip jobs run asynchronously within it. The entire `data/` folder is local and ignored by Git. Clip settings, prompts, seeds, exact generation inputs, request IDs, timings, provider media URLs, reference images, and downloaded clips are persisted; keys are not. Keep downloaded clips you need; provider URLs can expire. Media cache is retained locally and can be removed with the server stopped when no longer needed.

Reload reconnects without submitting another sequence. Server restart interrupts the sequence and never automatically submits its next clip; a known active Fal request can be monitored when the same key is configured. Paid POSTs are never automatically retried. If confirmation is lost or status monitoring times out after 20 minutes, a persistent `requestUncertain` flag blocks further generations. Check Fal request history before resolving that flag in `data/history.json` with the server stopped. Corrupt history fails startup rather than silently removing this block.

## Verification

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
npm ci
npm run typecheck
npm run check
npm test
```

Tests use fake adapters and synthetic media. They cover ordered chaining/keyframes, duplicate run tokens, concurrent runs, cancellation during upload/submission, reconnects, restart recovery, bounded inputs, one-shot queue submission, legacy modes, real final-frame decoding, and player order/buffering/autoplay. Browser verification uses local demo clips; paid generation and model quality have not been verified for this Python migration.

There is no configured lint command. Verification includes the Python integration suite, 19 JavaScript tests, and strict TypeScript checks. HTTP transport tests inspect each atomic bundle's actual H3 payload for text-only, initial-frame, initial/end-frame and mixed modes; forced out-of-order completion must still stitch in the original order. Tests cover invalid batches, duplication/reordering, asset associations across reload, seed preservation, account rejection, idempotent retry after a lost acknowledgement, and exact Regenerate inputs. A real FFmpeg test decodes the stitched output to check segment order, duration and audio.

The simplified player was checked with all 12 bundled clip inputs in an isolated local demo, using five-second synthetic clips and a delayed Clip 2. Clip 1 played while generation was active, the player waited for Clip 2, and all clips continued automatically through the end. Download produced a valid 60.02-second MP4; Regenerate retained every input and started a new run; reload recovered saved results without extra POSTs. The real saved failure was also checked: the UI now displays Fal's exhausted-balance error. No new paid generation was submitted during verification.

Adaptive integration tests use fake OpenAI/Fal HTTP transports and simulated viewer data. They check strict JSON schemas, refusal/incomplete/invalid output, bounded prompt changes without truncation, and exact ordered clip prompt/seed/first/end-frame associations in all four input modes. Live provider calls and Muse/gaze hardware are not verified by those tests. The main `/` player remains the saved-sequence generator; `/adaptive.html` runs the sensor-driven decision loop.

Sensor tests cover real OSC UDP bundles, single-good-contact operation, invalid/non-finite data, contact and brainwave dropouts, sender pinning, OSC setup/retry without launching MuseLSL, clean LSL baseline collection, flat/clipped/mains/movement rejection, frozen calibration and dropout invalidation, Muse stream selection/clock conversion, Gazekit attempt IDs, successful/failed calibration process lifecycles, bridge reuse on retry, and generation-route bypass prevention. The `ClipService.list` annotations use `builtins.list` to avoid the method name shadowing Python’s list type, including on Python versions that evaluate annotations eagerly.
