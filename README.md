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

If `.env` does not exist, copy `.env.example` to `.env`; set `FAL_KEY`, then restart the Python backend. `.env` is ignored by Git and never served. The migration intentionally starts with fresh history and does not copy credentials, source Git metadata, node_modules, or old generated media.

## Adaptive story: gaze + optional EEG → next scene

Open [Adaptive Story](http://127.0.0.1:3210/adaptive.html). The loop is presented video → captured gaze → verified scene targets → valid elapsed dwell → tentative attention preference → local decision → changed H3 request → downloaded/validated media → automatic playback. EEG is optional physiological context and never proves liking or enjoyment.

### Run the demo

1. Install the Python dependencies above. Keep your existing `.env`; create it from `.env.example` only if absent. Set `FAL_KEY` and restart the backend. Adaptive decisions and prompt composition run locally. Optional visual identity verification uses an existing `OPENAI_API_KEY` plus `GOZ_IDENTITY_MODEL` (or the existing `OPENAI_MODEL`); it checks scene pixels, not viewer responses. No model name is silently selected when these are absent.
2. For webcam gaze, use the existing [Gazekit checkout](https://github.com/ba13231/gazekit) at `../gazekit`, or set `GOZ_GAZEKIT_DIR` to its absolute path. `GOZ_GAZEKIT_PYTHON` optionally selects its Python interpreter. GOZ owns the camera process; do not start a second Gazekit stream against the same camera.
3. Run `.venv/bin/python -m backend`. With `GOZ_GAZE=gazekit`, select the webcam (including an available iPhone Continuity Camera) in the native window and follow the calibration targets. Keep the same viewer, camera, display, lighting and seating. The model is saved before the results screen is dismissed, then loaded in a fresh streaming process. Complete the independent check below before relying on target attribution.
4. Connect optional Muse using one of the supported local routes below. `GOZ_REQUIRE_SENSORS=0` allows gaze-only use when EEG is absent/noisy; missing gaze produces a balanced continuation. `GOZ_REQUIRE_SENSORS=1` is an optional strict rehearsal gate requiring both calibrated, fresh gaze and clean EEG.
5. Keep **Use the saved ordered prompts and frame pairs** selected for the supplied Secret Box episode. Its existing assets, seeds, durations, resolution and frame pairs remain unchanged. The original `presets/secret-box/frames/00-00.jpg` is nearly black; an empty detection there is expected. For a custom story, deselect the saved sequence and provide a clear opening frame or an episode clip up to 120 seconds/100 MB. `presets/secret-box/frames/00-45.jpg` is a visible supplied frame.
6. Use 100% browser zoom on the calibrated main display. Move the pointer over the player to anchor its screen position after moving/resizing the window or entering fullscreen. Start the story and enable sound. The marker shows the measured gaze point without snapping to boxes. Inspect the target/validity label, valid dwell, EEG status, evidence, selected shot, exact prompt/payload, decision IDs and measured readiness. Click the video if browser autoplay is blocked.
7. To demonstrate comparative attention, use a segment where SpongeBob and Patrick are both verified as visible, and sustain gaze on one. At least two seconds of valid comparative evidence is required; a single visible actor is not a proven preference. The continuation should give the selected actor the main shot. Inspect the actual generated video as well as its prompt.
8. **Stop** stops the session and requests cancellation of an active provider job. An already accepted request may still finish and be charged. Start a new session after provider errors; obsolete results remain associated with their original session.

The verified iPhone/Muse setup is currently served at `http://127.0.0.1:3214/adaptive.html`. After stopping that backend, restart the same configuration from this folder with:

```sh
GOZ_GAZE=gazekit GOZ_EEG=muse GOZ_GAZE_PORT=5594 \
GOZ_DATA_DIR=data GOZ_MAX_SCENES=12 PORT=3214 .venv/bin/python -m backend
```

It reloads the saved phone model. Do not launch another process against the same camera or data directory. The acceptance API runs used `scene_limit=2` to bound paid generation; normal UI starts retain the 12-scene episode limit.

### Calibration storage, reload and checks

`GOZ_GAZE_CAMERA` optionally selects a camera index for a new calibration; saved calibration resolves its recorded device identity instead of assuming that the index is unchanged. Keep the iPhone fixed near the display.

The default stable storage is `<repository>/data/calibration/`, independent of the launch working directory. Use an absolute `GOZ_DATA_DIR` to relocate local data. `GOZ_VIEWER_PROFILE=default` selects `saved-gaze.json`; other profile names select separate hashed manifest filenames. Each manifest refers to its saved model/report and optional `gaze_alignment.json` inside an individual attempt folder. Files remain local and out of Git; no existing model is overwritten when another attempt starts.

Schema version 2 records the viewer profile, camera identity/name, screen geometry in screen points, and a hash of Gazekit's feature/model code plus its scikit-learn version. Atomic saves and checksums detect incomplete or corrupt artifacts. Reuse resolves camera identity rather than trusting an old camera index. Incompatible/legacy models are rejected with a diagnostic path and reason; their artifacts remain preserved. A model loading successfully does not establish accuracy after the viewer or physical setup changes.

While the adaptive session is stopped, use **Check gaze** to present fresh targets excluded from model training. **Recenter** fits a small alignment adjustment, tests separate probe targets and saves it only if those checks pass. The UI/API pauses the owned camera stream and reloads it after success:

```sh
curl -X POST http://127.0.0.1:3210/api/sensors/gaze-check \
  -H 'Content-Type: application/json' -d '{"recenter":false}'
# Use {"recenter":true} for a checked alignment correction.
```

`GET /api/sensors` supplies `gaze.calibrationState`, `calibrationPath`, `failureReason`, `profileId` and the exact `checkCommand` for the saved model. To use that CLI command instead, stop the backend first to release its camera, run the command, then restart. The worker also supports `inspect` with the same `--repo`, `--model` and `--profile` arguments to check serialization/loading in a fresh process without opening the camera. **Remove eye calibration** removes the selected manifest and begins fresh calibration; older model files remain preserved. Recheck after changing seating; recalibrate when the viewer, camera or display geometry changes.

### Muse 2 connection

The browser does not connect directly to Bluetooth. Choose one existing local route:

- **Mind Monitor (`GOZ_EEG=mindmonitor`, default):** connect Muse in the phone app, with phone and Mac on the same Wi-Fi. Set its OSC destination to the Mac IP shown in the sensor panel, UDP `5000` (`GOZ_MINDMONITOR_PORT`), and send `/muse/elements/horseshoe`, `alpha_absolute` and `beta_absolute` using All Values or Average Only. Blink/jaw-clench events and optional `/muse/eeg` raw packets improve artifact rejection. `GOZ_MINDMONITOR_PHONE_IP` can pin the expected sender. Ten usable paired readings warm up the receiver. If the same iPhone is the Continuity Camera, use another device for Mind Monitor.
- **Direct MuseLSL (`GOZ_EEG=muse`):** GOZ owns a local MuseLSL bridge using Bleak with the Muse 2 legacy protocol, then consumes its LSL EEG stream. Allow Bluetooth for the Python/terminal process in macOS settings. Close Mind Monitor/Muse or another bridge connected to this headset before discovery. Set `GOZ_MUSE_ADDRESS` or `GOZ_MUSE_NAME` to select a device when needed. Direct EEG requires 60 seconds of clean baseline; adjust forehead/ear contacts and remain still. A connection is only useful when fresh samples actually arrive.

The dashboard separates disconnected/connecting/streaming, stale data, poor signal and errors. Artifact-contaminated or missing EEG contributes no response evidence; it is not zero physiological response. Reconnection uses one subscription and does not reset the video session. **EEG unavailable — gaze-only mode** is the expected degraded mode. Mind Monitor reports a relative alpha/beta baseline observation; direct LSL reports relative beta/(alpha+theta). Neither is an emotion/focus probability or a causal attribution to the viewed character.

Physical streaming was observed during this repair (2,616 samples over 10.215 seconds, 256 Hz); those windows were rejected for clipping/contact/movement/mains artifacts, with zero clean baseline seconds. This establishes connection/sample delivery, not usable physiological evidence. Fit/adjust the headset and obtain a clean baseline before the EEG portion of the demo. Evidence is in `output/sensor-verification/muse-live-evidence.json` when available locally.

### Detection, synchronization and adaptation

Florence uses the configured **dense-region-caption** endpoint on scene-video frames, not viewer webcam images. It runs asynchronously outside media readiness, capped at eight paid frame queries per clip and concurrency two. Adaptive sessions prioritize the first 3.5 seconds at 0.5-second spacing (2 frames/second), using the same eight-frame budget and native-cadence cut detection only inside that observation window. The general whole-clip detector retains its 0.5-frame/second default. Boxes still expire after 0.8 seconds; later playback attribution is unavailable when no fresh opening-window boxes remain. Florence region captions alone cannot establish named identity: actual generated frames showed that a caption can use the wrong franchise character name. Local reference matching checks the supplied episode assets. When `GOZ_IDENTITY=auto` and an OpenAI key/model are configured, one bounded Responses image-input request independently checks up to eight scene frames and the two available character reference crops per clip, concurrently with Florence. It returns visible names, normalized boxes and visual evidence; allowed names are possibilities, not required detections. `GOZ_IDENTITY=off` disables this verifier. Its 40-second request timeout has no automatic retry. On missing configuration, rejection, timeout or malformed output, only geometric reference-verified identities remain; novel appearances may be unknown. Neither route uses color/cast order as identity evidence. No provider confidence probability is invented; any local similarity/geometry measure is a heuristic. Unknown, ambiguous, empty and unavailable results are valid outcomes.

Content-addressed detections are cached under `data/detections/` with a schema version. Optional verifier results use `identity-<hash>.json`, keyed by actual scene pixels, model, identity names, reference appearance and schema. Only scene frames/reference crops go to the visual services; viewer webcam imagery and raw EEG stay local. Boxes carry clip/session provenance, frame timestamps and normalized coordinates. Local cut detection resets attribution; boxes expire after 0.8 seconds and do not cross into another clip. Sparse processing can therefore leave attribution unavailable between observations. Quick motion/occlusion and novel appearances remain limitations; inspect boxes on the actual episode before the demo. Optional local `GOZ_TRACKER=people` cannot verify cartoon identity.

Gazekit sends camera-read capture timestamps, screen-point coordinates, validity, blink/head pose and setup identity over localhost UDP (`GOZ_GAZE_PORT=5590`). Playback reports carry session/clip IDs, seek epochs, the actually presented frame time where the browser supports it, and server-clock-adjusted timestamps. The overlay and hit-test share the same object-fit content transform; clipping/letterbox areas, window position, scrolling and resize are represented separately from image coordinates. Device-pixel ratio scales canvas drawing only, not gaze coordinates. Windowed mapping needs the pointer anchor and 100% zoom; the calibrated primary display is the supported setup.

Dwell integrates adjacent valid elapsed intervals, not sample counts. It excludes tracking gaps, blink/face loss, poor quality, paused playback, stale detections, seeks and mismatched clips. Valid gaze on background, outside video, ambiguous targets and missing/stale tracking remain distinct. Missing tracking is never evidence of looking away.

The deterministic controller requires at least 2 seconds of comparative visibility, at least 1.25 seconds of target dwell and a mean gaze-quality heuristic of 0.6. It compares attention relative to visibility, with stronger evidence needed to switch an established emphasis. EEG magnitude never boosts a character's preference score. Pacing/dialogue changes require at least 3 seconds of fresh valid gaze with mean quality at least 0.6. Pacing additionally needs at least 1.4 seconds of actual outside-video gaze and a fraction of at least 0.35; dialogue comparisons require at least 1.5 valid seconds in each dialogue/silent condition and the existing 0.3 attention difference. This shorter decision window yields tentative evidence from less total viewing; neither missing samples nor expired boxes fill the gap. Dialogue comparisons require supplied timed dialogue/silent windows. An optional opening timeline uses:

```text
0-3 SpongeBob: I'm ready! #humor
3-7 Patrick hides the box #suspense
7-10 SpongeBob reacts #humor
```

Generated speech timings are unknown unless supplied; script/beat metadata is not observed character presence. The per-session profile is labelled tentative attention preference and is not automatically shared across viewers.

A focus decision assigns the observed, script-supported actor the main medium close-up for the continuation's interior. The saved prompt remains intact and the appended direction explicitly overrides conflicting interior camera directions while preserving scripted actor assignments, lines, identity and story outcome. Supplied first/last frames govern the boundary compositions. `scene_spec`, `prompt_changes`, the full provider request, decision ID and source observation window make that contract inspectable. A request exceeding the prompt limit fails explicitly rather than silently dropping adaptation. Custom stories retain the existing establish → notice → investigate → reveal controller and actual last-frame continuation.

### Latency, queue and failure behavior

The adaptive queue has one future continuation and one generation owner. The player preloads a second video element, waits for media readiness and automatically swaps in order. Only one element plays audio. Each clip freezes its available opening evidence at the first playing report reaching **3.5 seconds of playback**, independently of full-scene detection or whether a focus decision is actionable. Opening generation, download and browser preload time do not consume this observation window. Partial verified detections can contribute before the rest of the clip finishes processing. Missing, stale or insufficient evidence produces a balanced continuation using the unchanged character-quality/comparison/dwell gates and the opening-window readability requirements; late observations cannot revise the submitted request. Short clips use their earlier end as a bounded fallback. Custom clips decode and cache their **actual file end frame before media is marked ready**; saved bundles retain their existing frame pairs. Frame failure is explicit and never substitutes the opening or currently displayed frame. Scene/session changes discard obsolete results and request provider cancellation without holding the generation owner while cancellation is pending. The maximum of the last three measured preparation times (20 seconds before measurements) remains diagnostic and no longer delays the observation deadline.

This buffering trades feedback freshness for preparation time. Once submitted, a request is immutable; later observations cannot alter it. The dashboard distinguishes the decision behind the currently playing clip from the next request. Feedback appears after that next clip becomes ready and is reached in sequence.

When a clip is ready, the two-element player can transition immediately; browser events record the observed transition delay. When generation is late, the player displays **BRIDGE · holding the last frame** and collects no new playback evidence. It resumes at the next generated scene when ready. Generation failure retains the last frame with a visible error and stops further generation; it does not replay a clip or silently substitute a prerecorded branch. Failed opening generation displays its error without inventing a fallback scene. Fix the provider issue and restart the session.

Timing fields have distinct meanings:

| Field | Measured scope |
| --- | --- |
| `decisionMs`, `compositionMs`, `payloadConstructionMs` | Local analysis/decision, scene composition and final payload construction |
| `submissionMs` | Client's Fal submission request/acknowledgement time |
| `queueObservedMs` | Time from acknowledgement until the first polled in-progress status; an observation upper bound, not an exact queue measurement |
| `providerObservedElapsedMs`, `apiReadyMs` | Client-observed acknowledgement/submission to completed status, including polling uncertainty |
| `providerRunnerMs` | Provider `metrics.inference_time`, only when supplied |
| `providerInferenceMs` | H3 `timings.inference` GPU denoising seconds converted to milliseconds, only when supplied |
| `resultRetrievalMs` | Fetching the completed result JSON |
| `mediaDownloadMs`, `mediaValidationMs`, `mediaReadyElapsedMs` | Download, local stream validation, and job-start-to-validated-media readiness |
| `boundaryFrameMs`, `frameExtractionMs`, `continuationReadyMs` | Actual-end-frame precomputation, any remaining boundary wait at the decision, and continuation preparation through validated media and its next boundary frame |
| `observationMs`, `observationToGenerationMs`, `observationToSubmitMs`, `observationToReadyMs` | Playback observation span and wait-inclusive elapsed time from playback observation start to local generation, provider submission and continuation readiness; submission remains unknown until an actual submission timestamp exists |
| `detectionMs`, playback events | Separate scene-understanding duration and browser readiness/playing/ended timestamps |

Unavailable provider timings remain unknown. Report measured samples, ready transitions, late-provider behavior and feedback delay separately. A finite queue cannot sustain generation that consistently produces fewer video seconds than it consumes; the held-frame bridge is explicit downtime, not a throughput fix. No model/duration/resolution speed claim is inferred from a provider name.

Signals, ticks, detections, profiles, decisions and story state remain under `data/adaptive/<session>/`. Early decision evidence is frozen separately from the complete playback-tail/EEG log. Keep the backend running briefly after playback to finish that log. Browser reload reconnects to the current in-memory session but does not restore an exact mid-clip position; restart the adaptive session for a clean rehearsal. Backend restart retains media/jobs/logs but does not restore or continue an adaptive session. Paid submissions are not blindly retried; uncertain requests remain blocked pending inspection of Fal history.

### Free rehearsal

```sh
GOZ_DEMO=1 GOZ_GAZE=sim GOZ_EEG=sim GOZ_SIM_BIAS=0.95 \
GOZ_DATA_DIR="$PWD/output/demo-data" PORT=3211 .venv/bin/python -m backend
```

Open [the rehearsal](http://127.0.0.1:3211/adaptive.html). **DEMO / SIM** uses synthetic clips/targets/signals through the same downstream analysis, policy and player, with no paid calls, including the optional identity verifier. `GOZ_SIM_SEED=acceptance` makes rehearsal samples repeatable. `GOZ_SIM_FAVORITE=0` selects SpongeBob and `1` Patrick for the prefilled order. Simulated sensors with `GOZ_DEMO=0` make real paid H3 requests and are labelled **SIMULATED SENSORS · LIVE VIDEO**; they do not verify physical gaze or Muse accuracy. `GOZ_GAZE` supports `gazekit|sim|off`; `GOZ_EEG` supports `mindmonitor|muse|sim|off`.

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
- `backend/adaptive/mindmonitor.py`: Mind Monitor UDP/OSC receiver, paired alpha/beta freshness, contact/artifact quality and relative physiological observations.
- `backend/sensor_setup.py`, `backend/gaze_worker.py`: sensor process ownership, atomic/versioned calibration persistence, fresh-process reload, independent validation/recentering and Gazekit integration. `GET /api/sensors` reports diagnostics; `POST /api/sensors/setup` retries; `POST /api/sensors/gaze-check` checks/recenters while idle; `DELETE /api/sensors/gaze-calibration` removes only the selected manifest and starts fresh calibration.
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

The suites separate deterministic replay/fake-provider tests from physical hardware and live provider checks. They cover comparative gaze gates and hysteresis, Patrick-versus-SpongeBob exact request differences, exposure/dwell synchronization, detection abstention, media validation/timing provenance, calibration serialization/schema compatibility, OSC/LSL quality handling, ordered queues, duplicate requests, cancellation/restart and malformed/late results. `tests/test_adaptation_acceptance.py` exercises the same downstream replay → fusion → profile → prompt → payload pipeline as live input. A changed request proves the controller path, not that the generated video visibly follows it.

There is no configured lint command. Run Python tests, JavaScript tests, syntax checks, TypeScript checks and `git diff --check`. Live acceptance must additionally inspect actual camera predictions, positive/negative episode frames, generated videos/audio and at least three automatic playback transitions. Record live sample counts and hardware blockers separately from passing deterministic tests. Calibration reload alone does not verify fresh prediction accuracy; a mocked connection alone does not verify Muse streaming.

Provider schema references: [Florence dense region captions](https://fal.ai/models/fal-ai/florence-2-large/dense-region-caption/api), [H3 image-to-video](https://fal.ai/models/minimax/h3-max-turbo/image-to-video/api), [H3 text-to-video](https://fal.ai/models/minimax/h3-max-turbo/text-to-video/api), [Fal queue timing](https://docs.fal.ai/model-apis/model-endpoints/queue).


### Acceptance evidence from this repair

These are local observations, not claims of production accuracy or continuous throughput. Files below are ignored local verification artifacts; personal calibration/sensor data is not committed.

| Reported issue | Confirmed cause and fix | Evidence / present limit |
| --- | --- | --- |
| Absent characters on background | Per-name Florence grounding forcibly assigned its largest returned box. `adaptive/tracks.py` now uses unprompted regions, geometric references, abstention, shot expiry and optional independent `adaptive/identity.py` pixel checks. | Actual positive/negative/occlusion fixtures; strict unknown on unsupported evidence. One occluded-character miss remains. Generated frames exposed franchise-name confusion, so captions alone never prove identity. |
| Playback gaps / suspected Luna delay | Readiness previously waited for detection; the single video element loaded only after the prior clip ended. A live run also exposed 3.60s of last-frame preparation omitted from readiness estimates. `adaptive/session.py`, `adaptive.js` and the reused two-element queue measure preparation, decouple detection, preload media and label waits. | The reported 20s H3 delay was not reproduced. Identity verification itself took about 12–20s on moving-clip batches and can delay adaptation; no claim that the entire pipeline sustains continuous generation. |
| Muse cannot connect | No headset was initially advertising; supported local MuseLSL/Bleak bridge and optional setup now have bounded discovery, reconnect and fresh-stream diagnostics. | Physical 256Hz streaming verified; clipped/noisy channels currently have zero confidence and do not influence decisions. Adjust contacts/fit and obtain 60s of clean baseline for EEG use. |
| Calibration lost | The upstream results window blocked registration; metadata/version/camera compatibility were missing and validation probes were reused for training. | iPhone calibration 82.0 screen-point mean error across 6 held-out targets, 886 training samples; actual backend/worker restarts reloaded saved model and produced 28.5–30Hz predictions. Rejected FaceTime attempts and existing artifacts were preserved. |
| No gaze marker | Missing/incorrect screen-to-video attribution, stale points and single-player transitions were not handled together. | Real unsnapped phone-gaze marker inspected in fullscreen; window/contain/cover/letterbox/resize tests, stale session/clip rejection and invalid/outside states. Primary display at 100% browser zoom is the supported setup; move pointer over the window to anchor its screen origin. |
| No visible adaptation | Remote enum selection plus subtle appended cues could leave the provider’s main shot unchanged. A live test also caught premature submission before target dwell reached the policy threshold. `adaptive/profile.py`, `adaptive/director.py` and `adaptive/session.py` now require an actionable policy preview before early submission and explicitly change the main shot. | Same-source Patrick/SpongeBob replay requests generated visibly different sustained close-ups. Both violated the requested closed-box story constraint; provider compliance remains imperfect. First moving-video real-gaze run had 10.756s valid gaze but insufficient shared-character evidence, so its continuation honestly stayed balanced. |

Paid comparison media/payloads/timings: `output/acceptance-live/paired-report.json` and the corresponding contact sheets. Four automatically played rehearsal clips, three browser-event transition delays 0/1/0ms, and ~2.9s decision-window-to-playback delay are recorded in `browser-rehearsal.json`; this is synthetic footage/input and not a provider throughput measurement. Real phone-gaze→analysis→balanced request→generated continuation→automatic playback is recorded in `live-session.json`; that run used an explicit held-frame bridge for 8.21s, with 8.74s from frozen feedback window to playback. Missing/poor EEG did not break the gaze path. The latency measurements use browser events, not external photodiode measurements.

The two emphasis videos changed the requested primary shot, but both opened the box despite instructions. A changed prompt is not proof of story fidelity. Inspect generated footage before presenting the demo; script outcomes in the dashboard describe intended story state, not a verified video transcript.

Final conservative detection checks retained Patrick in 6/8 sampled frames of the Patrick-focused generated clip and SpongeBob in 4/8 of the SpongeBob-focused clip. No incorrect retained identities were observed in those samples, but empty/prop-only Florence regions still suppress some correct identity observations. An indexed seven-image fixture batch identified all seven identity sets in 10.44s; that is a small test set, not a general accuracy benchmark. Lower reasoning was rejected after it invented a character on a negative frame.

The first controlled still-reference live check exposed the premature submission: its frozen window had 2.228s shared visibility but only 0.537s Patrick dwell. Offline replay of the complete physical recording yielded 14.355s valid gaze, 4.5s shared visibility and 2.64s Patrick dwell versus 0.434s SpongeBob dwell, correctly selecting Patrick. This replay did not change the already-submitted balanced request. Evidence: `controlled-session.json` and `controlled-full-window-replay.json` under `output/acceptance-live/`.

After the trigger fix, the final physical-phone run observed valid outside-video gaze and selected **slower pacing**, changed the actual provider request, generated a continuation and played it automatically. It did not establish a character preference. Its opening was explicitly a 20-second still-reference check; only the continuation was live-generated. Final evidence: `controlled-final-session.json`, `phone-final-continuation-contact.jpg` and `final-dashboard.jpg`. The output was inspected; it contains readable gestures and steady framing, but a causal speed difference is not established by that one video, and it still opens the box. Live character-emphasis verification remains incomplete; the contrasting emphasis videos used replay input.

Five new 15-second 480P H3 generations had job-start-to-media-readiness times **7.22, 5.60, 5.92, 6.44 and 13.73 seconds** (n=5). The last run additionally measured **3.77s frame extraction**, **0.05ms composition**, **397ms submission**, **11.44s observed provider elapsed**, **11.41s provider runner**, **1.58s GPU denoising**, **1.09s download** and **48ms media validation**. Full continuation preparation was **17.50s**, its held-frame transition delay **4.93s**, and frozen-feedback-to-playback delay **17.70s**. These scopes overlap and must not be summed. This sample demonstrates why continuous generation cannot be claimed from fast GPU timing or a finite buffer.

Final verification: **211 Python tests**, **28 JavaScript tests**, `npm run check`, `npm run typecheck`, and `git diff --check` passed. There is no separate frontend build or lint script. Browser rehearsal played four clips automatically (synthetic media/input); physical-input runs played an opening plus one generated continuation. Timeout/malformed/out-of-order/stop-restart/slow-generation faults are regression-tested, not claimed as new paid-provider failures. For remaining live acceptance, use **Check gaze** in the actual seating position, keep both characters verifiably visible long enough for comparative dwell, and inspect the resulting focus video. For EEG, adjust contacts and obtain 60 clean baseline seconds; fresh streaming alone is not sufficient.
