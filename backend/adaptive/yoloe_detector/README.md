# Optional local YOLOE experiment

Disabled by default. Importing this package loads no vendor runtime or weights and makes no network request. This experiment did **not** establish reliable character recognition; keep it out of automatic gaze attribution. The integration owner controls selection/UI and shared files.

`YOLOECharacterDetector(names, config).step(rgb_uint8_pixels, media_timestamp_s)` returns video-normalized records with `boxes`, diagnostic `regions`, `unknown`, `shot_id`, `cut`, `valid_until`, and `experimental=True`. Generic `object0`/`object1` classes are checked and mapped to the labelled reference identities. Named results use `experimental_visual_prompt`, never `verified_reference`. Scores are uncalibrated. Every step has fresh inference; blank, failed, ambiguous, or low-score evidence cannot inherit an older identity.

`await detect_yoloe(video, names, clip_id=..., session_id=..., generation_id=..., on_progress=..., config=...)` returns chronological records and cumulative progress. Use `DetectorConfig.runtime_python` for a separate Python environment: GOZ's current Python 3.14 environment has no Torch/Ultralytics. The optional isolated child uses JSON lines and cooperative stdin-EOF stop, not a global interpreter change. One host-process-global slot, 1 Torch CPU thread, 1 decoder thread, bounded output queue, at most 4fps/120 media seconds. Default 2fps, 640px, queue4, wall45s; scheduler negotiated queue2. Cancellation discards later callbacks and retains the slot until actual child exit. Native inference cannot be forcibly interrupted under the no-process-kill constraint. The cache is per clip worker, not across clips; cold startup must never be awaited in playback or generation.

The default `prompt_strategy='scene'` uses the existing `00-30.jpg` image and its two already labelled reference boxes. The annotation snapshot is tested against `tracks.REFERENCE_CROPS`. `prompt_strategy='atlas'` keeps the failed five-crop experiment available for comparison. No new user boxes, training, fabricated target tracks, or cloud recognition are involved.

## Reproducible local runtime

Validated on Apple M3, macOS 26.3.1, Python3.12.12: Ultralytics8.4.36 / Torch2.10.0 / torchvision0.25.0. Install the exact requirements from `runtime.lock.txt` in a **new venv**, never upgrade the app/global environment. Runtime assets are stored outside Git. Do not load arbitrary checkpoints: `validated_weights()` requires the known official model filename and pinned official SHA256 before the vendor's pickle-capable loader.

Official checkpoint: https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt

SHA256: `8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3` (verified against official GitHub release API asset digest).

An explicit local configuration needs `weights`, `weights_sha256`, `runtime_python`, `license_reviewed=True`, `device='cpu'`, and `queue_size=2`. License acknowledgement is an explicit user decision, not a determination that an arbitrary distribution complies. Runtime network checks, autoinstall and vendor telemetry sync are disabled. No text prompt/CLIP download is used.

Ultralytics YOLOE code and pretrained weights are AGPL-3.0 by default; separate Enterprise licensing is available. Attribution: https://github.com/ultralytics/ultralytics . Retained notice: `ULTRALYTICS-LICENSE.txt`. Read GOZ README's optional YOLOE licensing section, https://www.ultralytics.com/license and https://www.gnu.org/licenses/agpl-3.0.html before distributing/hosting a covered integration. This experiment changes no whole-project license, publishes no repository, and accepts no Enterprise agreement.

## Actual results and limits

24 visually reviewed existing generated frames, 35 visible named instances, at IoU>=0.5; three separate 15s clips, 30 chronological frames each through14.5s. Scene3@0 was excluded because it repeats a labelled reference. Approximate Codex-reviewed annotation boxes are **not independent human gold**. Threshold0.45 was fixed before viewing predictions. Scene strategy was selected after atlas failed the known-reference sanity check, so this comparison is development evaluation, not an unbiased final acceptance set.

| Method | Precision | Recall | Unknown rate | Wrong identity labels |
|---|---:|---:|---:|---:|
| Existing SIFT, same24 frames |100% (3 positives)|8.57%|91.43%|0|
| YOLOE five-crop atlas |100% (3 positives)|8.57%|91.43%|0|
| YOLOE standard reference scene |68.75% (16 positives)|31.43%|57.14%|4|

The scene strategy correctly named both in the known reference (SpongeBob0.853 / Patrick0.451), but failed novel poses: scene1 recall0%, scene3 recall66.67%/precision90.91%, scene5 recall6.67%/precision20%. Zero observed switches does not excuse the four incorrect identities. No confidence/accuracy guarantee is made; no threshold was lowered to force labels.

Warm CPU sampled-frame latency p50=.338s, p95=.688s; three full15s replay walls14.05/10.15/10.98s. Process CPU11.53/11.31/11.29s (about one core); peak process RSS~802MB. Later startup import3.76s + model constructor5.09s, already beyond the3.5s decision window; first cold native import was about70s and model constructor17.44s. Native MPS is actually supported outside the tool sandbox, but first MPS inference took46.73s versus1.145s initial CPU inference with the failed atlas; that is cold startup evidence, not a warm MPS throughput comparison. MPS is not recommended for this experiment's live path. Brief CPU benchmark startup overlapped the end of MPS smoke; the corrected scene benchmark ran after the smoke completed. Actual gaze/EEG competition was not run; existing sensors/session were not restarted.

Evidence and annotated contact sheets: `output/yoloe-experiment/` (ignored, never staged). Tests: `tests/test_yoloe_detector.py` cover references, map/geometry/abstention, cuts/no inherited identity, official-hash/license gate, bounded config, fallback, cooperative cancellation/slot retention and metric semantics. Florence/OpenCV options are preserved by the integration owner. A persistent warm worker was deliberately deferred by the parent because accuracy failed first.
