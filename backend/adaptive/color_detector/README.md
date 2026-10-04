# Experimental two-character color detector

`from backend.adaptive.color_detector import ColorCharacterDetector, ColorDetectorConfig`

```python
detector = ColorCharacterDetector(
    ["SpongeBob", "Patrick"],
    clip_id=clip_id, session_id=session_id, generation_id=generation_id,
)
record = detector.step(rgb_uint8_frame, media_seconds)
```

`step` is synchronous CPU-only and requires RGB uint8 H×W×3 and strictly increasing finite timestamps. The default config is `fps=4.0`, `image_size=640` (long edge), `max_candidates_per_identity=8`. Config caps fps at4, resolution at640 and candidates at8 per supported identity. Unknown names are rejected; an allowed name never supplies presence. `reset(clip_id=..., session_id=..., generation_id=...)` clears timestamps, cut/continuity state and previous provenance. Instances belong to one clip/session and must be reset/recreated when a host worker changes that context.

The record has normalized `boxes`, candidate `regions`, `unknown`, `t`, `media_timestamp_s`, `shot_id`, `cut`, `valid_until=t+min(0.5,1/fps)`, `status`, `abstention_reason`, `clip_id`, `session_id`, `generation_id`, `inference_seconds`, and resource metadata. Exact `source` is `opencv_color_shape_demo`; accepted region `identity_status` is `demo_color_shape`. `provider_confidence_available=False` and `experimental=True`: this is an uncalibrated, domain-specific visual heuristic, not reference verification.

HSV plus Lab produces yellow and pink/peach skin masks. Connected contours supply actual colored-pixel area/fill, aspect, convex-hull solidity and distance-transform thickness. Thickness uses actual skin pixels, never a filled outline that would turn a thin cloud into a solid body. White eye support and nearby brown/green clothing help reject scenery. Similar disjoint candidates abstain; contained body parts prefer the larger torso. Clothes extend the approximate skin box toward the visible body. Temporal overlap only breaks score ties after visual gates; boxes are never inherited across absent frames. Uniform frames, detected cuts and long gaps clear continuity. Cut detection is a thumbnail difference heuristic and can miss gradual/low-difference transitions or flag rapid motion.

The module makes no network/model calls, trains nothing, launches no processes, modifies no global OpenCV thread settings, and does not decode video. The shared integration owner controls the host-wide single worker, CPU/decoder thread limits, 4fps cadence, queue2, playback cursor/stale-work dropping, cooperative cancellation, expiry and immutable 3.5s decision evidence. The worker must pass RGB, use media time, retain provenance, and never await full-clip processing in generation/playback.

## Frozen offline evidence

Detector SHA256: `1269741e22dea50a24125cefaf3afe78ac8e60917094aeaf6f2c99991b4f461f`.

Existing `output/yoloe-experiment/groundtruth-24.json` provides approximate Codex-reviewed annotations on generated clips, not independent human gold. Scene3 (8frames/15instances) and existing reference pairs were designated tuning data. Scenes1/5 (16frames/20instances) were reserved before their predictions/images were inspected. Thresholds were frozen before held-out execution and **were not retuned after failures**.

At identity match and IoU≥0.5:

| Data | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Tuning scene3 |15|0|0|100%|100%|
| Held-out scenes1/5 |14|1|6|93.3%|70%|

Early scene1 at0/0.5/1s produced no named boxes. Both existing reference pairs, the real background negative and three real upper-frame pink/yellow cloud negative crops passed. Nineteen independent regression tests passed, including synthetic thin outlines, unsupported colored blobs, absent Patrick, duplicate-body ambiguity, cut/blank/reset/provenance, invalid inputs and bounded work.

Full 15-second replay sampled60 chronological quarter-second frames per clip (180total), including late shots. OpenCV1/decoder1, 640long edge: p50 inference17.6–18.6ms, p9521.4–26.2ms; cold Lab initialization caused a219ms maximum. Seek/decode/inference replay took4.54–5.24wall seconds per15s clip,3.40–4.31process CPU seconds,first progress15–26ms. This is an offline replay with repeated seeks, not a live EEG/gaze concurrency result. The initial decoder-unbounded measurement is retained separately and must not be used to claim bounded CPU.

**Observed failures:** scene1 has two false Patrick boxes at4.25s (net/background) and5.25s (pineapple door). Scene5@2s produces a tiny false SpongeBob box on Patrick's shorts. Small/distant SpongeBob, raised arms and disconnected limbs are often missed. Clothes/body bounds remain approximate. The spatial-IoU metric reports zero identity errors, but that does **not** mean no wrong-object identities: these small wrong-object boxes do not overlap the true body by0.5. Cut flags also differed from the annotation shot IDs. Never describe the unlabelled full-clip samples as verified accuracy.

These measurements support a fast experimental testing default with explicit unknowns and rollback. They do not establish reliable general character recognition or readiness for identity-critical gaze attribution.

Evidence and rerunnable harness: `output/color-detector-experiment/`. Tests: `tests/test_color_detector.py`. No other provider is removed; integration/default selection is owned separately.
