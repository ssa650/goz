"""Measured input/mapping diagnostics, independent of identity and policy gates."""
from collections import Counter
import math


def summarize(samples, ticks, timeline):
    flags = Counter()
    for s in samples:
        if not s.get("valid", True): flags["sensor_invalid"] += 1
        if s.get("blink"): flags["blink_gated"] += 1
        if not s.get("face", True): flags["no_face"] += 1
        if s.get("stale"): flags["sensor_stale"] += 1
        yaw = s.get("yaw", 0)
        if isinstance(yaw, (int, float)) and abs(yaw) > 25: flags["yaw_gate"] += 1
    ages = sorted(s["sentAt"]-s["t"] for s in samples
        if all(type(s.get(k)) in (int,float) and math.isfinite(s[k]) for k in ("sentAt", "t"))
        and s["sentAt"] >= s["t"])
    mappings = [t["mapping"] for t in ticks if isinstance(t.get("mapping"), dict)]
    return dict(received_samples=len(samples), aligned_samples=len(timeline), unaligned_samples=len(samples)-len(timeline),
        states=dict(Counter(e["state"] for e in timeline)), flags=dict(flags),
        mapping_ticks=len(mappings), invalid_mapping_ticks=sum(not m.get("valid") for m in mappings),
        mapping_methods=sorted({m.get("method") for m in mappings if isinstance(m.get("method"), str)}),
        screen_scales=sorted({m["scale"] for m in mappings if isinstance(m.get("scale"), (int,float))}),
        coordinate_spaces=sorted({value if isinstance(value, str) else "unspecified" for value in
            (s.get("coordinateSpace",s.get("coordinate_space", "unspecified")) for s in samples)}),
        paused_ticks=sum(not t.get("playing") for t in ticks),
        playback_epochs=sorted({t.get("epoch", 0) for t in ticks})[:32],
        capture_delayed_over_500ms=sum(age>.5 for age in ages),
        capture_to_send_ms_p95=round(ages[min(len(ages)-1, int(.95*len(ages)))]*1000,3) if ages else None,
        capture_to_send_ms_max=round(max(ages)*1000, 3) if ages else None,
        pipeline_diagnostics=next((s["pipelineDiagnostics"] for s in reversed(samples)
                                   if isinstance(s.get("pipelineDiagnostics"), dict)), None),
        receipt_diagnostics=next((s["receiptDiagnostics"] for s in reversed(samples)
                                  if isinstance(s.get("receiptDiagnostics"), dict)), None),
        blink_reasons=dict(Counter(s["blinkDiagnostics"].get("reason", "unspecified") for s in samples
                                  if isinstance(s.get("blinkDiagnostics"), dict))),
        camera_mirror_verified=None, eye_calibration_verified=None,
        note="Mapping fit does not validate eye calibration; raw invalid/blink flags are preserved.")
