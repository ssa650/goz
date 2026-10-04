"""Experimental EEG feature -> categorical state -> bounded delivery cue.

Independently implemented after inspecting StiopaPopa/fable at commit
336348f24dcf6c76caa1864d70edd5cedfefea1e (backend/main.py and
OSC_focus_relax_smoother.py). No first-party license was found; no Fable code
or numeric classifier thresholds are copied. See EEG_POLICY_AUDIT.txt.

These states describe a feature relative to its own baseline, not focus,
relaxation, enjoyment or valence. Muse uses beta/(alpha+theta) robust z;
Mind Monitor uses its existing baseline-normalized, smoothed beta/alpha index.
Thresholds and cue mappings are experimental controller choices, not validated
physiological interpretations. Acquisition and calibration stay in the feeds.
"""
import math
import statistics

VERSION = "experimental-eeg-delivery-v1"
MIN_CONFIDENCE = .6
MIN_SPAN_S = 1.0
MAX_GAP_S = .5
FRESH_S = .75
BUCKET_S = .25
ENTER_Z = 1.0
RETAIN_Z = .5
AGREEMENT = .8


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def unavailable(reason, source=None):
    return dict(policy=VERSION, experimental=True, eligible=False, state="unknown",
                action="keep", reason=reason, source=source, median_z=None,
                valid_span_s=0.0, samples=0)


def observe(samples, quality=None, window=None):
    """Use only fresh, clean, calibrated features from the frozen clip window.

    The last clean contiguous run must span >=1s; 250ms buckets prevent packet
    bursts from masquerading as duration. Muse's 2s raw window must fit fully
    inside playback. No tail extrapolation or analysis wait is introduced.
    """
    q = quality if isinstance(quality, dict) else {}
    source = q.get("source")
    fail = lambda reason: unavailable(reason, source)
    if source not in ("muse", "mindmonitor", "sim"):
        return fail("Validated EEG source metadata unavailable")
    if (q.get("live") is not True or q.get("calibrated") is not True or q.get("qualityError")
            or not finite(q.get("confidence")) or q["confidence"] < MIN_CONFIDENCE
            or q.get("connectionState") not in ("streaming", "simulated")):
        return fail("EEG quality, calibration or live-sample gate failed")
    if source in ("muse", "sim") and (not finite(q.get("sampleAgeSeconds"))
            or not -.5 <= q["sampleAgeSeconds"] < 3):
        return fail("EEG raw samples stale or timestamp unavailable")
    if source == "mindmonitor":
        # Contacts alone cannot rule out clipped/raw movement artifacts.
        if q.get("artifactCoverage") != "raw clipping/movement and contacts":
            return fail("Mind Monitor raw artifact checks unavailable")
        if any(not finite(q.get(k)) or not 0 <= q[k] < 3
               for k in ("contactAgeSeconds", "alphaAgeSeconds", "betaAgeSeconds")):
            return fail("Mind Monitor contact or band samples stale")
    if (not isinstance(window, (tuple, list)) or len(window) != 2
            or not all(finite(t) for t in window) or window[1] <= window[0]):
        return fail("Frozen EEG observation window unavailable")
    start, end = window
    feature_start = start + (2.0 if source in ("muse", "sim") else 0.0)
    if not isinstance(samples, (tuple, list)):
        return fail("EEG features unavailable")
    rows = sorted((row for row in samples if isinstance(row, (tuple, list)) and len(row) == 4
                   and finite(row[0]) and feature_start - 1e-6 <= row[0] <= end + 1e-6),
                  key=lambda row: row[0])
    run = []
    for row in rows:
        t, value, z, artifact = row
        if (artifact is not False or not finite(value) or value <= 0
                or not finite(z) or abs(z) >= 5):
            run = []  # Clipped z scores and artifact rows cannot enter policy.
            continue
        if run and t == run[-1][0]:
            if (value, z) != (run[-1][1], run[-1][2]):
                return fail("Conflicting EEG features at one timestamp")
            continue
        if run and t - run[-1][0] > MAX_GAP_S + 1e-6:
            run = []
        run.append(row)
    if not run or end - run[-1][0] > FRESH_S:
        return fail("No fresh clean EEG feature run at the deadline")
    span = run[-1][0] - run[0][0]
    if span < MIN_SPAN_S - 1e-6 or span < .6 * max(0, end - feature_start):
        return fail("Insufficient clean EEG duration in this playback window")
    buckets = {}
    for t, _, z, _ in run:
        bucket = math.floor((t - feature_start + 1e-6) / BUCKET_S)
        buckets.setdefault(bucket, []).append(z)
    values = [statistics.median(v) for v in buckets.values()]
    if len(values) < 5:
        return fail("Insufficient distinct EEG time buckets")
    fractions = {name: sum(sign * z >= threshold for z in values) / len(values)
                 for name, sign, threshold in (("above_enter", 1, ENTER_Z),
                     ("below_enter", -1, ENTER_Z), ("above_retain", 1, RETAIN_Z),
                     ("below_retain", -1, RETAIN_Z))}
    return dict(policy=VERSION, experimental=True, eligible=True, source=source,
                feature=("beta/(alpha+theta) robust z" if source in ("muse", "sim")
                         else "smoothed beta/alpha baseline index"),
                median_z=round(statistics.median(values), 4), fractions=fractions,
                valid_span_s=round(span, 4), samples=len(values),
                confidence=q["confidence"], window=[start, end])


def decide(observation, previous=None):
    """Hysteresis uses current evidence; an old state alone never changes a cue."""
    if not isinstance(observation, dict) or not observation.get("eligible"):
        return dict(observation) if isinstance(observation, dict) else unavailable("EEG evidence unavailable")
    result = dict(observation)
    state, action = "near_baseline", "keep"
    fractions = result["fractions"]
    for label, action_name, prefix in (("above_baseline", "faster_pacing", "above"),
                                      ("below_baseline", "slower_pacing", "below")):
        mode = "retain" if previous == label else "enter"
        if fractions[prefix + "_" + mode] >= AGREEMENT:
            state, action = label, action_name
            break
    result.update(state=state, action=action,
        reason=(f"Experimental EEG feature {state.replace('_', ' ')} over {result['valid_span_s']:.1f}s "
                f"of clean playback features (median index {result['median_z']:+.2f}); "
                "try a small delivery cue; cause and valence unknown" if action != "keep" else
                "EEG feature near baseline or mixed; preserve scripted delivery"))
    return result


def delivery_cue(policy, pacing="same"):
    """At most one EEG cue; current gaze readability pacing has precedence."""
    if (not isinstance(policy, dict) or policy.get("policy") != VERSION
            or not policy.get("eligible") or policy.get("suppressed_by")
            or pacing in ("faster", "slower")):
        return ""
    return {
        "faster_pacing": "EEG DELIVERY TRIAL: Slightly quicken the existing gestures in the interior of this clip. Keep camera staging readable and retain all scripted actions, dialogue wording, cast and outcome. Preserve the required first and last frame compositions.",
        "slower_pacing": "EEG DELIVERY TRIAL: Let the existing gestures unfold slightly more slowly in the interior of this clip. Keep camera staging steady and retain all scripted actions, dialogue wording, cast and outcome. Preserve the required first and last frame compositions.",
    }.get(policy.get("action"), "")
