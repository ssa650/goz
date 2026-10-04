"""Synchronize captured gaze with presented video and integrate valid elapsed time.

All wall/capture timestamps reaching this module are epoch seconds in the server
clock domain. Playback ticks carry content bounds in screen CSS pixels; no DPR
conversion is applied here. Sensor capture time, never arrival time, picks a tick.
EEG remains a physiological observation, never automatic preference evidence.
"""
import math
import numpy as np

from .tracks import boxes_at, frame_at, target_at
from . import eeg_policy

TICK_TOLERANCE_S = 0.6
MAX_SAMPLE_GAP_S = 0.2
FIXATION_S = 0.3
EEG_LAG_S = (0.3, 2.0)
LOOK_AWAY_YAW = 25
STRONG_ATTENTION = 0.4
STRONG_EEG_Z = 0.5


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _id(item, key):
    return item.get(key + "Id", item.get(key + "_id"))


def _location(sample, ticks):
    if not _finite(sample.get("t")):
        return None
    t = sample["t"]
    candidates = [k for k in ticks if k["wall"] <= t + 1e-6 and all(
        _id(sample, kind) is None or _id(k, kind) is None or _id(sample, kind) == _id(k, kind)
        for kind in ("session", "clip"))]
    if not candidates:
        return None
    tick = max(candidates, key=lambda k: k["wall"])
    if t - tick["wall"] > TICK_TOLERANCE_S:
        return None
    rate = tick.get("playbackRate", 1)
    if not _finite(rate) or rate <= 0:
        return None
    video_t = tick["video_t"] + ((t - tick["wall"]) * rate if tick.get("playing") else 0)
    if sample.get("coordinate_space") == "video-normalized":
        nx, ny = sample.get("nx"), sample.get("ny")
    else:
        r = tick.get("rect")
        if not isinstance(r, dict) or not all(_finite(r.get(k)) for k in ("x", "y", "w", "h")) or r["w"] <= 0 or r["h"] <= 0:
            return None
        x, y = sample.get("x"), sample.get("y")
        if not _finite(x) or not _finite(y):
            return None
        nx, ny = (x - r["x"]) / r["w"], (y - r["y"]) / r["h"]
    if not _finite(nx) or not _finite(ny):
        return None
    on_video = 0 <= nx <= 1 and 0 <= ny <= 1
    visible = tick.get("visible_rect", tick.get("visibleRect"))
    if visible and sample.get("coordinate_space") != "video-normalized":
        on_video = on_video and visible["x"] <= sample["x"] <= visible["x"] + visible["w"] and visible["y"] <= sample["y"] <= visible["y"] + visible["h"]
    return video_t, nx, ny, on_video, tick


def locate(sample, ticks):
    located = _location(sample, ticks)
    return located[:3] if located and located[-1].get("playing") else None


def label(samples, ticks, track):
    """Frame-aligned observations with explicit validity/attribution states."""
    out = []
    for sample in sorted(samples, key=lambda s: s.get("t", 0)):
        located = _location(sample, ticks)
        if located is None:
            continue
        video_t, nx, ny, on_video, tick = located
        confidence = sample.get("confidence", 1.0)
        confidence = float(np.clip(confidence, 0, 1)) if _finite(confidence) else 0.0
        yaw = sample.get("yaw") or 0
        valid = (bool(sample.get("valid", True)) and sample.get("face", True) and
                 not sample.get("blink") and not sample.get("stale") and _finite(yaw) and
                 abs(yaw) <= LOOK_AWAY_YAW and confidence >= .3 and bool(tick.get("playing")))
        frame = frame_at(track, video_t, _id(tick, "clip"))
        boxes = frame.get("boxes", {}) if frame else {}
        target = target_at(boxes, nx, ny) if valid and on_video else None
        hits = [name for name, b in boxes.items() if b[0] <= nx <= b[2] and b[1] <= ny <= b[3]]
        attribution_available = bool(frame and frame.get("status", "observed") not in ("unavailable", "shot_boundary"))
        state = ("invalid" if not valid else "outside-video" if not on_video else
                 "unavailable" if not attribution_available else "ambiguous" if len(hits) > 1 else
                 "target" if target else "background")
        out.append(dict(t=sample["t"], capture_t=sample["t"], video_t=round(video_t, 6),
                        nx=nx, ny=ny, target=target, on_video=on_video, valid=valid, state=state,
                        attribution_available=attribution_available, visible_targets=list(boxes),
                        confidence=confidence, blink=bool(sample.get("blink")), yaw=yaw,
                        session_id=_id(tick, "session"), clip_id=_id(tick, "clip"),
                        epoch=tick.get("epoch", 0), playback_rate=tick.get("playbackRate", 1),
                        detection_t=frame["t"] if frame else None))
    return out


def _interval(a, b):
    """Observed elapsed valid seconds; never fill gaps or extrapolate a tail."""
    dt = b["t"] - a["t"]
    if not (0 < dt <= MAX_SAMPLE_GAP_S + 1e-6) or not a.get("valid") or not b.get("valid"):
        return 0.0
    if any(a.get(k) != b.get(k) for k in ("session_id", "clip_id", "epoch")):
        return 0.0
    media_dt = b["video_t"] - a["video_t"]
    if media_dt < 0 or abs(media_dt - dt * a.get("playback_rate", 1)) > .08:
        return 0.0
    return dt


def fixations(timeline):
    runs, current, start, last = [], None, None, None
    for i, e in enumerate(timeline):
        target = e.get("target") if e.get("valid") else None
        continuation = i > 0 and _interval(timeline[i - 1], e) > 0
        if target != current or not continuation:
            if current and last is not None and last - start >= FIXATION_S - 1e-6:
                runs.append((current, start, last))
            current, start = target, e["t"]
        last = e["t"]
    if current and last - start >= FIXATION_S - 1e-6:
        runs.append((current, start, last))
    return runs


def speaking_windows(timeline, beats):
    out = {}
    for beat in beats:
        window = [e["t"] for e in timeline if e.get("valid") and beat["t0"] <= e["video_t"] < beat["t1"]]
        if beat.get("speaker") and window:
            out.setdefault(beat["speaker"], []).append((window[0], window[-1]))
    return out


def analyze(timeline, eeg, track, names, plan=None, eeg_confidence=1.0, *, eeg_quality=None, eeg_window=None):
    """Exposure-normalized attention, comparative evidence, and separate EEG."""
    weights = [_interval(a, b) for a, b in zip(timeline, timeline[1:])] + ([0.0] if timeline else [])
    visible = {n: 0.0 for n in names}
    dwell = dict(visible)
    comparative_visible, comparative_dwell = dict(visible), dict(visible)
    valid_s = watched_s = comparison_s = outside_s = confidence_sum = 0.0
    for i, (e, dt) in enumerate(zip(timeline, weights)):
        if not dt:
            continue
        following = timeline[i + 1]
        valid_s += dt
        confidence_sum += dt * min(e.get("confidence", 1), following.get("confidence", 1))
        if not e["on_video"] or not following["on_video"]:
            if not e["on_video"] and not following["on_video"]:
                outside_s += dt
            continue
        watched_s += dt
        available = set(boxes_at(track, e["video_t"], e.get("clip_id"))) & set(boxes_at(track, following["video_t"], following.get("clip_id"))) & set(names)
        comparable = len(available) >= 2
        if comparable:
            comparison_s += dt
        for n in available:
            visible[n] += dt
            if comparable:
                comparative_visible[n] += dt
        target = e.get("target") if e.get("target") == following.get("target") else None
        if target in available:
            dwell[target] += dt
            if comparable:
                comparative_dwell[target] += dt
    clean = [(t, z) for t, _, z, artifact in eeg if not artifact and _finite(t) and _finite(z)]
    # Coverage reflects physiological sample availability, independently of magnitude.
    span = max(0, timeline[-1]["t"] - timeline[0]["t"]) if len(timeline) > 1 else 0
    clean = [(t, z) for t, z in clean if timeline and timeline[0]["t"] <= t <= timeline[-1]["t"] + EEG_LAG_S[1]]
    eeg_confidence = float(np.clip(eeg_confidence * min(1, len(clean) / max(8, span * 4)), 0, 1))
    clip_z = float(np.mean([z for _, z in clean])) if clean else 0.0
    fix = fixations(timeline)
    lines = speaking_windows(timeline, (plan or {}).get("beats", []))
    characters = {}
    for n in names:
        response = [z for name, onset, _ in fix if name == n for t, z in clean
                    if onset + EEG_LAG_S[0] <= t <= onset + EEG_LAG_S[1]]
        eeg_response = float(np.mean(response)) - clip_z if response else None
        spoken = [z for w0, w1 in lines.get(n, []) for t, z in clean if w0 + EEG_LAG_S[0] <= t <= w1 + EEG_LAG_S[0]]
        speaking = float(np.mean(spoken)) - clip_z if spoken else None
        attention = dwell[n] / visible[n] if visible[n] > .5 else 0.0
        characters[n] = dict(visible_s=round(visible[n], 3), dwell_s=round(dwell[n], 3),
            comparison_visible_s=round(comparative_visible[n], 3), comparison_dwell_s=round(comparative_dwell[n], 3),
            attention=round(attention, 3), fixations=sum(f[0] == n for f in fix),
            eeg_response=round(eeg_response, 3) if eeg_response is not None else None,
            speaking_response=round(speaking, 3) if speaking is not None else None,
            response=round(attention, 3), strong=False,
            physiological_observation="uncertain_association" if eeg_response is not None else "unavailable")
    beats = []
    for beat in (plan or {}).get("beats", []):
        indices = [i for i, e in enumerate(timeline) if beat["t0"] <= e["video_t"] < beat["t1"]]
        if not indices:
            continue
        w0, w1 = timeline[indices[0]]["t"], timeline[indices[-1]]["t"]
        values = [z for t, z in clean if w0 + EEG_LAG_S[0] <= t <= w1 + EEG_LAG_S[0]]
        beats.append(dict(beat, valid_s=round(sum(weights[i] for i in indices), 3),
            on_video_s=round(sum(weights[i] for i in indices if timeline[i]["on_video"]), 3),
            engagement_z=round(float(np.mean(values)) - clip_z, 3) if values else None))
    blinks = sum(b.get("blink", False) and not a.get("blink", False) for a, b in zip(timeline, timeline[1:]))
    return dict(watched_s=round(watched_s, 3), valid_gaze_s=round(valid_s, 3), comparison_s=round(comparison_s, 3),
        samples=len(timeline), characters=characters,
        gaze_confidence=round(confidence_sum / valid_s, 3) if valid_s else 0,
        eeg_confidence=round(eeg_confidence, 3),
        response_strength=round(max((c["attention"] for c in characters.values()), default=0), 3),
        eeg_mean_z=round(clip_z * eeg_confidence, 3), eeg_samples=len(clean),
        eeg_available=bool(clean and eeg_confidence > 0),
        eeg_policy=eeg_policy.decide(eeg_policy.observe(eeg, eeg_quality, eeg_window)),
        blink_rate_per_min=round(60 * blinks / span, 1) if span > 5 else None,
        look_away_frac=round(outside_s / valid_s, 3) if valid_s else None, beats=beats)
