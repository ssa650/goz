"""Match viewer signals to what was on screen at that moment.

A gaze sample at wall time t is placed on the video timeline with the
browser's playback ticks (wall time <-> video time + the video's on-screen
rect), then hit-tested against the character boxes at that video time.
EEG response to a character = mean engagement z in the window
[onset + EEG_LAG_S] after each fixation on them, relative to the clip mean.
"""
import numpy as np

from .tracks import boxes_at, target_at

TICK_TOLERANCE_S = 0.6
FIXATION_S = 0.3
EEG_LAG_S = (0.3, 2.0)
LOOK_AWAY_YAW = 25
STRONG_ATTENTION = 0.4
STRONG_EEG_Z = 0.5


def locate(sample, ticks):
    """(video_t, nx, ny) for a gaze sample, or None if not watching."""
    if not ticks:
        return None
    t = sample["t"]
    tick = min(ticks, key=lambda k: abs(k["wall"] - t))
    if abs(tick["wall"] - t) > TICK_TOLERANCE_S or not tick.get("playing"):
        return None
    r = tick["rect"]
    if not r or r["w"] <= 0 or r["h"] <= 0:
        return None
    return tick["video_t"] + (t - tick["wall"]), (sample["x"] - r["x"]) / r["w"], (sample["y"] - r["y"]) / r["h"]


def label(samples, ticks, track):
    """Per-sample timeline entries {t, video_t, nx, ny, target, on_video, valid}."""
    out = []
    for s in samples:
        where = locate(s, ticks)
        if where is None:
            continue
        video_t, nx, ny = where
        on_video = 0 <= nx <= 1 and 0 <= ny <= 1
        valid = bool(s.get("valid", True))
        target = target_at(boxes_at(track, video_t), nx, ny) if on_video and valid else None
        out.append(dict(t=s["t"], video_t=round(video_t, 3), nx=round(nx, 4), ny=round(ny, 4),
                        target=target, on_video=on_video, valid=valid,
                        blink=bool(s.get("blink")), yaw=s.get("yaw", 0.0)))
    return out


def fixations(timeline):
    """[(name, onset_t, end_t)] runs of the same target lasting >= FIXATION_S."""
    runs, current, start, last = [], None, None, None
    for e in timeline + [dict(t=float("inf"), target=None)]:
        if e["target"] != current:
            if current and last - start >= FIXATION_S:
                runs.append((current, start, last))
            current, start = e["target"], e["t"]
        last = e["t"]
    return runs


def speaking_windows(timeline, beats):
    """{speaker: [(wall_t0, wall_t1)]} for beats with a speaker, mapped from
    video time to wall time through the labelled gaze timeline."""
    out = {}
    for beat in beats:
        speaker = beat.get("speaker")
        window = [e["t"] for e in timeline if beat["t0"] <= e["video_t"] < beat["t1"]]
        if speaker and window:
            out.setdefault(speaker, []).append((window[0], window[-1]))
    return out


def analyze(timeline, eeg, track, names, plan=None):
    """Clip-level response summary used by the profile and the dashboard."""
    dt = np.diff([e["t"] for e in timeline]) if len(timeline) > 1 else np.array([])
    step = float(np.median(dt)) if len(dt) else 1 / 30
    watched = [e for e in timeline if e["valid"]]
    visible = {n: 0.0 for n in names}
    dwell = {n: 0.0 for n in names}
    for e in watched:
        for n in boxes_at(track, e["video_t"]):
            if n in visible:
                visible[n] += step
        if e["target"] in dwell:
            dwell[e["target"]] += step
    zs = [z for (_, _, z, a) in eeg if not a]
    clip_z = float(np.mean(zs)) if zs else 0.0
    fix = fixations(timeline)
    characters = {}
    lines = speaking_windows(timeline, (plan or {}).get("beats", []))
    for n in names:
        windows = [z for (name, onset, _) in fix if name == n
                   for (t, _, z, a) in eeg if not a and onset + EEG_LAG_S[0] <= t <= onset + EEG_LAG_S[1]]
        eeg_resp = float(np.mean(windows)) - clip_z if windows else 0.0
        spoken = [z for (w0, w1) in lines.get(n, []) for (t, _, z, a) in eeg
                  if not a and w0 + EEG_LAG_S[0] <= t <= w1 + EEG_LAG_S[0]]
        speaking = float(np.mean(spoken)) - clip_z if spoken else None
        attention = dwell[n] / visible[n] if visible[n] > 0.5 else 0.0
        boost = max(0.0, eeg_resp) + 0.5 * max(0.0, speaking or 0.0)
        characters[n] = dict(
            visible_s=round(visible[n], 2), dwell_s=round(dwell[n], 2), attention=round(attention, 3),
            fixations=sum(1 for f in fix if f[0] == n), eeg_response=round(eeg_resp, 3),
            speaking_response=round(speaking, 3) if speaking is not None else None,
            response=round(attention * (1 + boost), 3),
            strong=attention >= STRONG_ATTENTION and (eeg_resp >= STRONG_EEG_Z or (speaking or 0) >= STRONG_EEG_Z))
    span = (timeline[-1]["t"] - timeline[0]["t"]) if len(timeline) > 1 else 0.0
    blinks = sum(1 for a, b in zip(timeline, timeline[1:]) if b["blink"] and not a["blink"])
    away = [e for e in timeline if not e["on_video"] or abs(e.get("yaw") or 0) > LOOK_AWAY_YAW]
    beats = []
    for beat in (plan or {}).get("beats", []):
        window = [e for e in timeline if beat["t0"] <= e["video_t"] < beat["t1"]]
        if not window:
            continue
        w0, w1 = window[0]["t"], window[-1]["t"]
        bz = [z for (t, _, z, a) in eeg if not a and w0 + EEG_LAG_S[0] <= t <= w1 + EEG_LAG_S[0]]
        beats.append(dict(beat, engagement_z=round(float(np.mean(bz)) - clip_z, 3) if bz else None))
    return dict(
        watched_s=round(span, 1), samples=len(timeline), characters=characters,
        eeg_mean_z=round(clip_z, 3), eeg_samples=len(zs),
        blink_rate_per_min=round(60 * blinks / span, 1) if span > 5 else None,
        look_away_frac=round(len(away) / len(timeline), 3) if timeline else None,
        beats=beats)
