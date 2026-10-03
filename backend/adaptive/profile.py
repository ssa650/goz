"""Viewer profile (what they respond to) and the adaptation decision.

Everything is a small, inspectable number so the dashboard can show
exactly which preference moved and why the next scene changes.
"""
import copy

LEARN = 0.5
STEP = 0.3
FOCUS_MARGIN = 0.12
GENRES = ("suspense", "action", "humor", "romance", "drama")


def new_profile(names):
    share = round(1 / max(len(names), 1), 3)
    return dict(characters={n: share for n in names}, pacing=0.0, dialogue=0.0,
                genres={g: 0.0 for g in GENRES}, clips=0)


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def update(profile, analysis):
    """Returns (new_profile, changes[]) — each change names the evidence."""
    p = copy.deepcopy(profile)
    changes = []
    chars = analysis["characters"]
    total = sum(c["response"] for c in chars.values())
    if total > 0:
        for name, c in chars.items():
            before = p["characters"].get(name, 0.0)
            target = c["response"] / total
            after = round((1 - LEARN) * before + LEARN * target, 3)
            p["characters"][name] = after
            if abs(after - before) >= 0.03:
                why = f"looked at {name} {round(100 * c['attention'])}% of the time they were on screen"
                if c["eeg_response"]:
                    why += f", EEG {c['eeg_response']:+.2f}σ after looking"
                if c.get("speaking_response") is not None:
                    why += f", EEG {c['speaking_response']:+.2f}σ while {name} spoke"
                changes.append(dict(key=f"character:{name}", before=before, after=after,
                                    strong=c["strong"], why=why))
    z, blink, away = analysis["eeg_mean_z"], analysis["blink_rate_per_min"], analysis["look_away_frac"]
    bored = (z < -0.3) + (blink is not None and blink > 25) + (away is not None and away > 0.25)
    if bored:
        before = p["pacing"]
        p["pacing"] = round(clamp(before + 0.25 * bored), 3)
        changes.append(dict(key="pacing", before=before, after=p["pacing"], strong=bored >= 2,
                            why=f"signs of low engagement: EEG {z:+.2f}σ, blinks {blink}/min, looked away {round(100 * (away or 0))}%"))
    dialogue = [b["engagement_z"] for b in analysis["beats"] if b.get("dialogue") and b["engagement_z"] is not None]
    silent = [b["engagement_z"] for b in analysis["beats"] if not b.get("dialogue") and b["engagement_z"] is not None]
    if dialogue and silent:
        delta = sum(dialogue) / len(dialogue) - sum(silent) / len(silent)
        if abs(delta) >= 0.2:
            before = p["dialogue"]
            p["dialogue"] = round(clamp(before + clamp(0.3 * delta, -STEP, STEP)), 3)
            changes.append(dict(key="dialogue", before=before, after=p["dialogue"], strong=abs(delta) > 0.6,
                                why=f"EEG {delta:+.2f}σ during dialogue vs. silent moments"))
    for genre in GENRES:
        zs = [b["engagement_z"] for b in analysis["beats"] if genre in b.get("tags", []) and b["engagement_z"] is not None]
        if zs:
            before = p["genres"][genre]
            p["genres"][genre] = round(clamp(before + clamp(0.3 * sum(zs) / len(zs), -STEP, STEP)), 3)
            if abs(p["genres"][genre] - before) >= 0.1:
                changes.append(dict(key=f"genre:{genre}", before=before, after=p["genres"][genre], strong=False,
                                    why=f"EEG {sum(zs) / len(zs):+.2f}σ during {genre} moments"))
    p["clips"] += 1
    return p, changes


def decide(profile, analysis):
    """What the next scene should change, with human-readable reasons."""
    decision = dict(focus=None, tension="same", dialogue="same", pacing="same", tone=None, event=False, reasons=[])
    ranked = sorted(profile["characters"].items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] >= FOCUS_MARGIN:
        name = ranked[0][0]
        c = analysis["characters"].get(name, {})
        decision["focus"] = name
        decision["reasons"].append(
            f"Focus on {name}: affinity {ranked[0][1]:.2f} vs {ranked[1][0]} {ranked[1][1]:.2f}"
            + (" (strong gaze + EEG response)" if c.get("strong") else ""))
    if profile["pacing"] >= 0.25:
        decision["pacing"] = "faster"
        decision["reasons"].append(f"Speed up pacing: pacing preference {profile['pacing']:+.2f}")
    if analysis.get("look_away_frac") and analysis["look_away_frac"] > 0.25 or analysis["eeg_mean_z"] < -0.3:
        decision["tension"] = "higher"
        decision["event"] = True
        decision["reasons"].append("Introduce a new event and raise tension: attention dropped during this scene")
    elif profile["genres"]["suspense"] >= 0.3:
        decision["tension"] = "higher"
        decision["reasons"].append(f"Increase tension: suspense preference {profile['genres']['suspense']:+.2f}")
    if profile["dialogue"] <= -0.2:
        decision["dialogue"] = "less"
        decision["reasons"].append(f"Reduce dialogue: dialogue preference {profile['dialogue']:+.2f}")
    elif profile["dialogue"] >= 0.2:
        decision["dialogue"] = "more"
        decision["reasons"].append(f"More dialogue: dialogue preference {profile['dialogue']:+.2f}")
    genre, weight = max(profile["genres"].items(), key=lambda kv: kv[1])
    if weight >= 0.25:
        decision["tone"] = genre
        decision["reasons"].append(f"Shift tone toward {genre}: {genre} preference {weight:+.2f}")
    if not decision["reasons"]:
        decision["reasons"].append("No clear preference yet: continue the story evenly")
    return decision
