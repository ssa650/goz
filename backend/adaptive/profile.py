"""Conservative, deterministic observations and tentative attention preferences.

EEG is a physiological observation, never a label for enjoyment or a causal
explanation. Character learning uses gaze during simultaneous visibility so
looking at the only visible actor does not establish a comparative preference.
"""
import copy
import math

from . import eeg_policy

LEARN = 0.5
FOCUS_MARGIN = 0.12
MIN_VALID_S = 2.0
MIN_COMPARISON_S = 2.0
MIN_DWELL_S = 1.25
MIN_CONFIDENCE = 0.6
MIN_READABILITY_S = 3.0
MIN_OUTSIDE_S = 1.4  # Preserve the old 4s * .35 absolute outside-video requirement.
ENTER_MARGIN = 0.25
RETAIN_MARGIN = 0.12
SWITCH_MARGIN = 0.35
GENRES = ("suspense", "action", "humor", "romance", "drama")


def new_profile(names):
    share = round(1 / max(len(names), 1), 3)
    return dict(characters={n: share for n in names}, pacing=0.0, dialogue=0.0,
                genres={g: 0.0 for g in GENRES}, clips=0,
                label="tentative attention preference", policy=dict(focus=None))


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else 0.0


def evidence(analysis, names):
    """Missing exposure/quality fields fail closed; no script-derived evidence."""
    valid_s = number(analysis.get("valid_gaze_s"))
    comparison_s = number(analysis.get("comparison_s"))
    confidence = number(analysis.get("gaze_confidence"))
    eligible = {}
    if valid_s >= MIN_VALID_S and comparison_s >= MIN_COMPARISON_S and confidence >= MIN_CONFIDENCE:
        for name, c in analysis.get("characters", {}).items():
            exposure = number(c.get("comparison_visible_s"))
            dwell = number(c.get("comparison_dwell_s"))
            if name in names and exposure >= MIN_COMPARISON_S:
                eligible[name] = dict(visible_s=exposure, dwell_s=dwell,
                                      attention=clamp(dwell / exposure, 0, 1))
    return dict(valid_s=valid_s, comparison_s=comparison_s, gaze_confidence=confidence,
                characters=eligible, sufficient=len(eligible) >= 2)


def focus_from(e, previous=None):
    if not e["sufficient"]:
        return None
    ranked = sorted(e["characters"].items(), key=lambda kv: kv[1]["attention"], reverse=True)
    name, best = ranked[0]
    margin = best["attention"] - ranked[1][1]["attention"]
    threshold = RETAIN_MARGIN if name == previous else SWITCH_MARGIN if previous else ENTER_MARGIN
    return name if best["dwell_s"] >= MIN_DWELL_S and margin >= threshold else None


def readability_evidence(e):
    return e["valid_s"] >= MIN_READABILITY_S and e["gaze_confidence"] >= MIN_CONFIDENCE


def outside_evidence(e, analysis):
    away = analysis.get("look_away_frac")
    return (away is not None and number(away) >= .35
            and e["valid_s"] * number(away) + 1e-6 >= MIN_OUTSIDE_S)


def update(profile, analysis):
    """Returns (new_profile, changes[]), with auditable observation gates."""
    p, changes = copy.deepcopy(profile), []
    e = evidence(analysis, p["characters"])
    previous = p.get("policy", {}).get("last_focus") or p.get("policy", {}).get("focus")
    focus = focus_from(e, previous)
    previous_eeg = p.get("policy", {}).get("eeg_state")
    p["policy"] = dict(focus=focus, last_focus=focus or previous, previous_focus=previous, evidence=e)
    eeg = eeg_policy.decide(analysis.get("eeg_policy"), previous_eeg)
    p["policy"].update(eeg_state=eeg.get("state", "unknown"), eeg_previous_state=previous_eeg, eeg=eeg)
    if focus:
        total = sum(c["attention"] for c in e["characters"].values())
        # Preserve unseen character weights; redistribute only the observed mass.
        mass = sum(p["characters"].get(n, 0) for n in e["characters"])
        for name, c in e["characters"].items():
            before = p["characters"].get(name, 0.0)
            after = round((1 - LEARN) * before + LEARN * mass * c["attention"] / total, 3)
            p["characters"][name] = after
            if abs(after - before) >= 0.03:
                changes.append(dict(key=f"character:{name}", before=before, after=after,
                    strong=False, why=f"Observed {c['dwell_s']:.1f}s on {name} during {c['visible_s']:.1f}s of shared visibility; tentative attention preference, not enjoyment"))
    # Readability policy, not a diagnosis of boredom. Invalid/stale input does
    # not contribute to valid_gaze_s or the valid outside-video fraction.
    # The opening window admits 3s of valid data. Retain 1.4s of actual outside
    # gaze for pacing, and 1.5s in EACH dialogue/silent condition; quality and
    # comparison margins are unchanged. Shorter evidence remains tentative.
    if readability_evidence(e):
        if outside_evidence(e, analysis):
            before = p["pacing"]
            p["pacing"] = round(clamp(before - .25), 3)
            changes.append(dict(key="pacing", before=before, after=p["pacing"], strong=False,
                                why="Frequent valid gaze outside the video; try a slower, more readable action"))
        dialogue = [b for b in analysis.get("beats", []) if b.get("dialogue")]
        silent = [b for b in analysis.get("beats", []) if not b.get("dialogue")]
        valid = lambda bs: sum(number(b.get("valid_s")) for b in bs)
        on_video = lambda bs: sum(number(b.get("on_video_s")) for b in bs)
        if min(valid(dialogue), valid(silent)) >= 1.5:
            delta = on_video(dialogue) / valid(dialogue) - on_video(silent) / valid(silent)
            if abs(delta) >= .3:
                before = p["dialogue"]
                p["dialogue"] = round(clamp(before + .3 * (1 if delta > 0 else -1)), 3)
                changes.append(dict(key="dialogue", before=before, after=p["dialogue"], strong=False,
                                    why="Different valid on-video attention during dialogue and silent windows; tentative delivery adjustment"))
    p["clips"] += 1
    return p, changes


def decide(profile, analysis):
    """Only fresh, adequate observations can change the next scene."""
    e = evidence(analysis, profile["characters"])
    policy = profile.get("policy", {})
    focus = focus_from(e, policy.get("previous_focus"))
    decision = dict(focus=focus, tension="same", dialogue="same", pacing="same", tone=None,
                    event=False, reasons=[], evidence=e, policy="local-gaze-v3-opening-window")
    if focus:
        c = e["characters"][focus]
        decision["reasons"].append(f"Observed attention: {c['dwell_s']:.1f}s on {focus} of {c['visible_s']:.1f}s shared visibility; tentative preference")
    if readability_evidence(e):
        if outside_evidence(e, analysis) and profile["pacing"] <= -.25:
            decision["pacing"] = "slower"
            decision["reasons"].append("Try slower readable motion after valid outside-video gaze")
        beats = analysis.get("beats", [])
        dialogue_s = sum(number(b.get("valid_s")) for b in beats if b.get("dialogue"))
        silent_s = sum(number(b.get("valid_s")) for b in beats if not b.get("dialogue"))
        if min(dialogue_s, silent_s) >= 1.5 and abs(profile["dialogue"]) >= .2:
            decision["dialogue"] = "more" if profile["dialogue"] > 0 else "less"
            decision["reasons"].append("Try adjusted dialogue delivery after compared valid gaze windows")
    eeg = eeg_policy.decide(analysis.get("eeg_policy"), policy.get("eeg_previous_state"))
    eeg["applied"] = bool(eeg_policy.delivery_cue(eeg, decision["pacing"]))
    if eeg.get("action") != "keep" and not eeg["applied"]:
        eeg["suppressed_by"] = "gaze_readability_pacing"
    decision["eeg_policy"] = eeg
    decision["evidence"] = dict(e, eeg=eeg)
    if eeg["applied"]:
        decision["reasons"].append(eeg["reason"])
        decision["policy"] = "local-gaze-v3+" + eeg["policy"]
    if not decision["reasons"]:
        decision["reasons"].append("Insufficient comparative gaze evidence; continue the story evenly")
    return decision
