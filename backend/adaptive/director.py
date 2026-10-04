"""Local, bounded adaptation and deterministic provider prompt composition."""
import json
import re
import time
from pathlib import Path

from . import eeg_policy

ROOT = Path(__file__).resolve().parent.parent.parent
GENRES = {"suspense", "action", "humor", "romance", "drama"}
ACTIONS = ("keep", "focus_character", "faster_pacing", "slower_pacing", "subtle_suspense", "subtle_humor", "clearer_dialogue", "less_dialogue", "more_dialogue")


def continuation(story):
    """Small local story controller for custom clips; saved bundles bypass it."""
    number = len(story["scenes"])
    events = (
        "Establish the premise with a clear shared goal and one intriguing detail in the existing environment.",
        "Continue the previous action. One character notices that detail and shows it to the others; they react differently.",
        "The characters investigate the same detail together. Their first attempt has a small comic setback.",
        "Resolve the setback with a playful visual reveal, then finish with the characters reacting together.",
    )
    cast = "; ".join(f"{c['name']}: {c.get('description') or 'retain the reference appearance'}" for c in story["characters"])
    props = "; ".join(f"{o['name']}: {o.get('description', '')}" for o in story.get("objects", []))
    previous = story["scenes"][-1]["summary"][-600:] if number else "Opening scene."
    event = events[min(number, len(events) - 1)]
    story["current_event"] = event
    return (f"IMMUTABLE WORLD: {story['premise']}\nCHARACTERS: {cast}\nRECURRING OBJECTS: {props}\n"
            "Keep reference character designs, clothing, visual style, location and relationships consistent.\n"
            f"STORY STATE: Scene {number + 1}. Previous scene: {previous}\nNEXT EVENT: {event}\n"
            "CAMERA: Continue from the supplied final frame with readable staging and natural motion.\n"
            "DIALOGUE: Use brief natural reactions appropriate to the current event; do not repeat the opening.\n"
            "AUDIO: Consistent character voices, gentle underwater ambience and synchronized action sounds.")


def decision_schema(names):
    return dict(type="object", properties={
        "action": dict(type="string", enum=list(ACTIONS)),
        "focus": dict(type=["string", "null"], enum=[None, *names]),
        "reason": dict(type="string", maxLength=300),
    }, required=["action", "focus", "reason"], additionalProperties=False)


def validate_decision(value, names):
    if (not isinstance(value, dict) or set(value) != {"action", "focus", "reason"}
            or value["action"] not in ACTIONS or value["focus"] not in [None, *names]
            or not isinstance(value["reason"], str) or len(value["reason"]) > 300):
        raise ValueError("Invalid bounded adaptation decision.")
    if (value["action"] == "focus_character") != (value["focus"] is not None):
        raise ValueError("Character focus does not match its action.")
    # Keep the existing dashboard contract; these fields describe one decision.
    return dict(**value, tension="higher" if value["action"] == "subtle_suspense" else "same",
                pacing={"faster_pacing": "faster", "slower_pacing": "slower"}.get(value["action"], "same"),
                dialogue="more" if value["action"] == "clearer_dialogue" else "same",
                tone="humor" if value["action"] == "subtle_humor" else None,
                event=False, reasons=[value["reason"]])


def local_decision(hint, names):
    """One local change, prioritized by valid observed character attention."""
    action, focus = "keep", None
    if hint.get("focus") in names:
        action, focus = "focus_character", hint["focus"]
    elif hint.get("dialogue") in ("more", "less"):
        action = "more_dialogue" if hint["dialogue"] == "more" else "less_dialogue"
    elif hint.get("pacing") in ("faster", "slower"):
        action = "faster_pacing" if hint["pacing"] == "faster" else "slower_pacing"
    eeg = dict(hint.get("eeg_policy") or {})
    if hint.get("pacing") in ("faster", "slower"):
        eeg["suppressed_by"] = "gaze_readability_pacing"
    eeg_primary = action == "keep" and bool(eeg_policy.delivery_cue(eeg))
    if eeg_primary:
        action = eeg["action"]
    choice = validate_decision(dict(action=action, focus=focus, reason="; ".join(hint.get("reasons", []))[:300]), names)
    choice.update(evidence=hint.get("evidence", {}), policy=hint.get("policy", "local-gaze-v2"))
    choice["eeg_primary"] = eeg_primary
    choice["eeg_policy"] = eeg
    choice["eeg_applied"] = bool(eeg_policy.delivery_cue(eeg, "same" if eeg_primary else choice["pacing"]))
    eeg["applied"] = choice["eeg_applied"]
    choice["applied_actions"] = ([action] if action != "keep" else []) + (
        [eeg["action"]] if choice["eeg_applied"] and not eeg_primary else [])
    choice["dialogue"] = {"more_dialogue": "more", "less_dialogue": "less"}.get(action, choice["dialogue"])
    return choice


def adjustment_cue(decision):
    primary = {
        "keep": "",
        "focus_character": (
            f"PRIMARY SHOT: {decision['focus']} receives the main medium close-up for most of the interior of this clip. "
            f"Keep {decision['focus']}'s face and existing action prominently visible; other characters remain supporting in wider framing. "
            "Use one simple continuous camera move from the opening to this shot and back to the required ending. "
            "This shot priority overrides competing camera/framing directions in the script, but does not reassign scripted actions or lines. "
            "Preserve the story outcome, exact character designs, voices, location and continuity. "
            "Reference first/last compositions govern only the boundary frames; the primary shot governs the interior."),
        "faster_pacing": "Slightly quicken the existing gestures and camera movement; preserve scripted actions, dialogue and outcome.",
        "slower_pacing": "Use slower readable gestures and a steady camera; preserve scripted actions, dialogue and outcome.",
        "subtle_suspense": "Add a brief anticipatory pause to the existing action without introducing events or changing dialogue.",
        "subtle_humor": "Emphasize the humor of existing facial reactions without adding events or changing dialogue.",
        "clearer_dialogue": "Make the scripted spoken lines clear with restrained background sound; retain their wording.",
        "less_dialogue": "Use only the shortest existing line needed to preserve the story outcome; show other reactions silently. Keep voices and characterization consistent.",
        "more_dialogue": "Give the existing dialogue a clear foreground delivery and room to finish, with less silent padding. Preserve wording, voices and story outcome; do not add exposition.",
    }[decision["action"]]
    eeg = eeg_policy.delivery_cue(decision.get("eeg_policy"),
                                  "same" if decision.get("eeg_primary") else decision.get("pacing", "same"))
    return eeg if decision.get("eeg_primary") else " ".join(cue for cue in (primary, eeg) if cue)


def adjust_prompt(base, decision):
    """Preserve the script and add an explicit, reviewable camera priority."""
    cue = adjustment_cue(decision)
    if not cue:
        return base
    result = base + "\n\nADAPTIVE SCENE DIRECTION (interior shot/delivery priority): " + cue
    if len(result) > 8000:
        raise ValueError("The original scene plus its adaptation exceeds 8,000 characters; shorten the source scene. No adaptation was silently dropped.")
    return result


def system_prompt(duration):
    return (ROOT/"prompts"/"director.md").read_text().replace("{duration}", str(duration))


def validate(plan, names, duration):
    if not isinstance(plan, dict) or not isinstance(plan.get("video_prompt"), str) or not plan["video_prompt"].strip():
        raise ValueError("Scene plan has no video prompt.")
    beats = []
    for b in plan.get("beats") or []:
        try:
            t0, t1 = max(0.0, float(b["t0"])), min(float(duration), float(b["t1"]))
        except (KeyError, TypeError, ValueError):
            continue
        if t1 > t0:
            speaker = b.get("speaker") if b.get("speaker") in names else None
            beats.append(dict(t0=t0, t1=t1, description=str(b.get("description", ""))[:300],
                              characters=[c for c in b.get("characters", []) if c in names],
                              dialogue=bool(b.get("dialogue")) or speaker is not None, speaker=speaker,
                              tags=[t for t in b.get("tags", []) if t in GENRES]))
    if not beats:
        beats = [dict(t0=0.0, t1=float(duration), description=plan.get("summary", ""), characters=list(names),
                      dialogue=False, tags=[])]
    return dict(scene_title=str(plan.get("scene_title", "Next scene"))[:120],
                summary=str(plan.get("summary", ""))[:600], beats=beats,
                video_prompt=plan["video_prompt"],
                change_note=str(plan.get("change_note", ""))[:300])


def parse_timeline(text, names, duration):
    """Beats for a predefined clip, one per line:
         0-4 SpongeBob: I'm ready!     (speaker line)
         4-7 Squidward slams the door #humor
    Lines naming no known speaker become silent beats; #tags set genres."""
    import re
    beats = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+(.*)", line)
        if not m:
            continue
        t0, t1, rest = float(m[1]), min(float(m[2]), float(duration)), m[3]
        tags = [t for t in re.findall(r"#(\w+)", rest) if t in GENRES]
        rest = re.sub(r"#\w+", "", rest).strip()
        head, sep, said = rest.partition(":")
        speaker = head.strip() if sep and head.strip() in names else None
        if t1 > t0:
            beats.append(dict(t0=t0, t1=t1, description=rest[:300], speaker=speaker, dialogue=speaker is not None,
                              characters=[n for n in names if n.lower() in rest.lower()], tags=tags))
    return beats


def user_message(story, profile, decision, duration):
    return json.dumps(dict(
        premise=story["premise"],
        characters=story["characters"],
        story_so_far=[s["summary"] for s in story["scenes"]][-6:],
        scene_number=len(story["scenes"]) + 1,
        duration_s=duration,
        viewer_profile=profile,
        next_prompt=story.get("next_prompt", story["premise"]),
        viewer_analysis=story.get("viewer_analysis"),
        viewer_data_available=bool(story["scenes"])), indent=1)


async def write_scene(story, profile, decision, duration, transport=None):
    """Bounded choices and prompt composition require no remote model call.

    The optional transport argument is retained for callers, but no viewer data
    or OpenAI request is sent from this path. Creative plot remains the supplied
    episode or the existing local continuation controller.
    """
    started = time.perf_counter()
    names = [c["name"] for c in story["characters"]]
    plan = template(story, decision, duration, names)
    plan["decisionElapsedMs"] = round((time.perf_counter() - started) * 1000, 3)
    return plan, "local-rules"


def template(story, decision, duration, names):
    started = time.perf_counter()
    decision = dict(decision) if "action" in decision else local_decision(decision, names)
    base = story.get("next_prompt", story["premise"])
    if not isinstance(base, str) or not base.strip() or len(base) > 8000:
        raise ValueError("The next scene needs its original prompt (1–8,000 characters).")
    focus = decision.get("focus")
    if focus and (focus not in names or not re.search(r"(?<!\w)" + re.escape(focus) + r"(?!\w)", base, re.I)):
        decision = local_decision(dict(reasons=["Observed target is absent from this scripted continuation; preserve its cast and story"],
                                       eeg_policy=decision.get("eeg_policy")), names)
        focus = None
    # Boundary references remain authoritative; focus changes the interior shot,
    # not the saved opening/ending asset, cast, or episode outcome.
    spec = dict(primary_character=focus, shot="main medium close-up" if focus else "scripted",
                central_action="retain scripted action and actor", dialogue=decision.get("dialogue", "same"),
                pacing=decision.get("pacing", "same"), reference_policy="first/last frame boundaries; adaptive interior shot",
                frame_constraints=story.get("frame_constraints", {}), story_outcome="preserved",
                eeg_delivery_action=decision.get("eeg_policy", {}).get("action", "keep") if decision.get("eeg_applied") else "keep")
    prompt = adjust_prompt(base, decision)
    return dict(
        scene_title=f"{focus} in the main shot" if focus else "The story continues",
        summary=story.get("next_summary", base[:600]),
        beats=parse_timeline(story.get("next_timeline", ""), names, duration) or
              [dict(t0=0.0, t1=float(duration), description=base[:300], characters=list(names),
                    dialogue=False, speaker=None, tags=[])],
        base_prompt=base, video_prompt=prompt, decision=decision, scene_spec=spec,
        prompt_changes=adjustment_cue(decision),
        promptConstructionMs=round((time.perf_counter() - started) * 1000, 3),
        change_note=("; ".join(decision.get("reasons", [])))[:300])
