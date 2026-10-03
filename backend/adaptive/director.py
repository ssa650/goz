"""Bounded OpenAI decisions; local, small adjustments to existing prompts."""
import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
GENRES = {"suspense", "action", "humor", "romance", "drama"}
ACTIONS = ("keep", "focus_character", "faster_pacing", "slower_pacing", "subtle_suspense", "subtle_humor", "clearer_dialogue")


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
        raise ValueError("OpenAI returned an invalid engagement decision.")
    if (value["action"] == "focus_character") != (value["focus"] is not None):
        raise ValueError("OpenAI's character focus does not match its action.")
    # Keep the existing dashboard contract; these fields describe one decision.
    return dict(**value, tension="higher" if value["action"] == "subtle_suspense" else "same",
                pacing={"faster_pacing": "faster", "slower_pacing": "slower"}.get(value["action"], "same"),
                dialogue="more" if value["action"] == "clearer_dialogue" else "same",
                tone="humor" if value["action"] == "subtle_humor" else None,
                event=False, reasons=[value["reason"]])


def local_decision(hint, names):
    """Clearly labelled no-key rehearsal fallback, constrained to one choice."""
    action, focus = "keep", None
    if hint.get("focus") in names:
        action, focus = "focus_character", hint["focus"]
    elif hint.get("pacing") == "faster":
        action = "faster_pacing"
    elif hint.get("tone") == "humor":
        action = "subtle_humor"
    elif hint.get("tension") == "higher":
        action = "subtle_suspense"
    elif hint.get("dialogue") == "more":
        action = "clearer_dialogue"
    return validate_decision(dict(action=action, focus=focus, reason="; ".join(hint.get("reasons", []))[:300]), names)


def adjust_prompt(base, decision):
    """Append at most one controlled cue. Never truncate or replace the scene."""
    cue = {
        "keep": "",
        "focus_character": f"Gently emphasize {decision['focus']}'s existing reaction with a subtle camera push; preserve all scripted actions and lines.",
        "faster_pacing": "Slightly quicken the existing gestures and camera movement; preserve every scripted action and spoken line.",
        "slower_pacing": "Slightly soften the pace of existing gestures and camera movement; preserve every scripted action and spoken line.",
        "subtle_suspense": "Add a subtle anticipatory pause and restrained ambience to the existing action, without introducing events or changing dialogue.",
        "subtle_humor": "Gently emphasize the humor of the existing facial reactions, without adding jokes, events, or changing dialogue.",
        "clearer_dialogue": "Make the existing spoken lines slightly clearer through natural delivery and restrained background sound; retain their exact wording.",
    }[decision["action"]]
    if not cue:
        return base
    addition = "\n\nSubtle engagement adjustment: " + cue
    # A full-length user prompt wins over the optional cue.
    return base + addition if len(base + addition) <= 8000 else base


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
    names = [c["name"] for c in story["characters"]]
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or not story["scenes"]:
        return template(story, decision, duration, names), "template"
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"
    async with httpx.AsyncClient(timeout=60, transport=transport) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {key}"},
            json=dict(model=model, store=False, max_output_tokens=1200, reasoning={"effort": "none"},
                      text={"format": dict(type="json_schema", name="engagement_decision", strict=True,
                                          schema=decision_schema(names))},
                      input=[{"role": "system", "content": system_prompt(duration)},
                                {"role": "user", "content": user_message(story, profile, decision, duration)}]))
    if not response.is_success:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = ""
        raise ValueError(f"OpenAI request failed ({response.status_code}). {detail}".replace(key, "[redacted]")[:600])
    payload = response.json()
    if payload.get("status") != "completed":
        raise ValueError("OpenAI decision did not complete. No new Fal generation was submitted.")
    contents = [c for output in payload.get("output", []) if output.get("type") == "message"
                for c in output.get("content", [])]
    if any(c.get("type") == "refusal" for c in contents):
        raise ValueError("OpenAI declined the engagement decision. No new Fal generation was submitted.")
    texts = [c.get("text", "") for c in contents if c.get("type") == "output_text"]
    if len(texts) != 1:
        raise ValueError("OpenAI returned no usable engagement decision.")
    choice = validate_decision(json.loads(texts[0]), names)
    plan = template(story, choice, duration, names)
    plan["decisionRequestId"] = payload.get("id")
    return plan, f"openai:{model}"


def template(story, decision, duration, names):
    # This function assembles a plan locally; the model never supplies Fal text.
    decision = decision if "action" in decision else local_decision(decision, names)
    base = story.get("next_prompt", story["premise"])
    if not isinstance(base, str) or not base.strip() or len(base) > 8000:
        raise ValueError("The next scene needs its original prompt (1–8,000 characters).")
    focus = decision.get("focus")
    return dict(
        scene_title=f"{focus} takes the lead" if focus else "The story continues",
        summary=base[:600],
        beats=parse_timeline(story.get("next_timeline", ""), names, duration) or
              [dict(t0=0.0, t1=float(duration), description=base[:300], characters=list(names),
                    dialogue=False, speaker=None, tags=[])],
        base_prompt=base, video_prompt=adjust_prompt(base, decision), decision=decision,
        change_note=("; ".join(decision.get("reasons", [])))[:300])
