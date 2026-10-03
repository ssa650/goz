"""LLM scene writer: story state + viewer profile + decision -> scene plan.

Uses the OpenAI Chat Completions API over httpx (OPENAI_API_KEY,
OPENAI_MODEL). Without a key a deterministic template writes the plan and
is labelled "template" everywhere, so rehearsals never pretend to be AI.
"""
import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
GENRES = {"suspense", "action", "humor", "romance", "drama"}


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
                video_prompt=plan["video_prompt"].strip()[:1500],
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
        adaptation_decision=decision), indent=1)


async def write_scene(story, profile, decision, duration, transport=None):
    names = [c["name"] for c in story["characters"]]
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return template(story, decision, duration, names), "template"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    async with httpx.AsyncClient(timeout=60, transport=transport) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=dict(model=model, response_format={"type": "json_object"}, temperature=0.8,
                      messages=[{"role": "system", "content": system_prompt(duration)},
                                {"role": "user", "content": user_message(story, profile, decision, duration)}]))
    if not response.is_success:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = ""
        raise ValueError(f"OpenAI request failed ({response.status_code}). {detail}".replace(key, "[redacted]")[:600])
    content = response.json()["choices"][0]["message"]["content"]
    return validate(json.loads(content), names, duration), f"openai:{model}"


def template(story, decision, duration, names):
    focus = decision.get("focus")
    others = [n for n in names if n != focus]
    look = {c["name"]: c.get("description", "") for c in story["characters"]}
    if focus:
        action = (f"The camera slowly pushes in on {focus} ({look[focus]}), who steps into the center of the frame "
                  f"and takes charge of the moment while {', '.join(others) or 'the others'} fall back into soft focus.")
    else:
        action = f"The camera holds a balanced two-shot of {' and '.join(names)} as the moment unfolds."
    mood = {"higher": " Tension rises: low rumbling score, sharper light, quicker movements."}.get(decision.get("tension"), "")
    if decision.get("event"):
        mood += " Suddenly something unexpected crashes into the scene and everyone reacts."
    if decision.get("tone"):
        mood += f" The mood leans into {decision['tone']}."
    pace = " Fast, energetic movement." if decision.get("pacing") == "faster" else ""
    talk = " No dialogue, only ambience." if decision.get("dialogue") == "less" else ""
    half = duration / 2
    return dict(
        scene_title=f"{focus} takes the lead" if focus else "The story continues",
        summary=f"{focus or 'The group'} takes the lead as the story continues." ,
        beats=[dict(t0=0.0, t1=half, description="Setup", characters=list(names), dialogue=False, speaker=None,
                    tags=["drama"]),
               dict(t0=half, t1=float(duration), description=f"{focus or 'Everyone'} in focus",
                    characters=[focus] if focus else list(names), dialogue=decision.get("dialogue") != "less",
                    speaker=focus if decision.get("dialogue") != "less" else None,
                    tags=[decision.get("tone") or ("suspense" if decision.get("tension") == "higher" else "drama")])],
        video_prompt=f"{story['premise']} {action}{mood}{pace}{talk}"[:1500],
        change_note=("; ".join(decision.get("reasons", [])))[:300])
