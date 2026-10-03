"""Prompt parsing and timed scene splitting, ported from GOZ_TEST/public."""
import json
import math
import re
from .config import MAX_CLIPS, MAX_PROMPT_BYTES, DURATIONS

DURATION_RE = re.compile(r"\[?[ \t]*DURATION:[ \t]*(\d+)[ \t]*s(?:ec(?:ond)?s?)?[ \t]*\.?[ \t]*\]?", re.I)
BLOCK_RE = re.compile(r"^\s*\[?(\d{1,2}):(\d{2})\s*[–—-]\s*(\d{1,2}):(\d{2})\]?(.*)$")
FOOTER_RE = re.compile(r"^\s*(SOUND|AUDIO|MUSIC|END FRAME|CONSTRAINTS|NOTES)\b[^:\n]*:")
FRAME_NOTE = "The video must begin exactly on the provided first frame and end exactly on the provided last frame."
PART_DIALOGUE = "Only the dialogue written in this part's timed sections is spoken in this clip."


def parse_prompts(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("Prompt batch must be at most 100 KB.")
    if text.lstrip().startswith("["):
        try:
            prompts = json.loads(text)
        except ValueError:
            raise ValueError("Invalid JSON: use an array of prompt strings.") from None
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    validate_prompts(prompts)
    return [p.strip() for p in prompts]


def validate_prompts(prompts):
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_CLIPS or any(
        not isinstance(p, str) or not p.strip() or len(p) > 8000 for p in prompts
    ):
        raise ValueError("Supply 1–12 prompts, each 1–8,000 characters.")


def scene_seconds(prompt, fallback):
    match = DURATION_RE.search(prompt)
    return int(match[1]) if match else fallback


def time_label(seconds):
    return f"{int(seconds // 60)}:{math.floor(seconds % 60 + .5):02d}"


def split_scene(prompt, parts, clip_seconds):
    if parts <= 1:
        return [prompt]
    total = scene_seconds(prompt, parts * clip_seconds)
    span, scale = total / parts, clip_seconds / (total / parts)

    def with_label(text, k):
        label = f"DURATION: {clip_seconds} seconds. This is part {k + 1} of {parts} of a {total}-second scene."
        return DURATION_RE.sub(lambda _: label, text, count=1) if DURATION_RE.search(text) else f"{label}\n\n{text}"

    lines = prompt.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        m = BLOCK_RE.match(line)
        if m:
            blocks.append(dict(line=i, start=int(m[1])*60+int(m[2]), end=int(m[3])*60+int(m[4]), title=m[5]))
    if not blocks:
        result = []
        for k in range(parts):
            portion = "beginning" if k == 0 else "ending" if k == parts - 1 else "middle"
            result.append(with_label(f"{FRAME_NOTE}\nShow only seconds {time_label(k*span)}–{time_label((k+1)*span)} of the full scene described below: the {portion} portion of the action. Do not perform dialogue or action that belongs to other parts.\n\n{prompt}", k))
        return result
    footer_start = next((i for i in range(blocks[-1]["line"]+1, len(lines)) if FOOTER_RE.match(lines[i])), len(lines))
    for i, block in enumerate(blocks):
        stop = blocks[i+1]["line"] if i+1 < len(blocks) else footer_start
        block["body"] = "\n".join(lines[block["line"]+1:stop]).strip()
        block["part"] = max(range(parts), key=lambda k: min(block["end"], (k+1)*span)-max(block["start"], k*span))
    header = "\n".join(lines[:blocks[0]["line"]]).strip()
    footer = []
    for line in lines[footer_start:]:
        if FOOTER_RE.match(line) or not footer:
            footer.append(dict(endFrame=bool(re.match(r"^\s*END FRAME\b", line)), lines=[line]))
        else:
            footer[-1]["lines"].append(line)
    result = []
    for k in range(parts):
        def clamp(value):
            return min(clip_seconds, max(0, (value-k*span)*scale))
        timed = [f"{time_label(clamp(b['start']))}–{time_label(clamp(b['end']))}{b['title']}\n{b['body']}".strip() for b in blocks if b["part"] == k]
        if not timed:
            timed = [f"0:00–{time_label(clip_seconds)}:\nContinue the motion naturally from the first frame toward the last frame. No dialogue."]
        tail = ["\n".join(s["lines"]).strip() for s in footer if not s["endFrame"] or k == parts-1]
        if k < parts-1:
            tail.append("END FRAME:\nMatch the provided last frame exactly.")
        result.append("\n\n".join(filter(None, [with_label(header,k), FRAME_NOTE, *timed, *tail, PART_DIALOGUE])))
    return result


def plan(text, mode, duration):
    if mode not in ("keyframes", "chain"):
        raise ValueError("Choose keyframes or chain mode.")
    if type(duration) is not int or duration not in DURATIONS:
        raise ValueError("Choose a whole-number duration from 5 to 15 seconds.")
    scenes = parse_prompts(text)
    clips = []
    for scene in scenes:
        parts = max(1, math.floor(scene_seconds(scene, duration)/duration + .5)) if mode == "keyframes" else 1
        if len(clips) + parts > MAX_CLIPS:
            raise ValueError(f"Scenes split into more than {MAX_CLIPS} clips; increase clip duration or shorten the batch.")
        clips.extend(split_scene(scene, parts, duration))
    validate_prompts(clips)
    return dict(scenes=scenes, clips=clips, frameCount=len(clips)+1 if mode == "keyframes" else 1,
                totalDuration=len(clips)*duration)
