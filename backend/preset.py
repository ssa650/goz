"""The bundled Secret Box inputs; loading a preset never submits generation."""
import asyncio
import json
from pathlib import Path
from .clip_import import FRAME_FIELDS, parse_clip_import
from .config import MAX_PROMPT_BYTES
from .frames import verify_image

PRESET_ID = "secret-box"
PRESET_DIRECTORY = Path(__file__).resolve().parent.parent / "presets" / PRESET_ID


async def load_default_sequence(engine, directory=PRESET_DIRECTORY):
    engine.clips.require_editable()
    engine.clips.require_idle_cards()
    text = (directory / "prompts.json").read_text(encoding="utf-8")
    imported, structured = parse_clip_import(text)
    if not structured:
        raise ValueError("The built-in sequence must contain complete clip bundles.")
    filenames = list(dict.fromkeys(item[filename] for item in imported
                                 for _, filename, _ in FRAME_FIELDS if item.get(filename)))
    # Validate every image before replacing any editable cards. Only the bundled
    # manifest supplies filenames; HTTP requests cannot supply a filesystem path.
    def image(name):
        try:
            return verify_image((directory / "frames" / name).read_bytes(), name)
        except (OSError, ValueError) as error:
            raise ValueError(f"Built-in frame {name} could not load: {error}") from error
    images = await asyncio.to_thread(lambda: [image(name) for name in filenames])
    references = {img.name: engine.save_reference(img) for img in images}
    clips = []
    for item in imported:
        clip = dict(item["settings"], id=item["sourceId"], order=item["order"])
        for field, filename, _ in FRAME_FIELDS:
            clip[filename] = item.get(filename)
            if item.get(filename):
                clip[field] = references[item[filename]]["id"]
        clips.append(clip)
    # UUID references add a little JSON overhead to the bounded source file.
    payload = json.dumps({"clips": clips}, ensure_ascii=False)
    if len(payload.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("The built-in sequence exceeds the prompt import limit.")
    result = engine.clips.import_prompts(payload, mode="replace")
    engine.clips.library()["defaultPresetInitialized"] = PRESET_ID
    engine.persist()
    return result


async def ensure_default_sequence(engine, directory=PRESET_DIRECTORY):
    """Initialize once; preserve saved edits, deliberate deletions and results."""
    library = engine.clips.library()
    if library.get("defaultPresetInitialized"):
        return
    cards = library["clipDefinitions"]
    starter = (len(cards) == 1 and cards[0].get("starter") and not cards[0]["prompt"]
               and not cards[0].get("firstFrame") and not cards[0].get("endFrame")
               and not cards[0].get("jobId"))
    if not cards or starter:
        await load_default_sequence(engine, directory)
    else:
        library["defaultPresetInitialized"] = PRESET_ID
        engine.persist()
