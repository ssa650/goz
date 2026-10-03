"""Import clip settings and filename bindings without opening local file paths."""
import json
import unicodedata
from pydantic import ValidationError
from .clip_settings import ClipGenerationSettings
from .config import MAX_CLIPS, MAX_PROMPT_BYTES
from .prompts import parse_prompts

FRAME_FIELDS = (("firstFrame", "initialFrameFile", "Initial"), ("endFrame", "endFrameFile", "End"))


def filename_key(name):
    if not isinstance(name, str) or not name.strip() or len(name) > 255 or any(c in name for c in ("/", "\\", "\x00", "\n", "\r")) or name in (".", ".."):
        raise ValueError("Frame files must be plain filenames, for example 00-15.jpg.")
    return unicodedata.normalize("NFC", name).casefold()


def parse_clip_import(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_PROMPT_BYTES:
        raise ValueError("Prompt file must be at most 100 KB.")
    text = text.lstrip("\ufeff")
    if not text.lstrip().startswith(("{", "[")):
        return [{"settings":{"prompt":p}} for p in parse_prompts(text)], False
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError('Invalid JSON. Use prompt strings or {"clips": [...]} with complete clip objects.') from None
    items = value.get("clips") if isinstance(value, dict) else value
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_CLIPS:
        raise ValueError(f"Supply 1–{MAX_CLIPS} clips in a JSON array or a clips object.")
    if all(isinstance(item, str) for item in items):
        return [{"settings":{"prompt":p}} for p in parse_prompts(json.dumps(items))], False
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("Use either an array of prompt strings or an array of complete clip objects.")
    explicit_order = any("order" in item for item in items)
    orders, source_ids, records = set(), set(), []
    for position, item in enumerate(items):
        label = f"Clip {position+1}"
        try:
            order = item.get("order", position)
            if explicit_order and "order" not in item or type(order) is not int or order < 0 or order >= len(items) or order in orders:
                raise ValueError("Order must appear on every clip and be unique/consecutive, starting at zero.")
            orders.add(order)
            source_id = item.get("id")
            if source_id is not None:
                if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 200 or source_id in source_ids:
                    raise ValueError("Each imported clip ID must be a unique nonempty string, up to 200 characters.")
                source_ids.add(source_id)
            metadata = dict(sourceId=source_id)
            for _, field, _ in FRAME_FIELDS:
                name = item.get(field)
                if name is not None:
                    filename_key(name)
                metadata[field] = name
            if metadata["endFrameFile"] and not (metadata["initialFrameFile"] or item.get("firstFrame")):
                raise ValueError("An end frame file requires an initial frame file.")
            settings = ClipGenerationSettings.model_validate({k:v for k,v in item.items() if k not in {"id", "order", "initialFrameFile", "endFrameFile"}})
            if not settings.prompt.strip():
                raise ValueError("Prompt is empty.")
            records.append(dict(order=order, settings=settings.model_dump(mode="json"), **metadata))
        except (ValueError, TypeError) as error:
            detail = "; ".join(e["msg"] for e in error.errors()) if isinstance(error, ValidationError) else str(error)
            raise ValueError(f"{label}: {detail}") from None
    return sorted(records, key=lambda record:record["order"]), True
