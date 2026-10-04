"""Independent editable clips backed by the existing sequence/job histories."""
import builtins
from copy import deepcopy
from typing import TYPE_CHECKING
from uuid import uuid4, UUID
from .clip_settings import ClipGenerationSettings, generation_options
from .config import MAX_CLIPS
from .fal_adapter import FalError
from .clip_import import FRAME_FIELDS, filename_key, parse_clip_import

if TYPE_CHECKING:
    from .engine import Engine

WORKSPACE_ID = "individual-clips"
ACTIVE_STATUSES = {"uploading", "submitting", "queued", "generating"}


class ClipService:
    def __init__(self, engine: "Engine"):
        self.engine = engine

    def library(self) -> dict:
        e = self.engine
        if WORKSPACE_ID not in e.sequences:
            # Legacy strings are migrated once. Old runs remain available unchanged.
            recent = next(iter(reversed(list(e.sequences.values()))), None)
            definitions = []
            if recent:
                completed = {c.get("order", c.get("index", index)): c for index, c in enumerate(recent.get("clips", []))}
                source = recent.get("clipDefinitions") or (recent.get("clips") if recent.get("mode") == "bundles" else None)
                source = source or [{"prompt":p} for p in recent.get("prompts", [])]
                for index, definition in enumerate(source[:MAX_CLIPS]):
                    prompt = definition["prompt"]
                    result = completed.get(index, {})
                    job = e.jobs.get(definition.get("jobId") or result.get("jobId"), {})
                    settings = {**{k:v for k,v in definition.items() if k in ClipGenerationSettings.model_fields},
                                "prompt":prompt,"duration":definition.get("duration",recent.get("duration",15)),
                                "resolution":definition.get("resolution",recent.get("resolution","480P"))}
                    if type(job.get("seed")) is int:
                        settings["seed"] = job["seed"]
                    settings["promptExpansionMode"] = job.get("promptExpansionMode", "disabled")
                    for field in ("firstFrame", "endFrame"):
                        if job.get(field) and e.reference_exists(job[field]):
                            settings[field] = job[field]
                    record = self.new_record(settings)
                    if job:
                        record.update(jobId=job["id"], jobIds=[job["id"]], status=job["status"], legacyMetadata=not bool(job.get("generationInput")))
                    definitions.append(record)
            e.sequences[WORKSPACE_ID] = dict(id=WORKSPACE_ID, mode="individual", status="completed",
                                            clipDefinitions=definitions)
            e.persist()
        library = e.sequences[WORKSPACE_ID]
        # Normalize older per-clip records once, never generate a seed while rendering.
        changed = False
        for order, record in enumerate(library["clipDefinitions"]):
            if record.get("order") != order:
                record["order"] = order
                changed = True
            normalized = self.settings(record)
            for key, value in normalized.model_dump(mode="json").items():
                if key not in record or key in ("seed", "promptExpansionMode") and record[key] is None:
                    record[key] = value
                    changed = True
        if changed:
            e.persist()
        return library

    def settings(self, record: dict) -> ClipGenerationSettings:
        fields = ClipGenerationSettings.model_fields
        return ClipGenerationSettings.model_validate({k: v for k, v in record.items() if k in fields and not (k in ("seed", "promptExpansionMode") and v is None)})

    def new_record(self, body: dict) -> dict:
        settings = ClipGenerationSettings.model_validate(body)
        self.validate_references(settings)
        return dict(id=str(uuid4()), **settings.model_dump(mode="json"), status="ready", jobId=None, jobIds=[])

    def validate_references(self, settings: ClipGenerationSettings):
        for reference_id in (settings.firstFrame, settings.endFrame):
            if reference_id:
                self.engine.reference(str(reference_id))

    def validate_required_frames(self, record, settings):
        for field, filename, label in FRAME_FIELDS:
            if record.get(filename) and not getattr(settings, field):
                raise ValueError(f"{label} frame {record[filename]} is missing. Upload the matching frame files.")

    def require_idle_cards(self):
        if any(self.engine.jobs.get(c.get("jobId"), {}).get("status") in ACTIVE_STATUSES for c in self.library()["clipDefinitions"]):
            raise FalError("Wait for active clip requests before importing or attaching frames.", 409)

    def get(self, clip_id: str) -> dict:
        record = next((c for c in self.library()["clipDefinitions"] if c["id"] == clip_id), None)
        if not record:
            raise FalError("Clip not found.", 404)
        return record

    def snapshot(self, record: dict) -> dict:
        result = deepcopy(record)
        for field in ("firstFrame", "endFrame"):
            result[field] = self.engine.reference(record[field]) if record.get(field) else None
        job = self.engine.jobs.get(record.get("jobId"))
        result["result"] = self.engine.public_job(job) if job else None
        result["status"] = job["status"] if job and job["status"] in ACTIVE_STATUSES else record["status"]
        return result

    def list(self) -> builtins.list[dict]:
        return [self.snapshot(record) for record in self.library()["clipDefinitions"]]

    def require_editable(self):
        if self.engine.bundles.active():
            raise FalError("Wait for the sequence and its accepted requests before editing clip bundles.", 409)

    def reorder(self, clip_ids: builtins.list[str]) -> builtins.list[dict]:
        self.require_editable()
        records = self.library()["clipDefinitions"]
        if not isinstance(clip_ids, list) or any(not isinstance(i, str) for i in clip_ids) or len(clip_ids) != len(records) or set(clip_ids) != {c["id"] for c in records}:
            raise ValueError("Supply every current clip ID exactly once in the desired order.")
        lookup = {c["id"]:c for c in records}
        records[:] = [lookup[i] for i in clip_ids]
        self.library()  # Normalize and persist explicit order alongside array position.
        self.engine.persist()
        return self.list()

    def add(self, body: dict) -> dict:
        self.require_editable()
        library = self.library()
        if len(library["clipDefinitions"]) >= MAX_CLIPS:
            raise ValueError(f"Keep at most {MAX_CLIPS} clip cards.")
        record = self.new_record(body)
        record["order"] = len(library["clipDefinitions"])
        library["clipDefinitions"].append(record)
        self.engine.persist()
        return self.snapshot(record)

    def edit(self, clip_id: str, body: dict) -> dict:
        self.require_editable()
        record = self.get(clip_id)
        job = self.engine.jobs.get(record.get("jobId"))
        if job and job["status"] in ACTIVE_STATUSES:
            raise FalError("Wait for this clip's generation before editing its settings.", 409)
        old = self.settings(record).model_dump(mode="json")
        settings = ClipGenerationSettings.model_validate({**old, **body})
        self.validate_references(settings)
        current = settings.model_dump(mode="json")
        record.update(current)
        for field, filename, _ in FRAME_FIELDS:
            if current[field] != old[field] and filename in record:
                record[filename] = self.engine.reference(current[field])["name"] if current[field] else None
        if current != old:
            record.update(status="ready", error=None)
        self.engine.persist()
        return self.snapshot(record)

    def duplicate(self, clip_id: str) -> dict:
        record = self.get(clip_id)
        duplicate = self.add(self.settings(record).model_dump(mode="json"))
        copy = self.get(duplicate["id"])
        for key in ("sourceId", "initialFrameFile", "endFrameFile"):
            if key in record:
                copy[key] = record[key]
        self.engine.persist()
        return self.snapshot(copy)

    def remove(self, clip_id: str):
        self.require_editable()
        record = self.get(clip_id)
        job = self.engine.jobs.get(record.get("jobId"))
        if job and job["status"] in ACTIVE_STATUSES:
            raise FalError("Cancel this clip and wait for its request before removing it.", 409)
        self.library()["clipDefinitions"].remove(record)
        self.library()
        self.engine.persist()

    def import_prompts(self, text: str, mode=None) -> builtins.list[dict]:
        self.require_editable()
        self.require_idle_cards()
        imported, structured = parse_clip_import(text)
        mode = mode or ("replace" if structured else "append")
        if mode not in ("replace", "append"):
            raise ValueError("Choose Replace sequence or Append clips.")
        library = self.library()
        existing = library["clipDefinitions"]
        # Replace only the untouched blank starter, preserving all configured cards.
        placeholder = bool(len(existing) == 1 and existing[0].get("starter") and not existing[0]["prompt"] and not existing[0].get("firstFrame") and not existing[0].get("jobId"))
        if mode == "append" and len(existing) + len(imported) - int(placeholder) > MAX_CLIPS:
            raise ValueError(f"Append would exceed {MAX_CLIPS} clips. Choose Replace sequence to load this file.")
        records = []
        for item in imported:
            record = self.new_record(item["settings"])
            record.update({k:v for k,v in item.items() if k not in ("settings", "order")})
            records.append(record)
        library["beforeImport"] = deepcopy(existing)
        if placeholder or mode == "replace":
            existing.clear()
        existing.extend(records)
        self.library()
        self.engine.persist()
        return self.list()

    def undo_import(self):
        self.require_editable()
        self.require_idle_cards()
        library = self.library()
        if "beforeImport" not in library:
            raise ValueError("No import to undo.")
        library["clipDefinitions"] = library.pop("beforeImport")
        self.library()
        self.engine.persist()
        return self.list()

    def frame_plan(self, names):
        """Validate filenames against the current bundles before storing images."""
        self.require_editable()
        self.require_idle_cards()
        keys = [filename_key(name) for name in names]
        if len(set(keys)) != len(keys):
            raise ValueError("Two selected frames have the same filename. Select only one file for each name.")
        records = self.library()["clipDefinitions"]
        expected = {filename_key(c[field]) for c in records for _,field,_ in FRAME_FIELDS if c.get(field)}
        if not expected:
            raise ValueError("This sequence has no filename bindings. Import a clips JSON with initialFrameFile/endFrameFile, or use each card's uploader.")
        unexpected = [name for name,key in zip(names,keys) if key not in expected]
        if unexpected:
            raise ValueError("Files not referenced by this sequence: " + ", ".join(unexpected))
        missing = {}
        for record in records:
            errors = [f"{label} frame {record[filename]} is missing." for field,filename,label in FRAME_FIELDS
                      if record.get(filename) and not record.get(field) and filename_key(record[filename]) not in keys]
            if errors:
                missing[record["id"]] = " ".join(errors)
        if missing:
            from .bundle_sequence import ClipValidationError
            raise ClipValidationError(missing)
        return {c["id"]: {field:filename_key(c[filename]) for field,filename,_ in FRAME_FIELDS if c.get(filename) and filename_key(c[filename]) in keys} for c in records}

    def attach_frames(self, plan, references):
        for clip_id, fields in plan.items():
            record = self.get(clip_id)
            patch = {field:references[name]["id"] for field,name in fields.items()}
            if patch:
                settings = ClipGenerationSettings.model_validate({**self.settings(record).model_dump(mode="json"), **patch})
                self.validate_references(settings)
                record.update(settings.model_dump(mode="json"), status="ready", error=None)
        self.engine.persist()
        return self.list()

    async def generate(self, clip_id: str, token: str) -> dict:
        UUID(token)
        e = self.engine
        async with e.lock:
            if e.bundles.active():
                raise FalError("Wait for the sequence to finish before generating another clip.", 409)
            record = self.get(clip_id)
            existing = next((j for j in e.jobs.values() if j.get("clipId") == clip_id and j.get("clientToken") == token), None)
            if existing:
                return e.public_job(existing)
            e.generation_guard()
            if not e.adapter.configured():
                raise FalError("Add your Fal API key first.", 503)
            active = e.jobs.get(record.get("jobId"))
            if active and active["status"] in ACTIVE_STATUSES:
                raise FalError("This clip is already generating.", 409)
            if any(j.get("requestUncertain") for j in e.jobs.values()):
                raise FalError("A Fal submission is uncertain. Check Fal history before submitting another generation.", 409)
            if any(s.get("mode") != "individual" and s["status"] not in {"completed", "failed", "cancelled", "interrupted"} for s in e.sequences.values()) or any(j["status"] in ACTIVE_STATUSES and not j.get("clipId") for j in e.jobs.values()):
                raise FalError("Wait for the legacy sequence to finish before generating individual clips.", 409)
            settings = self.settings(record)
            if not settings.prompt.strip():
                raise ValueError("Write a prompt between 1 and 8,000 characters.")
            self.validate_references(settings)
            self.validate_required_frames(record, settings)
            options = generation_options(settings)
            job = e.new_job(options, clipId=clip_id, clientToken=token,
                            settings=settings.model_dump(mode="json"), frameControl="keyframes" if settings.endFrame else "initial-image" if settings.firstFrame else "none")
            record.update(status="uploading", jobId=job["id"], error=None)
            record["jobIds"].append(job["id"])
            e.persist()
            e.spawn(e.run_job(job, {}))
            return e.public_job(job)

    def job_updated(self, job: dict):
        if not job.get("clipId"):
            return
        record = next((c for c in self.engine.sequences.get(WORKSPACE_ID, {}).get("clipDefinitions", []) if c["id"] == job["clipId"]), None)
        if record and record.get("jobId") == job["id"]:
            record.update(status=job["status"], error=job.get("error"))
