"""Validated per-clip settings and shared H3 Max Turbo routing."""
import secrets
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from .config import DURATION, DURATIONS, MODELS, build_input

PromptExpansionMode = Literal["disabled", "balanced", "quality"]
MAX_SAFE_SEED = 2**53 - 1


def random_seed() -> int:
    return secrets.randbelow(2**31)


class ReferenceFrame(BaseModel):
    id: UUID
    name: str
    previewUrl: str
    contentType: Literal["image/png", "image/jpeg", "image/webp"]
    size: int
    providerUrl: str | None = None


class ClipGenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="", max_length=8000)
    seed: StrictInt = Field(default_factory=random_seed, ge=-MAX_SAFE_SEED, le=MAX_SAFE_SEED)
    firstFrame: UUID | None = None
    endFrame: UUID | None = None
    promptExpansionMode: PromptExpansionMode = "disabled"
    duration: StrictInt = DURATION
    resolution: Literal["480P", "768P", "1080P"] = "480P"

    @model_validator(mode="after")
    def validate_frames_and_duration(self):
        if self.endFrame and not self.firstFrame:
            raise ValueError("Add an initial frame before adding an end frame.")
        if self.duration not in DURATIONS:
            raise ValueError("Choose a whole-number duration from 5 to 15 seconds.")
        return self


def generation_options(settings: ClipGenerationSettings) -> dict:
    return dict(mode="frames" if settings.firstFrame else "text", prompt=settings.prompt,
                seed=settings.seed, promptExpansionMode=settings.promptExpansionMode,
                duration=settings.duration, resolution=settings.resolution,
                firstFrame=str(settings.firstFrame) if settings.firstFrame else None,
                endFrame=str(settings.endFrame) if settings.endFrame else None)


def build_h3_request(options: dict, urls: dict) -> tuple[str, dict]:
    """One shared input builder; routing depends only on the initial frame."""
    if urls.get("end") and not urls.get("start"):
        raise ValueError("Add an initial frame before adding an end frame.")
    routed = {**options, "mode": "frames" if urls.get("start") else "text"}
    return MODELS[routed["mode"]], build_input(routed, urls)
