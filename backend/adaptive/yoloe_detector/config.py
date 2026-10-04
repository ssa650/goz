"""Local experiment settings, deliberately independent of shared app configuration."""
from dataclasses import dataclass
from pathlib import Path
import math

MODEL_NAME = "yoloe-11s-seg.pt"
OFFICIAL_WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt"
OFFICIAL_WEIGHT_SHA256 = "8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3"
# Resolve/lock these in a NEW venv only after the license decision. Never pip -U
# the app environment. Exact runtime versions remain unvalidated until executed.
RUNTIME_REQUIREMENTS = ("ultralytics==8.4.36", "torch==2.10.0", "torchvision==0.25.0")
# Existing human-labelled exemplars, copied as data from tracks.REFERENCE_CROPS
# to keep the separate inference runtime independent of app/cloud dependencies.
# Tests compare this snapshot against the current shared annotation catalogue.
REFERENCE_CROPS = {
    "SpongeBob": [("00-30.jpg", (.555, .51, .795, .93)),
                  ("01-45.jpg", (.175, .475, .48, .94)),
                  ("00-15.jpg", (.80, .54, .89, .75))],
    "Patrick": [("00-30.jpg", (.24, .18, .445, .80)),
                ("01-45.jpg", (.59, .15, .79, .84))],
}


@dataclass(frozen=True)
class DetectorConfig:
    weights: str = ""
    weights_sha256: str = ""
    license_reviewed: bool = False
    runtime_python: str = ""
    prompt_strategy: str = "scene"
    device: str = "auto"
    allow_cpu_fallback: bool = True
    image_size: int = 640
    fps: float = 2.0
    identity_threshold: float = .45
    candidate_threshold: float = .12
    ambiguity_score_gap: float = .15
    ambiguity_overlap: float = .6
    duplicate_score_ratio: float = .75
    max_detections: int = 12
    max_seconds: float = 120.0
    max_wall_seconds: float = 45.0
    queue_size: int = 4

    def __post_init__(self):
        if self.device not in ("auto", "mps", "cpu"):
            raise ValueError("Only local auto/mps/cpu devices are supported")
        if self.prompt_strategy not in ("scene", "atlas"):
            raise ValueError("Prompt strategy must be scene or atlas")
        if self.image_size not in (320, 480, 640):
            raise ValueError("Use a bounded image size: 320, 480 or 640")
        for key in ("fps", "max_seconds", "max_wall_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        if self.fps > 4 or self.max_seconds > 120 or self.max_wall_seconds > 120:
            raise ValueError("Experiment budget exceeded")
        for key in ("identity_threshold", "candidate_threshold", "ambiguity_score_gap",
                    "ambiguity_overlap", "duplicate_score_ratio"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid {key}")
        if self.candidate_threshold > self.identity_threshold:
            raise ValueError("Candidate threshold exceeds identity threshold")
        if not isinstance(self.queue_size, int) or not 1 <= self.queue_size <= 8:
            raise ValueError("Queue must hold 1..8 records")
        if not isinstance(self.max_detections, int) or not 1 <= self.max_detections <= 32:
            raise ValueError("Detection count must be 1..32")

    def validated_weights(self):
        """Verify bytes BEFORE the vendor's pickle-capable checkpoint loader."""
        import hashlib
        import re
        if self.license_reviewed is not True:
            raise RuntimeError("YOLOE license decision required before runtime loading")
        path = Path(self.weights).expanduser().resolve()
        if path.name != MODEL_NAME or not path.is_file():
            raise ValueError("Supply an existing official yoloe-11s-seg.pt; no implicit download")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.weights_sha256):
            raise ValueError("Require SHA256 verified against the official release asset")
        if self.weights_sha256.lower() != OFFICIAL_WEIGHT_SHA256:
            raise ValueError("SHA256 must match the pinned official v8.4.0 release asset")
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != self.weights_sha256.lower():
            raise ValueError("Checkpoint SHA256 mismatch")
        return path
