"""Opt-in local experiment; importing this package does not load YOLOE or weights."""
from .config import DetectorConfig
from .detector import YOLOECharacterDetector, build_reference_prompt
from .worker import detect_yoloe

__all__ = ["DetectorConfig", "YOLOECharacterDetector", "build_reference_prompt", "detect_yoloe"]
