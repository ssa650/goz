"""Reference-conditioned inference with conservative, uncalibrated abstention.

No temporal identity inference: every nonblank step obtains fresh detections.
The class map is built from labelled exemplars, never result captions/cast order.
"""
import math
import time

from .config import DetectorConfig


def build_reference_prompt(names, root=None, crops=None, strategy="scene"):
    """Five existing labelled crops -> one bounded RGB atlas and pixel boxes.

    An atlas permits all examples to share one reference encoding. It is only a
    prompting strategy, not training or ground truth for a generated target.
    """
    from pathlib import Path
    import numpy as np
    from PIL import Image
    if crops is None:
        from .config import REFERENCE_CROPS
        crops = REFERENCE_CROPS
    root = Path(root) if root else Path(__file__).resolve().parents[3] / "presets/secret-box/frames"
    requested = list(dict.fromkeys(names))
    supported = [name for name in requested if name in crops and crops[name]]
    if not supported:
        raise ValueError("No labelled visual references for requested identities")
    if strategy == "scene":
        # Preserve natural context/scale from the existing shared reference image.
        # Choose a scene containing one labelled exemplar for every supported ID;
        # this excludes generated targets and involves no runtime user boxes.
        files = set(filename for filename, _ in crops[supported[0]])
        for name in supported[1:]:
            files &= {filename for filename, _ in crops[name]}
        if not files:
            raise ValueError("No single labelled reference scene covers supported identities")
        filename = next(filename for filename, _ in crops[supported[0]] if filename in files)
        path = (root / filename).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Reference path escapes reference directory")
        with Image.open(path) as image:
            image = image.convert("RGB")
            # Keep pixel boxes aligned after bounded resize; model handles its
            # own letterbox. No crops/atlas distort the natural reference scene.
            original_w, original_h = image.size
            image.thumbnail((640,640))
            w,h = image.size
            reference = np.asarray(image)
        boxes,provenance = [],[]
        for class_id,name in enumerate(supported):
            box = next(box for file,box in crops[name] if file == filename)
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("Invalid labelled reference box")
            boxes.append([v*(w if i%2==0 else h) for i,v in enumerate(box)])
            provenance.append(dict(identity=name,class_id=class_id,file=filename,box=list(box)))
        return reference,dict(bboxes=np.asarray(boxes,dtype=np.float32),cls=np.arange(len(supported),dtype=np.int64)),dict(enumerate(supported)),provenance
    if strategy != "atlas":
        raise ValueError("Unknown reference prompting strategy")
    examples = [(i, name, filename, box) for i, name in enumerate(supported)
                for filename, box in crops[name]]
    if len(examples) > 8:
        raise ValueError("Reference atlas is limited to eight existing exemplars")
    side = 208
    columns = min(3, len(examples))
    rows = math.ceil(len(examples) / columns)
    atlas = Image.new("RGB", (columns * side, rows * side), (114, 114, 114))
    boxes, classes, provenance = [], [], []
    for index, (class_id, name, filename, box) in enumerate(examples):
        if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("Invalid labelled reference box")
        path = (root / filename).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Reference path escapes reference directory")
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            patch = image.crop(tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box)))
            patch.thumbnail((side - 16, side - 16))
        x = (index % columns) * side + (side - patch.width) // 2
        y = (index // columns) * side + (side - patch.height) // 2
        atlas.paste(patch, (x, y))
        boxes.append([x, y, x + patch.width, y + patch.height])
        classes.append(class_id)
        provenance.append(dict(identity=name, class_id=class_id, file=filename, box=list(box)))
    return np.asarray(atlas), dict(bboxes=np.asarray(boxes, dtype=np.float32),
                                  cls=np.asarray(classes, dtype=np.int64)), dict(enumerate(supported)), provenance


def overlap_smaller(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return intersection / smaller if smaller > 0 else 0


def parse_predictions(predictions, class_map, names, config):
    """Validate raw class IDs/geometry, then abstain on conflicting candidates.

    Score gap compares overlapping returned candidates; it is NOT a posterior
    class margin (Ultralytics Results does not expose the complete score vector).
    """
    regions = []
    for prediction in predictions[:config.max_detections]:
        try:
            box = [float(v) for v in prediction["box"]]
            score = float(prediction["score"])
            raw_id = float(prediction["class_id"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3] or not math.isfinite(score)
                or not config.candidate_threshold <= score <= 1):
            continue
        known_id = math.isfinite(raw_id) and raw_id.is_integer() and int(raw_id) in class_map
        identity = class_map[int(raw_id)] if known_id else None
        if identity not in names:
            identity = None
        status = "experimental_visual_prompt" if identity and score >= config.identity_threshold else "unknown_low_score" if identity else "unknown_class_id"
        regions.append(dict(box=box, identity=identity if status == "experimental_visual_prompt" else None,
                            candidate_identity=identity, identity_status=status,
                            verification=dict(method="yoloe_visual_reference", score_kind="uncalibrated_model_score",
                                              score=score, class_id=int(raw_id) if math.isfinite(raw_id) and raw_id.is_integer() else None)))
    # Cross-class alternatives with substantial shared geometry and similar scores.
    for i, a in enumerate(regions):
        for b in regions[i+1:]:
            if (a["candidate_identity"] and b["candidate_identity"] and a["candidate_identity"] != b["candidate_identity"]
                    and overlap_smaller(a["box"], b["box"]) >= config.ambiguity_overlap
                    and abs(a["verification"]["score"]-b["verification"]["score"]) <= config.ambiguity_score_gap):
                a.update(identity=None, identity_status="ambiguous_identity")
                b.update(identity=None, identity_status="ambiguous_identity")
    for name in class_map.values():
        matches = sorted((r for r in regions if r["identity"] == name), key=lambda r: r["verification"]["score"], reverse=True)
        if len(matches) > 1:
            best = matches[0]
            ambiguous = any(overlap_smaller(best["box"], r["box"]) < .5 and
                            r["verification"]["score"] >= best["verification"]["score"] * config.duplicate_score_ratio for r in matches[1:])
            for r in matches if ambiguous else matches[1:]:
                r.update(identity=None, identity_status="ambiguous_reference_locations" if ambiguous else "suppressed_duplicate")
    return {r["identity"]: r["box"] for r in regions if r["identity"]}, regions


class UltralyticsRuntime:
    """Loaded only after license review and official-asset checksum verification."""
    def __init__(self, config, reference, prompts):
        weights = config.validated_weights()
        import os
        from pathlib import Path
        directory = weights.parent / "ultralytics-config"
        (directory / "Ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "true"
        import torch
        import cv2
        from importlib.metadata import version
        from ultralytics import YOLOE
        from ultralytics.utils import SETTINGS
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        SETTINGS.update({"sync": False})
        torch.set_num_threads(1)
        # Run in a dedicated worker; changing interop/BLAS limits in the app would
        # affect gaze/sensors. Interop can only be set before parallel work begins.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        self.config, self.reference, self.prompts = config, reference, prompts
        self.model = YOLOE(str(weights))
        self.predictor_type = YOLOEVPSegPredictor
        available = torch.backends.mps.is_available()
        self.device = "mps" if config.device in ("auto", "mps") and available else "cpu"
        if config.device == "mps" and not available and not config.allow_cpu_fallback:
            raise RuntimeError("Native MPS unavailable and CPU fallback disabled")
        self.initialized = False
        self.fallback_reason = "native_mps_unavailable" if config.device == "mps" and not available else None
        self.versions = {n: version(n) for n in ("ultralytics", "torch", "torchvision")}

    def _predict(self, pixels):
        import numpy as np
        # Ultralytics interprets numpy images as BGR; the app/atlas are RGB.
        kwargs = dict(device=self.device, imgsz=self.config.image_size,
                      conf=self.config.candidate_threshold, iou=.7, agnostic_nms=False,
                      max_det=self.config.max_detections, verbose=False, save=False)
        if not self.initialized:
            kwargs.update(refer_image=np.ascontiguousarray(self.reference[..., ::-1]),
                          visual_prompts=self.prompts, predictor=self.predictor_type)
        result = self.model.predict(np.ascontiguousarray(pixels[..., ::-1]), **kwargs)[0]
        self.initialized = True
        expected = {i: f"object{i}" for i in range(len(set(self.prompts["cls"].tolist())))}
        if result.names != expected:
            raise RuntimeError("YOLOE visual prompt class map changed; refusing identity attribution")
        b = result.boxes
        return [dict(box=box, score=score, class_id=class_id) for box, score, class_id in
                zip(b.xyxyn.detach().cpu().tolist(), b.conf.detach().cpu().tolist(), b.cls.detach().cpu().tolist())] if b is not None else []

    def predict(self, pixels):
        try:
            return self._predict(pixels)
        except (RuntimeError, NotImplementedError) as error:
            message = str(error).lower()
            # Only an MPS-specific runtime/operator failure may downgrade once;
            # geometry/class-map/model errors must not be disguised as fallback.
            if self.device != "mps" or not self.config.allow_cpu_fallback or not any(s in message for s in ("mps", "metal")):
                raise
            self.device = "cpu"
            self.fallback_reason = type(error).__name__
            self.model.to("cpu")
            self.model.predictor = None
            self.initialized = False
            return self._predict(pixels)


class YOLOECharacterDetector:
    def __init__(self, names, config=None, *, runtime=None, reference_root=None):
        self.names = list(dict.fromkeys(names))
        if not self.names or any(not isinstance(n, str) or not n for n in self.names):
            raise ValueError("Supply character names")
        self.config = config or DetectorConfig()
        reference, prompts, self.class_map, self.references = build_reference_prompt(self.names, reference_root, strategy=self.config.prompt_strategy)
        self.runtime = runtime or UltralyticsRuntime(self.config, reference, prompts)
        self.thumb, self.shot, self.previous_t = None, 0, None

    def step(self, pixels, t):
        import numpy as np
        from PIL import Image
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2:
            raise ValueError("Frame must be RGB uint8 HxWx3")
        if isinstance(t, bool) or not math.isfinite(t) or t < 0 or (self.previous_t is not None and t <= self.previous_t):
            raise ValueError("Media timestamps must increase monotonically")
        thumb = np.asarray(Image.fromarray(pixels).resize((64, 36))).astype(np.float32)
        cut = self.thumb is not None and float(np.abs(thumb-self.thumb).mean()) > 38
        blank = float(pixels.astype(np.float32).mean(axis=2).std()) < 5
        self.shot += int(cut)
        self.thumb, self.previous_t = thumb, t
        started = time.perf_counter()
        boxes, regions = ({}, []) if blank else parse_predictions(self.runtime.predict(pixels), self.class_map, self.names, self.config)
        return dict(t=round(t, 6), media_timestamp_s=round(t, 6), boxes=boxes, regions=regions,
                    unknown=[n for n in self.names if n not in boxes], shot_id=self.shot, cut=cut,
                    valid_until=round(t + min(.5, 1/self.config.fps), 6),
                    source="yoloe_visual_reference_experimental", status="blank_frame" if blank else "observed" if boxes else "unavailable",
                    abstention_reason=None if boxes else "blank_frame" if blank else "no_unambiguous_visual_match",
                    coordinate_space="video-normalized", provider_confidence_available=False,
                    inference_seconds=round(time.perf_counter()-started, 6), experimental=True,
                    resources=dict(device=getattr(self.runtime, "device", "test"), cpu_threads=1,
                                   recognition_fps=self.config.fps, image_size=self.config.image_size,
                                   fallback_reason=getattr(self.runtime, "fallback_reason", None)))
