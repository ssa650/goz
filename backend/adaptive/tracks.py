"""Where each character is on screen over a clip's timeline.

People are detected with MediaPipe EfficientDet-Lite0 (COCO "person";
~7 MB model, downloaded on first use to data/models/). Identities come
from seeds: clip 1 uses the left-to-right character order of the opening
frame; every later clip starts from the previous clip's actual last frame,
so it is seeded with the previous clip's final tracked boxes. Boxes are
normalized [x0, y0, x1, y1] in video coordinates.
"""
import asyncio
from pathlib import Path

import httpx
import imageio_ffmpeg
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
             "efficientdet_lite0/float16/1/efficientdet_lite0.tflite")
SAMPLE_FPS = 4
MIN_SCORE = 0.35
MAX_JUMP = 0.35


def model_path(directory):
    path = Path(directory)/"models"/"efficientdet_lite0.tflite"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        response = httpx.get(MODEL_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path


def detect_people(video, directory, fps=SAMPLE_FPS):
    """[(t, [box, ...])] person boxes sampled at `fps`."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(directory))),
        running_mode=vision.RunningMode.IMAGE, score_threshold=MIN_SCORE,
        category_allowlist=["person"], max_results=6)
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = reader.__next__()
    reader.close()
    sw, sh = meta["size"]
    w = 640
    h = int(round(sh * w / sw / 2)) * 2
    reader = imageio_ffmpeg.read_frames(str(video), output_params=["-vf", f"fps={fps},scale={w}:{h}"])
    reader.__next__()
    out = []
    with vision.ObjectDetector.create_from_options(options) as detector:
        for i, raw in enumerate(reader):
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
            boxes = []
            for d in result.detections:
                b = d.bounding_box
                boxes.append([b.origin_x / w, b.origin_y / h, (b.origin_x + b.width) / w, (b.origin_y + b.height) / h])
            out.append((round(i / fps, 3), boxes))
    return out


def center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def assign(detections, names, seeds=None):
    """Greedy nearest-centroid tracking.

    detections: [(t, [box])]; names: characters in left-to-right staging
    order; seeds: optional {name: box} from the previous clip's last frame.
    Returns [{"t": t, "boxes": {name: box}}].
    """
    last = dict(seeds or {})
    frames = []
    for t, boxes in detections:
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:len(names)]
        current = {}
        if not last and boxes:
            for name, box in zip(names, sorted(boxes, key=lambda b: center(b)[0])):
                current[name] = box
        else:
            pairs = sorted((float(np.linalg.norm(center(b) - center(last[n]))), n, i)
                           for n in last for i, b in enumerate(boxes))
            used_names, used_boxes = set(), set()
            for dist, name, i in pairs:
                if name in used_names or i in used_boxes or dist > MAX_JUMP:
                    continue
                current[name] = boxes[i]
                used_names.add(name)
                used_boxes.add(i)
            spare_names = [n for n in names if n not in current]
            spare_boxes = sorted((b for i, b in enumerate(boxes) if i not in used_boxes), key=lambda b: center(b)[0])
            for name, box in zip(spare_names, spare_boxes):
                current[name] = box
        last.update(current)
        frames.append({"t": t, "boxes": {n: [round(v, 4) for v in b] for n, b in current.items()}})
    return frames


async def track_clip(video, directory, names, seeds=None):
    detections = await asyncio.to_thread(detect_people, video, directory)
    return assign(detections, names, seeds)


def boxes_at(track, t):
    """Boxes of the sampled frame nearest to video time t."""
    if not track:
        return {}
    frame = min(track, key=lambda f: abs(f["t"] - t))
    return frame["boxes"] if abs(frame["t"] - t) <= 1.0 / SAMPLE_FPS + 0.01 else {}


def final_boxes(track):
    for frame in reversed(track):
        if frame["boxes"]:
            return frame["boxes"]
    return {}


def target_at(boxes, nx, ny, margin=0.04):
    """Character whose (slightly padded) box contains the point; smallest wins."""
    hits = [(abs((b[2] - b[0]) * (b[3] - b[1])), name) for name, b in boxes.items()
            if b[0] - margin <= nx <= b[2] + margin and b[1] - margin <= ny <= b[3] + margin]
    return min(hits)[1] if hits else None
