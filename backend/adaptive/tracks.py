"""Frame evidence for character attribution.

Florence supplies unprompted dense visual regions. Names come from unambiguous
visual captions or grayscale local-reference geometry, never requested cast,
color, staging, or the generation script. Unknown regions are retained. All
boxes use normalized video coordinates and bounded past-only frame validity.
"""
import asyncio
import hashlib
import json
import math
from functools import lru_cache
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
    # A COCO person detector supplies no character identity. Staging/seeds are
    # story metadata, not visual evidence, so live people mode must abstain.
    return [dict(t=t, boxes={}, regions=[dict(box=b, identity=None, identity_status="unknown") for b in boxes],
                 source="mediapipe_person_no_identity", coordinate_space="video-normalized",
                 valid_until=t + MAX_DETECTION_AGE_S) for t, boxes in detections]


# Unprompted visual regions avoid forcing every expected character into a frame.
FAL_DETECTOR = "https://fal.run/fal-ai/florence-2-large/dense-region-caption"
FAL_FPS = 0.5
FAL_CONCURRENCY = 2
MAX_DETECTION_AGE_S = 0.8
CACHE_SCHEMA = 5
TRACKING_PROVIDERS = {"color", "fal", "opencv", "people", "yoloe"}


def tracking_provider(value=None):
    import os
    value = os.getenv("GOZ_TRACKER", "color") if value is None else value
    if value not in TRACKING_PROVIDERS:
        raise ValueError("Tracking provider must be color, fal, opencv, people or yoloe.")
    return value



def yoloe_config():
    """Explicit local experiment config. Never download/install on selection."""
    import os
    from .yoloe_detector import DetectorConfig
    options = dict(weights=os.getenv("GOZ_YOLOE_WEIGHTS", ""),
        weights_sha256=os.getenv("GOZ_YOLOE_SHA256", ""),
        license_reviewed=os.getenv("GOZ_YOLOE_LICENSE_REVIEWED") == "1",
        device=os.getenv("GOZ_YOLOE_DEVICE", "cpu"), fps=2, queue_size=2)
    # External isolated interpreter is optional in the provider's evolving contract.
    runtime = os.getenv("GOZ_YOLOE_PYTHON", "")
    if runtime:
        if "runtime_python" not in DetectorConfig.__dataclass_fields__:
            raise RuntimeError("YOLOE isolated runtime adapter is not available yet")
        options["runtime_python"] = runtime
    return DetectorConfig(**options)


def yoloe_availability():
    """Offline experiment only: held-out wrong labels must not drive gaze."""
    return dict(available=False, experimental=True,
        reason="held-out identity check failed; offline only",
        benchmark=dict(labelled_frames=24, visible_instances=35, true_positive=11,
            false_positive=5, false_negative=24, wrong_identity_labels=4,
            precision=.6875, recall=.3143,
            annotation_quality="Codex-reviewed approximate boxes; not independent human gold"))


def failure_details(error, key=""):
    """Useful provider diagnostics without response bodies, URLs or secrets."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        category = {401:"authentication", 402:"payment_or_credit", 403:"authorization",
                    429:"rate_limit"}.get(status, "provider_http_error")
        return dict(error=type(error).__name__, http_status=status, error_category=category,
                    error_message=f"Florence returned HTTP {status} ({category}).")
    if isinstance(error, httpx.TimeoutException):
        return dict(error=type(error).__name__, error_category="timeout",
                    error_message="Florence timed out; connection cause is unverified.")
    message = str(error)[:240] if isinstance(error, ValueError) else "Vision request or parsing failed."
    if key:
        message = message.replace(key, "[redacted]")
    return dict(error=type(error).__name__, error_category="parse_or_provider_rejection" if isinstance(error,ValueError) else "request_error",
                error_message=message)


def merge_tracking(local, remote):
    """Full-clip reference/flow records plus fresh cloud evidence; never cross cuts.

    A cloud name can remain usable for its existing .8s TTL. Local recognition
    has priority on conflicting identity; overlapping different names abstain.
    Recombining late cloud results updates logs/overlay only, not frozen decisions.
    """
    from copy import deepcopy
    cuts = [f["t"] for f in local if f.get("cut")]
    local_at = {f["t"]: f for f in local}
    remote_at = {f["t"]: f for f in remote}
    result = []
    for t in sorted(set(local_at) | set(remote_at)):
        lf = local_at.get(t)
        candidates = [f for f in remote if f["t"] <= t + 1e-6 and t <= f.get("valid_until",f["t"]+.8) + 1e-6
                      and not any(f["t"] < cut <= t for cut in cuts)]
        rf = max(candidates,key=lambda f:f["t"]) if candidates else None
        record = deepcopy(lf or remote_at[t])
        boxes = deepcopy(lf.get("boxes",{}) if lf else {})
        sources = {name: "opencv_reference_flow" for name in boxes}
        if rf:
            for name,box in rf.get("boxes",{}).items():
                if name not in boxes:
                    boxes[name] = box
                    sources[name] = rf.get("source","florence")
            record["cloud_detection_t"] = rf["t"]
        conflicts = set()
        for name,a in boxes.items():
            for other,b in boxes.items():
                if name != other and _intersection(a,b) > .6*min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])):
                    conflicts.update((name,other))
        record.update(boxes={n:b for n,b in boxes.items() if n not in conflicts},
                      box_sources={n:s for n,s in sources.items() if n not in conflicts},
                      ambiguous_identities=sorted(conflicts), source="florence+opencv_reference_flow")
        if record["boxes"]:
            record["status"] = "observed"
        if lf and rf and any(n in sources and sources[n] != "opencv_reference_flow" for n in record["boxes"]):
            record["valid_until"] = min(record["valid_until"],rf["valid_until"])
        if not lf:
            next_cut = next((cut for cut in cuts if cut > t),None)
            if next_cut is not None:
                record["valid_until"] = min(record["valid_until"],next_cut)
        result.append(record)
    return result

# Visually inspected face/body crops from the existing episode references.
# These are identity exemplars, never evidence that a character is in a new frame.
# Matching uses grayscale SIFT descriptors and RANSAC geometry, not color or cast order.
REFERENCE_CROPS = {
    "SpongeBob": [("00-30.jpg", (.555, .51, .795, .93)),
                  ("01-45.jpg", (.175, .475, .48, .94)),
                  ("00-15.jpg", (.80, .54, .89, .75))],
    "Patrick": [("00-30.jpg", (.24, .18, .445, .80)),
                ("01-45.jpg", (.59, .15, .79, .84))],
}


def sample_scene_frames(video, fps=FAL_FPS, width=768, observation_seconds=None):
    """Exact decoded frame times, bounded API frames, and local shot boundaries.

    Decode at native cadence so a cut between sparse API samples immediately
    invalidates the prior box. The cut heuristic is intentionally conservative;
    max-age expiry still bounds missed cuts. Never interpolate across a cut.
    """
    import cv2
    from io import BytesIO
    from PIL import Image
    reader = imageio_ffmpeg.read_frames(str(video))
    meta = next(reader)
    reader.close()
    sw, sh = meta["size"]
    native_fps = float(meta["fps"])
    if observation_seconds is None:
        interval = max(1 / fps, float(meta.get("duration") or 1) / 8)
    else:
        if not isinstance(observation_seconds, (int, float)) or not math.isfinite(observation_seconds) or observation_seconds <= 0:
            raise ValueError("Observation duration must be a positive finite number.")
        # Same eight-frame request budget, concentrated in the decision window.
        # 3.5s gives real frames every .5s, still with the .8s expiry and cuts.
        interval = max(1 / native_fps, observation_seconds / 7)
    h = int(round(sh * width / sw / 2)) * 2
    reader = imageio_ffmpeg.read_frames(str(video), output_params=["-vf", f"scale={width}:{h}"])
    next(reader)
    out, previous, next_sample, count, shot = [], None, 0.0, 0, 0
    try:
        for i, raw in enumerate(reader):
            t = i / native_fps
            if observation_seconds is not None and t > observation_seconds + 1e-6:
                break
            pixels = np.frombuffer(raw, np.uint8).reshape(h, width, 3)
            thumb = cv2.resize(pixels, (64, 36))
            # A heuristic cut marker, not a probability or a semantic assertion.
            cut = previous is not None and float(np.abs(thumb.astype(float) - previous).mean()) > 38
            previous = thumb.astype(float)
            if cut:
                shot += 1
                out.append(dict(t=round(t, 6), jpeg=None, width=width, height=h, shot=shot, cut=True))
            if t + 1e-6 < next_sample or count >= 8:
                continue
            count += 1
            next_sample += interval
            buffer = BytesIO()
            if float(pixels.mean(axis=2).std()) >= 5:
                Image.frombytes("RGB", (width, h), raw).save(buffer, format="JPEG", quality=90)
            entry = dict(t=round(t, 6), jpeg=buffer.getvalue() or None,
                         width=width, height=h, shot=shot, cut=cut)
            if out and out[-1]["t"] == entry["t"]:
                out[-1] = entry
            else:
                out.append(entry)
    finally:
        reader.close()
    return out


def sample_jpegs(video, fps=FAL_FPS, width=768):
    return [(f["t"], f["jpeg"], f["width"], f["height"]) for f in sample_scene_frames(video, fps, width)]


def _reference_features():
    import cv2
    from PIL import Image
    root = Path(__file__).resolve().parents[2] / "presets" / "secret-box" / "frames"
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=.025)
    result = {}
    for name, crops in REFERENCE_CROPS.items():
        result[name] = []
        for filename, rect in crops:
            with Image.open(root / filename) as image:
                w, h = image.size
                crop = image.crop(tuple(int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(rect)))
                crop.thumbnail((512, 512))
                grey = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
            keys, descriptors = sift.detectAndCompute(grey, None)
            result[name].append((keys, descriptors, grey.shape, filename))
    return result


_reference_features = lru_cache(maxsize=1)(_reference_features)


def verify_region(pixels, box, allowed_names):
    """Return identity and auditable local-feature evidence, or abstain.

    SIFT inliers are a geometric match count, NOT model confidence. Two identities
    with similar match counts are ambiguous. This deliberately does not generalize
    identity from yellow/pink colors or a provider's franchise-name caption.
    """
    import cv2
    h, w = pixels.shape[:2]
    x0, y0, x1, y1 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box)]
    crop = pixels[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    evidence = dict(method="grayscale_sift_ransac", score_kind="geometric_inlier_count", matches={}, reference_boxes={})
    if crop.size == 0 or min(crop.shape[:2]) < 15:
        return None, evidence
    factor = min(1.0, 512 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=factor, fy=factor)
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=900, contrastThreshold=.025).detectAndCompute(grey, None)
    if descriptors is None or len(descriptors) < 2:
        return None, evidence
    for name, references in _reference_features().items():
        if name not in allowed_names:
            continue
        best = 0
        for keys, desc, shape, filename in references:
            pairs = cv2.BFMatcher().knnMatch(desc, descriptors, k=2)
            good = [a for a, b in pairs if a.distance < .7 * b.distance]
            if len(good) < 8:
                continue
            src = np.float32([keys[m.queryIdx].pt for m in good])
            dst = np.float32([keypoints[m.trainIdx].pt for m in good])
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4)
            if matrix is None or mask is None:
                continue
            inliers = int(mask.sum())
            support = src[mask.ravel().astype(bool)]
            # Reject matches on one tiny/shared cartoon feature.
            coverage = cv2.contourArea(cv2.convexHull(support)) / (shape[0] * shape[1]) if len(support) >= 3 else 0
            if inliers >= 8 and inliers / len(good) >= .5 and coverage >= .025:
                if inliers > best:
                    corners = np.float32([[0, 0], [shape[1], 0], [shape[1], shape[0]], [0, shape[0]]]).reshape(-1, 1, 2)
                    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
                    if np.isfinite(projected).all():
                        lo, hi = projected.min(axis=0), projected.max(axis=0)
                        area = (hi[0] - lo[0]) * (hi[1] - lo[1]) / (grey.shape[0] * grey.shape[1])
                        if .002 < area < 1.5:
                            best = inliers
                            evidence["reference_boxes"][name] = [float(np.clip(lo[0] / grey.shape[1], 0, 1)),
                                float(np.clip(lo[1] / grey.shape[0], 0, 1)), float(np.clip(hi[0] / grey.shape[1], 0, 1)),
                                float(np.clip(hi[1] / grey.shape[0], 0, 1))]
        evidence["matches"][name] = best
    ranked = sorted(evidence["matches"].items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 8:
        return None, evidence
    if len(ranked) > 1 and ranked[0][1] < max(ranked[1][1] + 5, ranked[1][1] * 1.8):
        evidence["ambiguous"] = True
        return None, evidence
    return ranked[0][0], evidence


def visual_identity(label, names):
    """An unprompted region caption is visual model evidence, not ground truth.

    Florence prefixes Patrick's proper name with the show title. Strip only that
    measured title prefix; multiple other identities remain ambiguous. Descriptions,
    colors, expected cast and the user's generation prompt never supply identity.
    """
    import re
    text = re.sub(r"[^a-z0-9 ]", " ", label.lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^spongebob squarepants (?=patrick star\b)", "", text)
    # Observed generated-frame failure: the show title prefixed a treasure
    # chest region on Patrick. A franchise title plus a prop is not an identity.
    if any(term in text for term in ("treasure chest", "flower", "logo", "pattern")) and "patrick star" not in text:
        return None
    aliases = {"SpongeBob": ["spongebob", "sponge bob"], "Patrick": ["patrick star"],
               "Squidward": ["squidward"], "Sandy": ["sandy cheeks"], "Mr. Krabs": ["mr krabs"],
               "Gary": ["gary the snail"], "Plankton": ["plankton"]}
    matches = {name for name, words in aliases.items()
               if any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in words)}
    # Custom targets require their actual proper name in an unprompted caption;
    # no fuzzy nearest-name or descriptive-color matching.
    for name in names:
        if name not in aliases and re.search(r"\b" + re.escape(name.lower()) + r"\b", text):
            matches.add(name)
    return next(iter(matches)) if len(matches) == 1 and next(iter(matches)) in names else None


def parse_regions(response, jpeg, width, height, characters):
    """Interpret actual visual regions; a missing expected identity stays missing."""
    from io import BytesIO
    from PIL import Image
    pixels = np.array(Image.open(BytesIO(jpeg)).convert("RGB"))
    regions, identified = [], {}
    raw = response.get("results", {})
    if not isinstance(raw, dict) or not isinstance(raw.get("bboxes"), list):
        raise ValueError("Florence returned no bounding-box result list.")
    names = [c["name"] for c in characters]
    for b in raw["bboxes"]:
        if not isinstance(b, dict) or not all(isinstance(b.get(k), (int, float)) and math.isfinite(b[k]) for k in ("x", "y", "w", "h")):
            continue
        if b["w"] <= 0 or b["h"] <= 0:
            continue
        box = [b["x"] / width, b["y"] / height, (b["x"] + b["w"]) / width, (b["y"] + b["h"]) / height]
        box = [round(max(0.0, min(1.0, v)), 6) for v in box]
        if not .002 <= (box[2] - box[0]) * (box[3] - box[1]) <= .85:
            continue
        name, evidence = verify_region(pixels, box, names)
        caption_name = visual_identity(str(b.get("label") or ""), names)
        status = "verified_reference" if name else "unknown"
        if name and caption_name and name != caption_name:
            name, status = None, "ambiguous_conflicting_evidence"
        elif not name and caption_name and not evidence.get("ambiguous"):
            name, status = caption_name, "model_observed"
        evidence["caption_identity"] = caption_name
        region = dict(box=box, label=str(b.get("label") or ""), identity=name,
                      identity_status=status, verification=evidence)
        regions.append(region)
        if name:
            identified.setdefault(name, []).append(box)
    # A small, clearly marked local reference fallback can recover an exact
    # visible pose that Florence omitted; it never invents a provider detection.
    for name in names:
        if name not in identified and name in REFERENCE_CROPS:
            matched, evidence = verify_region(pixels, [0, 0, 1, 1], [name])
            if matched and evidence["matches"].get(name, 0) >= 12:
                box = evidence["reference_boxes"].get(name)
                if box:
                    identified[name] = [box]
                    regions.append(dict(box=box, label="", identity=name, identity_status="verified_reference",
                                        source="local_reference_geometry", verification=evidence))
    boxes = {}
    for name, candidates in identified.items():
        largest = max(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        # Duplicate face/body regions are fine only when spatially consistent.
        if all(_intersection(b, largest) >= .8 * (b[2] - b[0]) * (b[3] - b[1]) for b in candidates):
            boxes[name] = largest
        else:
            for region in regions:
                if region["identity"] == name:
                    region.update(identity=None, identity_status="ambiguous_duplicate")
    return boxes, regions


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def corroborates_body(record, identity, box):
    """Reject unsupported vision identities without treating cast as presence.

    This geometric guard needs an actual local-reference match or an independently
    observed face/body/clothing region. A franchise label on a prop/background is
    explicitly insufficient. It lowers recall on poorly described cartoon shots.
    """
    import re
    for region in record.get("regions", []):
        candidate = region["box"]
        area = (candidate[2]-candidate[0])*(candidate[3]-candidate[1])
        if not area or _intersection(candidate, box) / area < .5:
            continue
        if region.get("identity_status") == "verified_reference" and region.get("identity") == identity:
            return True
        label = region.get("label", "").lower()
        if any(word in label for word in ("flower", "arrow", "logo", "pattern", "background")):
            continue
        if any(re.search(r"\b"+word+r"\b",label) for word in ("face", "character", "person", "starfish", "tie", "shirt", "shorts", "expression", "spongebob", "squidward", "patrick star")):
            return True
    return False


async def detect_characters(video, characters, key, transport=None, fps=FAL_FPS, directory=None,
                            clip_id=None, session_id=None, on_progress=None, observation_seconds=None):
    """Eight bounded, cached calls; adaptive callers sample their opening window."""
    import base64
    import time
    clip_id = clip_id or hashlib.sha256(Path(video).read_bytes()).hexdigest()
    cache_dir = Path(directory) / "detections" if directory else None
    frames = await asyncio.to_thread(sample_scene_frames, video, fps, 768, observation_seconds)
    gate, rejected = asyncio.Semaphore(FAL_CONCURRENCY), False
    requests = {}
    completed = {}
    identities, provenance = None, {"status": "processing"}

    def verified_record(record):
        from copy import deepcopy
        record = deepcopy(record)
        if transport is not None:
            return record  # Explicit mocked transport retains fixture behavior.
        record["identity_verification"] = provenance
        if identities is not None and record["t"] in identities:
            detections = deepcopy(identities[record["t"]])
            grouped = {}
            for detection in detections:
                if detection["identity"] != "unknown" and corroborates_body(record, detection["identity"], detection["box"]):
                    grouped.setdefault(detection["identity"], []).append(detection["box"])
                    detection["identity_status"] = "model_observed_with_foreground_support"
                else:
                    detection["identity_status"] = "unknown_unconfirmed_foreground"
            record["boxes"] = {name: boxes[0] for name, boxes in grouped.items() if len(boxes) == 1}
            record.update(identity_detections=detections, source="openai_reference_vision",
                          candidate_region_source=FAL_DETECTOR, identity_status="model_observed", status="observed")
        else:
            # Until independent verification arrives, only actual reference
            # geometry is available; a Florence franchise caption is insufficient.
            record["boxes"] = {name: box for name, box in record["boxes"].items()
                if any(r.get("identity") == name and r.get("identity_status") == "verified_reference"
                       for r in record.get("regions", []))}
        return record

    def available():
        result = [verified_record(completed[t]) for t in sorted(completed)]
        # Account for known sampled cuts even if their requests are unfinished.
        for record in result:
            next_t = next((f["t"] for f in frames if f["t"] > record["t"]), None)
            if next_t is not None:
                record["valid_until"] = min(record["valid_until"], next_t)
        return result

    def publish(record=None):
        if record is not None:
            completed[record["t"]] = record
        if on_progress:
            on_progress(available())

    async def fetch(client, jpeg, digest):
        nonlocal rejected
        cache = cache_dir / f"regions-{digest}.json" if cache_dir else None
        if cache and cache.is_file():
            try:
                return json.loads(cache.read_text()), True
            except (ValueError, OSError):
                pass
        async with gate:
            if rejected:
                raise ValueError("Florence unavailable after provider rejection.")
            response = await client.post(FAL_DETECTOR, headers={"Authorization": f"Key {key}"}, json=dict(
                image_url="data:image/jpeg;base64," + base64.b64encode(jpeg).decode()))
            if response.status_code in (401, 402, 403, 429):
                rejected = True
        response.raise_for_status()
        payload = response.json()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload))
            temporary.replace(cache)
        return payload, False

    async def query(client, frame):
        nonlocal rejected
        t, jpeg = frame["t"], frame["jpeg"]
        record = dict(t=t, media_timestamp_s=t, clip_id=clip_id, session_id=session_id,
                      coordinate_space="video-normalized", shot_id=frame["shot"], cut=frame["cut"],
                      valid_until=t + MAX_DETECTION_AGE_S, boxes={}, regions=[], unknown=[],
                      source="florence_dense_regions+reference_geometry", provider_confidence_available=False)
        if jpeg is None:
            record["status"] = "shot_boundary" if frame["cut"] else "blank_frame"
            publish(record)
            return record
        digest = hashlib.sha256(jpeg + str(CACHE_SCHEMA).encode()).hexdigest()
        started = time.perf_counter()
        try:
            shared = digest in requests
            if not shared:
                requests[digest] = asyncio.create_task(fetch(client, jpeg, digest))
            payload, cached = await requests[digest]
            record["provider_request_shared"] = shared
            boxes, regions = await asyncio.to_thread(parse_regions, payload, jpeg, frame["width"], frame["height"], characters)
            record.update(boxes=boxes, regions=regions, status="observed", cache_hit=cached,
                          provenance=dict(endpoint=FAL_DETECTOR, frame_sha256=hashlib.sha256(jpeg).hexdigest(),
                                          frame_width=frame["width"], frame_height=frame["height"], cache_schema=CACHE_SCHEMA),
                          processing_seconds=round(time.perf_counter() - started, 4))
        except Exception as error:
            record.update(status="unavailable", unknown=[c["name"] for c in characters], **failure_details(error,key))
        publish(record)
        return record

    from .identity import verify_frames
    async def florence():
        async with httpx.AsyncClient(timeout=25, transport=transport) as client:
            return await asyncio.gather(*(query(client, f) for f in frames))
    if transport is None:
        async def verify():
            nonlocal identities, provenance
            identities, provenance = await verify_frames(frames, [c["name"] for c in characters], directory)
            publish()
        jobs = [asyncio.create_task(florence()), asyncio.create_task(verify())]
    else:
        jobs = [asyncio.create_task(florence())]
    try:
        await asyncio.gather(*jobs)
    finally:
        for task in jobs + list(requests.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(*jobs, *requests.values(), return_exceptions=True)
    track = available()

    return track


def frame_at(track, t, clip_id=None):
    """Past-only bounded attribution. No future boxes and no carry across cuts."""
    available = [f for f in track if f["t"] <= t + 1e-6 and
                 (clip_id is None or f.get("clip_id", clip_id) == clip_id)]
    if not available:
        return None
    frame = max(available, key=lambda f: f["t"])
    end = frame.get("valid_until", frame["t"] + MAX_DETECTION_AGE_S)
    return frame if t <= end + 1e-6 else None


def boxes_at(track, t, clip_id=None):
    frame = frame_at(track, t, clip_id)
    return frame.get("boxes", {}) if frame else {}


def final_boxes(track):
    # Never seed a new scene from some older nonempty frame.
    return track[-1].get("boxes", {}) if track else {}


def target_at(boxes, nx, ny, margin=0.0):
    """Only an unambiguous hit is a character. Outside remains outside."""
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (nx, ny)) or not (0 <= nx <= 1 and 0 <= ny <= 1):
        return None
    hits = [name for name, b in boxes.items()
            if b[0] - margin <= nx <= b[2] + margin and b[1] - margin <= ny <= b[3] + margin]
    return hits[0] if len(hits) == 1 else None
