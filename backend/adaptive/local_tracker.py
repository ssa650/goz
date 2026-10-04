"""Bounded local reference recognition and short-lived optical flow.

Five existing episode crops are exemplars, not a trained character model.
Recognition supplies identity; optical flow can only preserve that identity for
0.8 seconds. A separate, single-threaded process makes cancellation enforceable.
"""
import asyncio
import multiprocessing
import os
import queue
import time
import weakref

WIDTH = 640
FLOW_FPS = 8
RECOGNITION_FPS = 2
MAX_SECONDS = 120
MAX_WALL_SECONDS = 30
IDENTITY_TTL = .8
_gates = weakref.WeakKeyDictionary()


def reference_regions(pixels, names):
    """Independent per-reference SIFT/RANSAC localization, with spatial abstention."""
    import cv2
    import numpy as np
    from .tracks import _reference_features, _intersection
    grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    keys, descriptors = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.025).detectAndCompute(grey, None)
    if descriptors is None or len(descriptors) < 2:
        return {}, []
    h, w = grey.shape
    candidates = {}
    for name, references in _reference_features().items():
        if name not in names:
            continue
        found = []
        for ref_keys, ref_desc, shape, filename in references:
            if ref_desc is None:
                continue
            pairs = cv2.BFMatcher().knnMatch(ref_desc, descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
            if len(good) < 12:
                continue
            src = np.float32([ref_keys[m.queryIdx].pt for m in good])
            dst = np.float32([keys[m.trainIdx].pt for m in good])
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3, maxIters=1000)
            if matrix is None or mask is None:
                continue
            support = mask.ravel().astype(bool)
            count = int(support.sum())
            coverage = cv2.contourArea(cv2.convexHull(src[support])) / (shape[0] * shape[1]) if count >= 3 else 0
            if count < 12 or count / len(good) < .6 or coverage < .04:
                continue
            corners = np.float32([[0, 0], [shape[1], 0], [shape[1], shape[0]], [0, shape[0]]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
            if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
                continue
            lo, hi = projected.min(axis=0), projected.max(axis=0)
            area = abs(cv2.contourArea(projected)) / (w * h)
            if not .002 <= area <= .8 or np.any(lo < [-.15*w, -.15*h]) or np.any(hi > [1.15*w, 1.15*h]):
                continue
            box = [float(np.clip(lo[0]/w, 0, 1)), float(np.clip(lo[1]/h, 0, 1)),
                   float(np.clip(hi[0]/w, 0, 1)), float(np.clip(hi[1]/h, 0, 1))]
            found.append(dict(box=box, identity=name, identity_status="verified_reference",
                verification=dict(method="grayscale_sift_ransac", score_kind="geometric_inlier_count",
                                  inliers=count, reference=filename, reference_coverage=round(coverage, 4))))
        if found:
            found.sort(key=lambda r: r["verification"]["inliers"], reverse=True)
            best = found[0]
            # Multiple materially different locations for one identity are ambiguous.
            ambiguous = any(r["verification"]["inliers"] >= best["verification"]["inliers"]*.7 and
                _intersection(r["box"], best["box"]) < .5 * min(
                    (r["box"][2]-r["box"][0])*(r["box"][3]-r["box"][1]),
                    (best["box"][2]-best["box"][0])*(best["box"][3]-best["box"][1])) for r in found[1:])
            if ambiguous:
                best.update(identity=None, identity_status="ambiguous_reference_locations")
            candidates[name] = best
    regions = list(candidates.values())
    # A shared feature must not name the same foreground as both characters.
    for i, a in enumerate(regions):
        for b in regions[i+1:]:
            overlap = _intersection(a["box"], b["box"])
            smaller = min((a["box"][2]-a["box"][0])*(a["box"][3]-a["box"][1]),
                          (b["box"][2]-b["box"][0])*(b["box"][3]-b["box"][1]))
            if overlap > .6 * smaller:
                a.update(identity=None, identity_status="ambiguous_identity")
                b.update(identity=None, identity_status="ambiguous_identity")
    return {r["identity"]: r["box"] for r in regions if r["identity"]}, regions


def move_box(previous, grey, box):
    """Track geometry only. Forward/backward flow and RANSAC reject drift."""
    import cv2
    import numpy as np
    h, w = grey.shape
    mask = np.zeros_like(previous)
    x0, y0, x1, y1 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box)]
    mask[max(0,y0):min(h,y1), max(0,x0):min(w,x1)] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=60, qualityLevel=.02, minDistance=4, mask=mask)
    if points is None or len(points) < 8:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, grey, points, None, maxLevel=2)
    if nxt is None:
        return None
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(grey, previous, nxt, None, maxLevel=2)
    if back is None:
        return None
    valid = status.ravel().astype(bool) & reverse.ravel().astype(bool) & (np.linalg.norm(points-back,axis=2).ravel() < 1.5)
    if valid.sum() < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(points[valid], nxt[valid], method=cv2.RANSAC,
                                               ransacReprojThreshold=2, maxIters=500)
    if matrix is None or inliers is None or int(inliers.sum()) < 8 or inliers.mean() < .6:
        return None
    scale = float(np.linalg.norm(matrix[:,0]))
    if not .85 < scale < 1.18 or np.linalg.norm(matrix[:,2]/[w,h]) > .15:
        return None
    corners = np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    projected = cv2.transform(corners.reshape(-1,1,2), matrix).reshape(-1,2)
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    if not np.isfinite(projected).all() or np.any(lo < [0,0]) or np.any(hi > [w,h]):
        return None
    return [float(lo[0]/w),float(lo[1]/h),float(hi[0]/w),float(hi[1]/h)]


class ReferenceTracker:
    def __init__(self, names):
        self.names = names
        self.active, self.previous, self.thumb = {}, None, None
        self.shot, self.next_recognition = 0, 0

    def step(self, pixels, t):
        import cv2
        import numpy as np
        grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        thumb = cv2.resize(pixels, (64,36)).astype(np.float32)
        cut = self.thumb is not None and float(np.abs(thumb-self.thumb).mean()) > 38
        blank = float(grey.std()) < 5
        if cut or blank:
            self.active.clear()
            self.previous = None
            self.next_recognition = t
        if cut:
            self.shot += 1
        regions = []
        if not blank and t + 1e-6 >= self.next_recognition:
            boxes, regions = reference_regions(pixels, self.names)
            # Re-recognition failure immediately abstains instead of extending drift.
            self.active = {n:dict(box=b, recognized_at=t) for n,b in boxes.items()}
            self.next_recognition = t + 1/RECOGNITION_FPS
        elif not blank and self.previous is not None:
            moved = {}
            for name, item in self.active.items():
                box = move_box(self.previous, grey, item["box"]) if t-item["recognized_at"] <= IDENTITY_TTL else None
                if box:
                    moved[name] = dict(box=box, recognized_at=item["recognized_at"])
                    regions.append(dict(box=box, identity=name, identity_status="tracked_reference",
                        recognized_at=item["recognized_at"], verification=dict(method="lk_flow_ransac", score_kind="not_probability")))
            self.active = moved
        self.previous, self.thumb = grey, thumb
        boxes = {n:item["box"] for n,item in self.active.items()}
        return dict(t=round(t,6), media_timestamp_s=round(t,6), boxes=boxes, regions=regions,
            shot_id=self.shot, cut=cut, valid_until=round(min(t+1/FLOW_FPS,
                min((v["recognized_at"]+IDENTITY_TTL for v in self.active.values()),default=t+1/FLOW_FPS)),6),
            unknown=[n for n in self.names if n not in boxes], source="opencv_reference_flow",
            status="blank_frame" if blank else "observed" if boxes else "unavailable",
            abstention_reason=None if boxes else "no_reference_match",
            coordinate_space="video-normalized", provider_confidence_available=False)


def _worker(video, names, tags, output):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    import cv2
    import imageio_ffmpeg
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    # This macOS wheel uses GCD: positive values are ignored. Zero disables
    # internal parallel regions across backends (getNumThreads then reports1).
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    reader = None
    try:
        metadata = imageio_ffmpeg.read_frames(video,input_params=["-threads","1"],output_params=["-threads","1"])
        try:
            meta = next(metadata)
        finally:
            metadata.close()
        sw, sh = meta["size"]
        height = max(2, int(round(sh*WIDTH/sw/2))*2)
        if height > 1280:
            raise ValueError("Local tracking rejects extreme portrait aspect ratios.")
        reader = imageio_ffmpeg.read_frames(video, input_params=["-threads","1"],
            output_params=["-threads","1","-vf",f"fps={FLOW_FPS},scale={WIDTH}:{height}","-t",str(MAX_SECONDS)])
        next(reader)
        tracker = ReferenceTracker(names)
        for i, raw in enumerate(reader):
            if i >= MAX_SECONDS*FLOW_FPS:
                break
            pixels = np.frombuffer(raw,np.uint8).reshape(height,WIDTH,3)
            record = tracker.step(pixels,i/FLOW_FPS)
            record.update(**tags, resources=dict(opencv_threads=cv2.getNumThreads(),decode_threads=1,
                width=WIDTH,flow_fps=FLOW_FPS,recognition_fps=RECOGNITION_FPS))
            output.put(record,timeout=2)
        output.put(None,timeout=2)
    except Exception as error:
        output.put(dict(worker_error=type(error).__name__, message=str(error)[:200]),timeout=2)
    finally:
        if reader is not None:
            reader.close()
        output.close()
        output.join_thread()


def _dispose(process, output):
    if process.is_alive():
        process.terminate()
    process.join(.3)
    if process.is_alive():
        process.kill()
        process.join(.3)
    output.close()
    process.close()


async def detect_local(video, names, *, clip_id, session_id, generation_id=None, on_progress=None):
    """One local worker globally per event loop; bounded queue/CPU/time/cancel."""
    loop = asyncio.get_running_loop()
    gate = _gates.setdefault(loop, asyncio.Semaphore(1))
    async with gate:
        context = multiprocessing.get_context("spawn")
        output = context.Queue(maxsize=32)
        process = context.Process(target=_worker, args=(str(video),list(names),
            dict(clip_id=clip_id,session_id=session_id,generation_id=generation_id),output),daemon=True)
        process.start()
        result = []
        try:
            async with asyncio.timeout(MAX_WALL_SECONDS):
                while True:
                    received = False
                    for _ in range(32):
                        try:
                            record = output.get_nowait()
                        except queue.Empty:
                            break
                        if record is None:
                            if on_progress:
                                on_progress(list(result))
                            return result
                        if "worker_error" in record:
                            raise RuntimeError(f"OpenCV worker {record['worker_error']}: {record['message']}")
                        result.append(record)
                        received = True
                    if received and on_progress:
                        on_progress(list(result))
                    if not process.is_alive() and not received:
                        raise RuntimeError("OpenCV worker exited without a completion record.")
                    await asyncio.sleep(.02)
        finally:
            # Shield cleanup: cancellation must never leave unbounded CPU work.
            await asyncio.shield(asyncio.to_thread(_dispose,process,output))
