"""Bounded local reference recognition and short-lived optical flow.

Five existing episode crops are exemplars, not a trained character model.
Recognition supplies identity; optical flow can only preserve that identity for
0.8 seconds. A separate, single-threaded process cooperatively stops between native calls.
"""
import asyncio
import multiprocessing
import os
import queue
import threading
import time
import weakref

WIDTH = 640
FLOW_FPS = 8
RECOGNITION_FPS = 2
MAX_SECONDS = 120
MAX_WALL_SECONDS = 30
IDENTITY_TTL = .8
OUTPUT_CAPACITY = 2
LIVE_MAX_AGE_S = .25
MAX_FRAME_BYTES = 60_000
_gates = weakref.WeakKeyDictionary()
_worker_slot = threading.BoundedSemaphore(1)


class LatestFrameSlot:
    """One pending live input; replace old work rather than accumulating latency.

    Not used for offline whole-clip history. Callers own a single inference worker;
    pending inputs and in-flight publication are guarded separately by provenance.
    """
    def __init__(self):
        self.pending = None
        self.dropped = 0
        self.provenance = None

    def offer(self, frame, media_t, provenance):
        if provenance != self.provenance:
            self.pending = None
            self.provenance = provenance
        if self.pending is not None:
            self.dropped += 1
        self.pending = (frame, media_t, provenance)

    def take(self, media_t):
        item, self.pending = self.pending, None
        if item is not None and (item[2] != self.provenance or media_t-item[1] > LIVE_MAX_AGE_S or item[1] > media_t+.001):
            self.dropped += 1
            return None
        return item

    def cancel(self):
        self.provenance = None
        self.pending = None

    def publishable(self, item, media_t):
        return (item is not None and item[2] == self.provenance and self.provenance is not None
                and -.001 <= media_t-item[1] <= LIVE_MAX_AGE_S)


def playback_target(state, now):
    """Extrapolate only recent presented media, never pauses, seeks or old ticks."""
    if not state or not state.get("playing") or not state.get("current", True):
        return None
    age = now-state["at"]
    if not 0 <= age <= .6:
        return None
    return state["media_t"] + age*state.get("rate", 1)


def inference_due(media_t, state, now):
    target = playback_target(state, now)
    return target is None or media_t >= target-LIVE_MAX_AGE_S


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



def _new_tracker(names, tags, provider):
    if provider == "color":
        from .color_detector import ColorCharacterDetector
        return ColorCharacterDetector(names, clip_id=tags["clip_id"], session_id=tags["session_id"],
            generation_id=tags.get("generation_id"))
    if provider != "opencv":
        raise ValueError("Local worker provider must be color or opencv")
    return ReferenceTracker(names)

def _emit(output, record, stop):
    while not stop.is_set():
        try:
            output.put(record, timeout=.05)
            return True
        except queue.Full:
            continue
    return False


def _worker(video, names, tags, output, stop, playback=None, provider="opencv"):
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    if stop.is_set():
        output.cancel_join_thread()
        output.close()
        return
    _emit(output, dict(diagnostic_event="worker_entered", worker_pid=os.getpid()), stop)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    capture = None
    try:
        import cv2
        import numpy as np
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        imports_ms = (time.perf_counter()-wall_started)*1000
        opened = time.perf_counter()
        # One decoder, explicitly single-threaded. The old metadata reader and
        # second FFmpeg reader both had to start before the first output frame.
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_N_THREADS) != 1:
            raise ValueError("Single-thread OpenCV video decoder unavailable.")
        sw, sh = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        source_fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not all(np.isfinite(n) and n > 0 for n in (sw, sh, source_fps, count)):
            raise ValueError("Local tracking requires finite video metadata.")
        width = min(WIDTH, max(2, int(round(WIDTH*sw/sh/2))*2)) if provider == "color" else WIDTH
        height = max(2, int(round(sh*width/sw/2))*2)
        fps = 4 if provider == "color" else FLOW_FPS
        if height > 1280:
            raise ValueError("Local tracking rejects extreme portrait aspect ratios.")
        _emit(output, dict(diagnostic_event="decoder_open", imports_ms=round(imports_ms,3),
            decoder_open_ms=round((time.perf_counter()-opened)*1000,3),
            decoder="opencv_ffmpeg", decode_threads=1, source_fps=source_fps,
            width=width, height=height), stop)
        duration = min(MAX_SECONDS, count/source_fps)
        tracker = _new_tracker(names, tags, provider)
        import hashlib, marshal
        diagnostic_functions = dict(worker=_worker, trackerStep=type(tracker).step)
        if provider == "color":
            diagnostic_functions["candidateFeatures"] = type(tracker)._candidates
        else:
            diagnostic_functions.update(referenceRegions=reference_regions, opticalFlow=move_box)
        _emit(output, dict(diagnostic_event="tracker_loaded", provider=provider,
            opencv_version=cv2.__version__, functionCodeSha256={name:hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
                for name,fn in diagnostic_functions.items()}), stop)
        dropped, epoch, sample_index, previous_t = 0, None, 0, -1.0
        while sample_index/fps < duration-1e-6 and not stop.is_set():
            if time.perf_counter()-wall_started >= MAX_WALL_SECONDS:
                break
            state = None
            reset = False
            if playback is not None:
                with playback.get_lock():
                    media_t, at, playing, current, current_epoch, rate = playback[:]
                state = dict(media_t=media_t, at=at, playing=bool(playing), current=bool(current), epoch=current_epoch, rate=rate)
                reset = epoch is not None and current_epoch != epoch
                epoch = current_epoch
            requested_t = sample_index/fps
            target = playback_target(state, time.monotonic())
            skipped = 0
            if target is not None and requested_t < target-LIVE_MAX_AGE_S:
                # Seek over stale decoded input instead of spending queue/decoder
                # time emitting every expired frame. This is an input gap, not
                # fabricated evidence of a visual scene cut.
                next_index = max(sample_index, int(target*fps))
                skipped = next_index-sample_index
                sample_index, requested_t = next_index, next_index/fps
                if requested_t >= duration-1e-6:
                    break
                dropped += skipped
            if reset or skipped:
                tracker = _new_tracker(names, tags, provider)
            decoded = time.perf_counter()
            # Normal cadence decodes forward. Seeking every sample repeatedly
            # decodes the preceding GOP; only an actual input gap needs a seek.
            if previous_t < 0 or skipped or requested_t < previous_t:
                capture.set(cv2.CAP_PROP_POS_MSEC, requested_t*1000)
            ok, bgr = capture.read()
            while ok:
                actual_t = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not np.isfinite(actual_t) or actual_t < 0:
                    raise ValueError("Decoder omitted its actual media timestamp.")
                if actual_t+1e-6 >= requested_t or stop.is_set():
                    break
                ok, bgr = capture.read()
            if not ok or stop.is_set():
                break
            sample_index += 1
            if actual_t <= previous_t+1e-6:
                continue
            previous_t = actual_t
            pixels = cv2.cvtColor(cv2.resize(bgr, (width,height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            decode_ms = (time.perf_counter()-decoded)*1000
            inference_started = time.perf_counter()
            record = tracker.step(pixels,actual_t)
            record["input_gap"] = bool(reset or skipped)
            record["decode"] = dict(decoder="opencv_ffmpeg", requested_media_t=round(requested_t,6),
                actual_media_t=round(actual_t,6), decode_ms=round(decode_ms,3))
            record["scheduling"] = dict(mode="playback_priority" if playback is not None else "offline_full_clip",
                dropped_stale_inputs=dropped, queue_capacity=OUTPUT_CAPACITY,
                inference_ms=round((time.perf_counter()-inference_started)*1000,3),
                worker_cpu_scope="opencv_python_process_includes_decode", decoder="opencv_ffmpeg",
                worker_cpu_s=round(time.process_time()-cpu_started,4),
                worker_wall_s=round(time.perf_counter()-wall_started,4))
            record.update(**tags, produced_at=time.time(), resources=dict(opencv_threads=cv2.getNumThreads(),
                decode_threads=1, width=width, flow_fps=fps, recognition_fps=fps if provider == "color" else RECOGNITION_FPS))
            if not _emit(output, record, stop):
                break
        _emit(output, None, stop)
    except Exception as error:
        _emit(output, dict(worker_error=type(error).__name__, message=str(error)[:200]), stop)
    finally:
        if capture is not None:
            capture.release()
        if stop.is_set():
            output.cancel_join_thread()
        output.close()
        if not stop.is_set():
            output.join_thread()



def _frames_worker(names, tags, output, stop, playback, inputs, provider):
    """Infer only bounded presented video pixels; never open a provider URL."""
    import cv2
    import numpy as np
    started = time.perf_counter()
    tracker = _new_tracker(names, tags, provider)
    previous_t, previous_epoch = -1, None
    counts = {}
    def stage(reason, **fields):
        count = counts[reason] = counts.get(reason,0)+1
        if count & (count-1) == 0:
            _emit(output,dict(diagnostic_event="presented_worker_stage",reason=reason,count=count,**fields),stop)
    _emit(output, dict(diagnostic_event="worker_entered", worker_pid=os.getpid(),
        input_source="presented_video_frame"), stop)
    try:
        while not stop.is_set() and time.perf_counter()-started < MAX_SECONDS:
            try:
                item = inputs.get(timeout=.05)
            except queue.Empty:
                continue
            if item.get("tags") != tags:
                stage("provenance_mismatch")
                continue
            t, epoch = item["t"], item["epoch"]
            with playback.get_lock():
                media_t, at, playing, current, current_epoch, rate = playback[:]
            target = playback_target(dict(media_t=media_t, at=at, playing=bool(playing),
                current=bool(current), rate=rate), time.monotonic())
            if target is None or epoch != current_epoch or not -.001 <= target-t <= LIVE_MAX_AGE_S:
                stage("inactive_or_stale",mediaTime=t,playbackTarget=target,epoch=epoch,currentEpoch=current_epoch)
                continue
            if epoch == previous_epoch and t <= previous_t:
                stage("out_of_order",mediaTime=t)
                continue
            gap = previous_epoch != epoch or t-previous_t > LIVE_MAX_AGE_S+.001
            if gap:
                tracker = _new_tracker(names, tags, provider)
            encoded = item["jpeg"]
            if not 0 < len(encoded) <= MAX_FRAME_BYTES:
                stage("payload_limit")
                continue
            # Inspect dimensions before native allocation; a small compressed
            # payload must not cause an unbounded decoded image allocation.
            from io import BytesIO
            from PIL import Image
            try:
                with Image.open(BytesIO(encoded)) as image:
                    if image.format != "JPEG" or image.width > WIDTH or image.height > 1280:
                        stage("dimensions_or_format")
                        continue
            except (ValueError,OSError):
                stage("invalid_jpeg")
                continue
            bgr = cv2.imdecode(np.frombuffer(encoded,np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[1] > WIDTH or bgr.shape[0] > 1280:
                stage("decode_failed")
                continue
            stage("infer",mediaTime=t)
            pixels = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            inference = time.perf_counter()
            record = tracker.step(pixels,t)
            previous_t, previous_epoch = t, epoch
            record.update(**tags, playback_epoch=epoch, input_gap=gap, produced_at=time.time(),
                decode=dict(decoder="browser_presented_jpeg",actual_media_t=t),
                scheduling=dict(mode="presented_latest_frame",queue_capacity=1,
                    inference_ms=round((time.perf_counter()-inference)*1000,3)),
                resources=dict(width=bgr.shape[1],height=bgr.shape[0],opencv_threads=cv2.getNumThreads()))
            if not _emit(output,record,stop):
                break
        _emit(output,None,stop)
    except Exception as error:
        _emit(output,dict(worker_error=type(error).__name__,message="Presented frame inference failed."),stop)


def _dispose(process, output):
    """Cooperatively owned worker exits; never terminate/kill a process."""
    process.join()
    output.close()
    process.close()


def _retire(process, output, gate, loop, completion, keepalive):
    # keepalive retains spawn synchronization handles until actual child exit.
    def finished():
        gate.release()
        if not completion.done():
            completion.set_result(None)
    try:
        _dispose(process, output)
    finally:
        _worker_slot.release()
        try:
            loop.call_soon_threadsafe(finished)
        except RuntimeError:
            pass  # This loop is closed; the host-wide slot has still been released.


WARM_IDLE_SECONDS = 120


class _JobOutput:
    """One clip's messages on a reusable process channel; no inherited boxes."""
    def __init__(self, channel, token):
        self.channel, self.token = channel, token

    def put(self, record, timeout=.05):
        self.channel.put(dict(token=self.token, record=record), timeout=timeout)

    def close(self):
        pass  # The pool owns the channel, not an individual clip.

    cancel_join_thread = close
    join_thread = close


class _CombinedStop:
    def __init__(self, job_stop, lifetime_stop):
        self.job_stop, self.lifetime_stop = job_stop, lifetime_stop

    def is_set(self):
        return self.job_stop.is_set() or self.lifetime_stop.is_set()


def _warm_host(commands, output, job_stop, lifetime_stop, playback, inputs, provider, idle_seconds):
    """One CPU interpreter; block without CPU work between bounded clip jobs."""
    try:
        started = time.perf_counter()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[key] = "1"
        import cv2
        import numpy as np
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        imports_ms = (time.perf_counter()-started)*1000
        native = time.perf_counter()
        sample = np.zeros((32,32,3), dtype=np.uint8)
        cv2.cvtColor(sample,cv2.COLOR_RGB2LAB)
        cv2.cvtColor(sample,cv2.COLOR_RGB2HSV)
        if provider == "opencv":
            from .tracks import _reference_features
            _reference_features()
            cv2.SIFT_create().detectAndCompute(cv2.cvtColor(sample,cv2.COLOR_RGB2GRAY),None)
        ready = dict(diagnostic_event="warm_ready", provider=provider, worker_pid=os.getpid(),
            ready_at=time.time(), imports_ms=round(imports_ms,3),
            native_initialization_ms=round((time.perf_counter()-native)*1000,3),
            opencv_version=cv2.__version__, opencv_threads=cv2.getNumThreads(), idle_timeout_s=idle_seconds)
        if not _emit(output,dict(token=None,record=ready),lifetime_stop):
            return
        while not lifetime_stop.is_set():
            try:
                command = commands.get(timeout=idle_seconds)
            except queue.Empty:
                break
            if command is None or lifetime_stop.is_set():
                break
            token = command["token"]
            try:
                if command["video"] is None:
                    _frames_worker(command["names"],command["tags"],_JobOutput(output,token),
                        _CombinedStop(job_stop,lifetime_stop),playback,inputs,provider)
                else:
                    _worker(command["video"],command["names"],command["tags"],_JobOutput(output,token),
                        _CombinedStop(job_stop,lifetime_stop),
                        playback if command["live"] else None,provider)
            finally:
                # Retirement must acknowledge native cleanup even after job
                # cancellation. The parent drains results while retaining slot.
                _emit(output,dict(token=token,idle=True),lifetime_stop)
    except Exception as error:
        _emit(output,dict(token=None,record=dict(worker_error=type(error).__name__,
            message=str(error)[:200])),lifetime_stop)
    finally:
        commands.close()
        if lifetime_stop.is_set():
            output.cancel_join_thread()
        output.close()
        if not lifetime_stop.is_set():
            output.join_thread()


class _WarmLease:
    """Existing detector/reaper interface, scoped to one acknowledged job."""
    def __init__(self, owner, video, names, tags, live):
        if owner.closed or not owner.ready or not owner.process.is_alive():
            raise RuntimeError("Local worker is not warm; prepare it before playback.")
        if owner.active_token is not None:
            raise RuntimeError("Previous local job has not retired.")
        owner.sequence += 1
        self.owner, self.token = owner, owner.sequence
        self.command = dict(token=self.token,video=str(video) if video is not None else None,names=list(names),tags=tags,live=live)
        self.idle_seen = False
        self.pid = owner.process.pid

    def start(self):
        self.owner.job_stop.clear()
        with self.owner.playback.get_lock():
            self.owner.playback[:] = [0,0,0,0,0,1]
        self.owner.active_token = self.token
        self.owner.job_retired.clear()
        try:
            self.owner.commands.put_nowait(self.command)
        except BaseException:
            self.close()
            raise

    def is_alive(self):
        return self.owner.process.is_alive()

    def get(self, block=True, timeout=None):
        while True:
            envelope = self.owner.output.get(block=block,timeout=timeout)
            if envelope.get("token") not in (None,self.token):
                return dict(transport_rejection="warm_job_token_mismatch")
            if envelope.get("idle"):
                self.idle_seen = True
                return None
            return envelope["record"]

    def get_nowait(self):
        return self.get(block=False)

    def join(self):
        while not self.idle_seen:
            try:
                self.get(timeout=.1)
            except queue.Empty:
                if not self.is_alive():
                    self.owner.process.join()
                    break

    def close(self):
        if self.owner.active_token == self.token:
            self.owner.active_token = None
            self.owner.job_retired.set()


class WarmLocalWorker:
    """Explicit setup-owned process; never starts implicitly at a decision.

    Await prewarm_local_tracker before initial generation/presentation. Pass
    this object to detect_local for every clip, and await close after cancelling
    and draining session tracking. Idle expiry is bounded and fails closed.
    """
    def __init__(self, provider, idle_seconds):
        if provider not in ("color","opencv"):
            raise ValueError("Warm local provider must be color or opencv")
        if isinstance(idle_seconds,bool) or not 1 <= idle_seconds <= WARM_IDLE_SECONDS:
            raise ValueError("Local warm idle timeout must be in 1..120 seconds")
        self.provider, self.sequence = provider, 0
        self.ready, self.closed, self.active_token = None, False, None
        self._closing_task = None
        self.job_retired = threading.Event()
        self.job_retired.set()
        context = multiprocessing.get_context("spawn")
        self.commands = context.Queue(maxsize=1)
        self.output = context.Queue(maxsize=OUTPUT_CAPACITY)
        self.inputs = context.Queue(maxsize=1)
        self.job_stop, self.lifetime_stop = context.Event(), context.Event()
        self.playback = context.Array("d",[0,0,0,0,0,1])
        self.process = context.Process(target=_warm_host,args=(self.commands,self.output,self.job_stop,
            self.lifetime_stop,self.playback,self.inputs,provider,idle_seconds),daemon=False)

    def request_close(self):
        self.closed = True
        self.lifetime_stop.set()
        self.job_stop.set()
        try:
            self.commands.put_nowait(None)
        except queue.Full:
            pass

    def _dispose(self):
        self.process.join()
        self.job_retired.wait()  # Active lease retirement owns the inference slot.
        self.commands.close()
        # The child has retired. Drain the single pending feeder message before
        # closing so cooperative shutdown cannot wait on unread JPEG bytes.
        try:
            self.inputs.get(timeout=.1)
        except queue.Empty:
            pass
        self.inputs.close()
        self.inputs.join_thread()
        self.output.close()
        self.process.close()

    async def close(self):
        if self._closing_task is None:
            self.request_close()
            self._closing_task = asyncio.create_task(asyncio.to_thread(self._dispose))
        await asyncio.shield(self._closing_task)


async def prewarm_local_tracker(provider="color", *, idle_seconds=WARM_IDLE_SECONDS, on_diagnostic=None):
    """Setup-only warm-up, serialized with inference; no provider/device calls.

    The caller explicitly waits during setup, never at a 3.5s deadline or clip
    transition. Native startup timeout/cancellation retains the host slot until
    actual cooperative exit. The returned process is reused by detect_local.
    """
    loop = asyncio.get_running_loop()
    gate = _gates.setdefault(loop,asyncio.Semaphore(1))
    held, acquired, started, retiring = False, False, False, False
    worker = None
    def diagnostic(event):
        if on_diagnostic:
            try:
                on_diagnostic(dict(event))
            except Exception:
                pass  # An observer cannot bypass cooperative startup cleanup.
    try:
        diagnostic(dict(diagnostic_event="warming",provider=provider))
        async with asyncio.timeout(MAX_WALL_SECONDS):
            await gate.acquire()
            held = True
            while not _worker_slot.acquire(blocking=False):
                await asyncio.sleep(.02)
            acquired = True
            worker = WarmLocalWorker(provider,idle_seconds)
            worker.process.start()
            started = True
            while True:
                try:
                    envelope = worker.output.get_nowait()
                except queue.Empty:
                    if not worker.process.is_alive():
                        raise RuntimeError("Warm local worker exited before readiness.")
                    await asyncio.sleep(.02)
                    continue
                record = envelope["record"]
                if record.get("worker_error"):
                    raise RuntimeError(f"Local warm-up {record['worker_error']}: {record['message']}")
                if record.get("diagnostic_event") == "warm_ready":
                    worker.ready = record
                    diagnostic(record)
                    return worker
    except BaseException:
        diagnostic(dict(diagnostic_event="warm_failed",provider=provider))
        if worker is not None:
            worker.request_close()
            if started:
                # Keep spawn handles and ownership alive until native exit.
                def retire(retirement_worker=worker, retirement_gate=gate):
                    try:
                        retirement_worker._dispose()
                    finally:
                        _worker_slot.release()
                        try:
                            loop.call_soon_threadsafe(retirement_gate.release)
                        except RuntimeError:
                            pass
                retiring = True
                threading.Thread(target=retire,name="opencv-warm-retirement",daemon=True).start()
            else:
                worker.inputs.close()
                worker.commands.close()
                worker.output.close()
                worker.process.close()
        raise
    finally:
        if not retiring:
            if acquired:
                _worker_slot.release()
            if held:
                gate.release()


async def detect_local(video, names, *, clip_id, session_id, generation_id=None, on_progress=None, playback_state=None, provider="opencv", diagnostics_directory=None, warm_worker=None, frame_source=None):
    """Offline coverage within a wall budget or playback stale-work skipping.

    Playback/generation never await tracking. Cancellation signals stop between
    native calls and returns within one second; a stuck native call retains the
    worker slot until it exits. No process termination is authorized or attempted.
    """
    from . import tracking_diagnostics
    journal = tracking_diagnostics.for_video(video, session_id, clip_id, generation_id, diagnostics_directory) if video is not None or diagnostics_directory else None
    def diagnostic(kind, **fields):
        if journal:
            journal.record(kind, **fields)
    if frame_source is not None and warm_worker is None:
        raise ValueError("Presented frames require an already warm worker.")
    diagnostic("job_requested", provider=provider)
    queued = time.perf_counter()
    loop = asyncio.get_running_loop()
    gate = _gates.setdefault(loop, asyncio.Semaphore(1))
    try:
        await gate.acquire()
    except asyncio.CancelledError:
        diagnostic("job_cancelled", stage="waiting_loop_slot", records=0)
        raise
    try:
        while not _worker_slot.acquire(blocking=False):
            await asyncio.sleep(.02)
    except BaseException:
        diagnostic("job_cancelled", stage="waiting_host_slot", records=0)
        gate.release()
        raise
    diagnostic("worker_slot_acquired", queueWaitMs=round((time.perf_counter()-queued)*1000,3))
    output = None
    try:
        if warm_worker is None:
            context = multiprocessing.get_context("spawn")
            output = context.Queue(maxsize=OUTPUT_CAPACITY)
            stop = context.Event()
            playback = context.Array("d", [0, 0, 0, 0, 0, 1]) if playback_state else None
            process = context.Process(target=_worker, args=(str(video),list(names),
                dict(clip_id=clip_id,session_id=session_id,generation_id=generation_id),output,stop,playback,provider),daemon=True)
        else:
            if warm_worker.provider != provider:
                raise ValueError("Local warm provider does not match this clip.")
            process = _WarmLease(warm_worker,video,names,
                dict(clip_id=clip_id,session_id=session_id,generation_id=generation_id),bool(playback_state))
            output, stop = process, warm_worker.job_stop
            playback = warm_worker.playback if playback_state else None
        process.start()
        diagnostic("worker_reused" if warm_worker is not None else "worker_spawned",workerPid=process.pid,
            warmReadyAt=warm_worker.ready.get("ready_at") if warm_worker is not None else None)
    except BaseException:
        if output is not None:
            output.close()
        gate.release()
        _worker_slot.release()
        raise
    result = []
    input_queued = 0
    def receive(record):
        if record is None:
            diagnostic("worker_complete", records=len(result))
            return False
        if "transport_rejection" in record:
            diagnostic("transport_rejected",reason=record["transport_rejection"])
            return True
        if "diagnostic_event" in record:
            diagnostic(record["diagnostic_event"], **{k:v for k,v in record.items() if k!="diagnostic_event"})
            return True
        if "worker_error" in record:
            diagnostic("worker_failed", error=record["worker_error"], message=record.get("message"))
            raise RuntimeError(f"OpenCV worker {record['worker_error']}: {record['message']}")
        if any(record.get(key) != value for key,value in
                (("clip_id",clip_id),("session_id",session_id),("generation_id",generation_id))):
            diagnostic("transport_rejected", reason="provenance_mismatch", mediaTime=record.get("t"))
            return True
        if frame_source is not None:
            state = playback_state()
            if not state or not state.get("current") or not state.get("playing") or record.get("playback_epoch") != state.get("epoch",0):
                diagnostic("transport_rejected",reason="playback_epoch_mismatch")
                return True
        record["received_at"] = time.time()
        produced = record.get("produced_at")
        record["transport"] = dict(status="accepted", received_at=record["received_at"],
            delay_ms=round(max(0,record["received_at"]-produced)*1000,3) if produced is not None else None)
        diagnostic("frame_received", **tracking_diagnostics.frame_evidence(record,
            playback_state() if playback_state else None))
        result.append(record)
        return True
    try:
        async with asyncio.timeout(MAX_SECONDS if frame_source else MAX_WALL_SECONDS):
            while True:
                if playback is not None:
                    state = playback_state()
                    with playback.get_lock():
                        playback[:] = ([state["media_t"], state["at"], state.get("playing", False),
                            state.get("current", True), state.get("epoch", 0), state.get("rate", 1)]
                            if state else [0, 0, 0, 0, 0, 1])
                if frame_source is not None:
                    item = frame_source()
                    if item is not None:
                        try:
                            warm_worker.inputs.put_nowait(item)
                            input_queued += 1
                            if input_queued & (input_queued-1) == 0:
                                diagnostic("presented_input_queued",count=input_queued,mediaTime=item["t"],epoch=item["epoch"])
                        except queue.Full:
                            # Replace the older pending input when its feeder
                            # has delivered it. Never block on a feeder race.
                            try:
                                warm_worker.inputs.get_nowait()
                                warm_worker.inputs.put_nowait(item)
                            except (queue.Empty,queue.Full):
                                pass
                            diagnostic("input_dropped",reason="pending_frame")
                received = False
                for _ in range(OUTPUT_CAPACITY):
                    try:
                        record = output.get_nowait()
                    except queue.Empty:
                        break
                    if not receive(record):
                        if on_progress:
                            on_progress(list(result))
                        return result
                    received = True
                if received and on_progress:
                    on_progress(list(result))
                # Queue feeder can deliver completion just after process exit.
                if not process.is_alive() and not received:
                    try:
                        final = await asyncio.to_thread(output.get, True, .1)
                    except queue.Empty:
                        raise RuntimeError("OpenCV worker exited without a completion record.")
                    if not receive(final):
                        if on_progress:
                            on_progress(list(result))
                        return result
                await asyncio.sleep(.02)
    except asyncio.CancelledError:
        diagnostic("job_cancelled", records=len(result))
        raise
    except Exception as error:
        diagnostic("job_failed", error=type(error).__name__)
        raise
    finally:
        diagnostic("job_retiring", records=len(result))
        if journal:
            # Append runs on the journal's background writer, never in inference.
            try:
                await asyncio.wait_for(asyncio.shield(journal.flush()), .2)
            except TimeoutError:
                pass
        stop.set()
        # Retirement owns the slot even if a native call outlives cancellation.
        cleanup = loop.create_future()
        threading.Thread(target=_retire, args=(process, output, gate, loop, cleanup, (stop, playback, warm_worker)),
            name="opencv-cooperative-reaper", daemon=True).start()
        try:
            await asyncio.wait_for(asyncio.shield(cleanup), 1)
        except TimeoutError:
            pass
