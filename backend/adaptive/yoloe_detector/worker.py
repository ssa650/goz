"""One cooperative worker, bounded queue and media length; no process killing.

Cancellation requests stop between frames. A hung native model call cannot be
hard-preempted under the task's no-process-kill constraint: the global slot stays
occupied until that worker exits. The scheduler must treat this as experimental.
"""
import asyncio
import multiprocessing
import os
import queue
import threading
import time
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import asdict

from .config import DetectorConfig

_slot = threading.Lock()


def _put(output, stop, item):
    while not stop.is_set():
        try:
            output.put(item, timeout=.1)
            return True
        except queue.Full:
            continue
    return False


def _worker(video, names, tags, config, output, stop):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    # Disable network/autoinstall behavior before importing vendor runtime.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    reader = None
    started = time.monotonic()
    try:
        import imageio_ffmpeg
        import numpy as np
        from .detector import YOLOECharacterDetector
        if stop.is_set():
            return
        detector = YOLOECharacterDetector(names, config)
        reader = imageio_ffmpeg.read_frames(video, input_params=["-threads", "1"],
            output_params=["-threads", "1", "-vf", f"fps={config.fps},scale={config.image_size}:-2",
                           "-t", str(config.max_seconds)])
        metadata = next(reader)
        width, height = metadata["size"]
        if height > 1280:
            raise ValueError("Extreme portrait frame rejected")
        for index, raw in enumerate(reader):
            if stop.is_set() or time.monotonic()-started >= config.max_wall_seconds or index >= config.max_seconds*config.fps:
                break
            pixels = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            record = detector.step(pixels, index/config.fps)
            record.update(tags)
            record["resources"].update(queue_capacity=config.queue_size, decode_threads=1)
            if not _put(output, stop, record):
                return
        _put(output, stop, dict(worker_done=True, elapsed_seconds=time.monotonic()-started,
                               budget_exhausted=time.monotonic()-started >= config.max_wall_seconds))
    except Exception as error:
        # Class only: paths/provider exceptions can contain unexpected secrets.
        _put(output, stop, dict(worker_error=type(error).__name__))
    finally:
        if reader is not None:
            reader.close()
        if stop.is_set():
            output.cancel_join_thread()
        output.close()
        if not stop.is_set():
            output.join_thread()


def _reap(process, output, stop):
    try:
        process.join()
        output.close()
        process.close()
    finally:
        _slot.release()


async def detect_yoloe(video, names, *, clip_id, session_id, generation_id=None, on_progress=None, config=None):
    """Scheduler adapter. Progress is cumulative and tagged, same as detect_local.

    Returns records on completion. Timeout/cancel publishes no later callbacks,
    requests cooperative stop, and keeps the single global slot until actual exit.
    Configuration must be explicitly license-reviewed and checksum-verified.
    """
    config = config or DetectorConfig()
    config.validated_weights()
    if config.runtime_python:
        return await _detect_isolated(video, names, clip_id=clip_id, session_id=session_id,
            generation_id=generation_id, on_progress=on_progress, config=config)
    if not _slot.acquire(blocking=False):
        raise RuntimeError("YOLOE worker busy (or finishing cooperative cancellation)")
    context = multiprocessing.get_context("spawn")
    output, stop = context.Queue(maxsize=config.queue_size), context.Event()
    process = context.Process(target=_worker, args=(str(video), list(names),
        dict(clip_id=clip_id, session_id=session_id, generation_id=generation_id), config, output, stop), daemon=True)
    started = False
    try:
        process.start()
        started = True
        result = []
        async with asyncio.timeout(config.max_wall_seconds):
            while True:
                received = False
                for _ in range(config.queue_size):
                    try:
                        record = output.get_nowait()
                    except queue.Empty:
                        break
                    if record.get("worker_done"):
                        if record.get("budget_exhausted"):
                            raise TimeoutError("YOLOE worker wall budget exhausted")
                        if on_progress:
                            on_progress(list(result))
                        return result
                    if "worker_error" in record:
                        raise RuntimeError(f"YOLOE worker failed ({record['worker_error']})")
                    result.append(record)
                    received = True
                if received and on_progress:
                    on_progress(list(result))
                if not process.is_alive() and not received:
                    # Queue feeder may deliver its final record after process exit.
                    await asyncio.sleep(.05)
                    try:
                        record = output.get_nowait()
                    except queue.Empty:
                        raise RuntimeError("YOLOE worker exited without completion") from None
                    if record.get("worker_done"):
                        if record.get("budget_exhausted"):
                            raise TimeoutError("YOLOE worker wall budget exhausted")
                        if on_progress:
                            on_progress(list(result))
                        return result
                    if "worker_error" in record:
                        raise RuntimeError(f"YOLOE worker failed ({record['worker_error']})")
                    result.append(record)
                await asyncio.sleep(.02)
    finally:
        stop.set()
        if started:
            # Keep Event/SemLock alive until spawned child has finished unpickling.
            threading.Thread(target=_reap, args=(process, output, stop), daemon=True,
                             name="yoloe-cooperative-reaper").start()
        else:
            output.close()
            process.close()
            _slot.release()


def _read_lines(process, records, stop):
    try:
        for line in process.stdout:
            if len(line) > 131072:
                item = dict(worker_error="OversizeRecord")
            else:
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    item = dict(worker_error="MalformedWorkerRecord")
            # After cancellation continue draining so the worker can exit without
            # a full stdout pipe blocking it. Never publish these later records.
            while not stop.is_set():
                try:
                    records.put(item, timeout=.05)
                    break
                except queue.Full:
                    continue
    finally:
        while not stop.is_set():
            try:
                records.put(dict(worker_eof=True), timeout=.05)
                break
            except queue.Full:
                continue


def _reap_isolated(process):
    try:
        process.wait()
        process.stdout.close()
    finally:
        _slot.release()


async def _detect_isolated(video, names, *, clip_id, session_id, generation_id, on_progress, config):
    executable = Path(config.runtime_python).expanduser().absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Isolated runtime_python must be an existing local executable")
    if not _slot.acquire(blocking=False):
        raise RuntimeError("YOLOE worker busy (or finishing cooperative cancellation)")
    process = None
    stop = threading.Event()
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1",
               YOLO_CONFIG_DIR=str(Path(config.weights).expanduser().absolute().parent / "ultralytics-config"),
               YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    try:
        process = subprocess.Popen([str(executable), "-m", "backend.adaptive.yoloe_detector.worker"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, bufsize=1, cwd=str(root), env=env)
        process.stdin.write(json.dumps(dict(video=str(video), names=list(names), tags=dict(
            clip_id=clip_id, session_id=session_id, generation_id=generation_id), config=asdict(config)))+"\n")
        process.stdin.flush()
        records = queue.Queue(maxsize=config.queue_size)
        threading.Thread(target=_read_lines, args=(process, records, stop), daemon=True,
                         name="yoloe-isolated-reader").start()
        result = []
        async with asyncio.timeout(config.max_wall_seconds):
            while True:
                received = False
                for _ in range(config.queue_size):
                    try:
                        record = records.get_nowait()
                    except queue.Empty:
                        break
                    if "worker_error" in record:
                        raise RuntimeError(f"YOLOE worker failed ({record['worker_error']})")
                    if record.get("worker_done"):
                        if record.get("budget_exhausted"):
                            raise TimeoutError("YOLOE worker wall budget exhausted")
                        if on_progress:
                            on_progress(list(result))
                        return result
                    if record.get("worker_eof"):
                        raise RuntimeError("YOLOE worker exited without completion")
                    result.append(record)
                    received = True
                if received and on_progress:
                    on_progress(list(result))
                await asyncio.sleep(.02)
    finally:
        stop.set()
        if process is not None:
            # EOF on stdin is the stop request; no terminate/kill or signal.
            try:
                process.stdin.close()
            except OSError:
                pass
            threading.Thread(target=_reap_isolated, args=(process,), daemon=True,
                             name="yoloe-isolated-reaper").start()
        else:
            _slot.release()


def _main():
    payload = json.loads(sys.stdin.readline())
    stop = threading.Event()
    def read_stop():
        sys.stdin.read()
        stop.set()
    threading.Thread(target=read_stop, daemon=True).start()
    protocol_output = sys.stdout
    # Keep third-party log output out of the JSON-lines protocol.
    sys.stdout = sys.stderr
    class Output:
        def put(self, value, timeout=None):
            protocol_output.write(json.dumps(value, allow_nan=False)+"\n")
            protocol_output.flush()
        def close(self): pass
        def cancel_join_thread(self): pass
        def join_thread(self): pass
    _worker(payload["video"], payload["names"], payload["tags"], DetectorConfig(**payload["config"]), Output(), stop)


if __name__ == "__main__":
    _main()
