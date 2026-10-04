"""Gazekit-compatible capture bound to an AVFoundation UID, never an index.

The helper's list command only enumerates devices. Capture runs exclusively when
the user starts sensor setup. No camera probing, OpenCV fallback, or PyObjC install.
"""
from functools import lru_cache
import json
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading
import time
import math


@lru_cache(maxsize=1)
def helper_path():
    root = Path(__file__).resolve().parent
    directory = tempfile.TemporaryDirectory(prefix="goz-camera-")
    binary = Path(directory.name) / "native-camera"
    result = subprocess.run([
        "/usr/bin/swiftc", "-swift-version", "5", str(root / "native_camera.swift"),
        "-module-cache-path", str(Path(directory.name) / "modules"),
        "-o", str(binary), "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
        "-Xlinker", "__info_plist", "-Xlinker", str(root / "native_camera.plist"),
    ], capture_output=True, text=True, timeout=120)
    if result.returncode:
        directory.cleanup()
        raise ValueError("Native camera helper could not compile. Install Apple's command line tools; "
                         "no camera was opened. " + result.stderr[-1000:])
    # Keep the private directory alive for this worker's lifetime.
    helper_path.directory = directory
    return str(binary)


def list_cameras():
    result = subprocess.run([helper_path(), "list"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError("Native camera enumeration failed; no camera was opened. " + result.stderr[-700:])
    devices = json.loads(result.stdout)
    if not isinstance(devices, list) or any(not c.get("deviceId") or not c.get("name") for c in devices):
        raise ValueError("Native camera enumeration returned invalid device identities.")
    return devices


def verify_identity(selected, opened):
    if (not selected.get("deviceId") or opened.get("deviceId") != selected["deviceId"]
            or opened.get("verified") is not True or opened.get("backend") != "avfoundation-uid"
            or not opened.get("name")):
        raise ValueError("Opened camera identity does not match the selection. Previous calibration "
                         "is preserved. Choose the intended camera and run full eye recalibration.")
    return dict(opened)


class NativeCapture:
    def __init__(self, selected, on_wait=None):
        self.process = subprocess.Popen([helper_path(), "capture", selected["deviceId"]],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.packets = queue.Queue(maxsize=1)
        self.closed = False
        self.error = None
        self.ready = threading.Event()
        self.last_frame = time.monotonic()
        self.last_metadata = None
        self._stats_lock = threading.Lock()
        self._stats = dict(received=0, replaced=0, consumed=0, readTimeouts=0,
                           nativeSequenceGaps=0, lastSequence=0, lastReceivedAt=None,
                           transportMsMax=0., conversionMsMax=0.)
        self._stderr_tail = ""
        self._stderr_thread = threading.Thread(target=self._receive_diagnostics, daemon=True)
        self._stderr_thread.start()
        threading.Thread(target=self._receive, daemon=True).start()
        started = time.monotonic()
        try:
            while True:
                try:
                    opened = self.packets.get(timeout=.1)
                    break
                except queue.Empty:
                    if self.error: raise ValueError(self.error)
                    if time.monotonic() - started > 30:
                        raise ValueError("Selected camera did not confirm its identity within 30 seconds. Reconnect it and retry.")
                    if on_wait: on_wait(time.monotonic() - started)
            self.identity = verify_identity(selected, opened)
            if opened.get("frameProtocol") != 2:
                raise ValueError("Native camera frame protocol mismatch; no fallback was attempted.")
            self.last_frame = time.monotonic()
            # Let the receiver begin frames only after the handshake is checked.
            self.ready.set()
        except BaseException:
            self.release()
            raise

    def _read_exact(self, length):
        chunks = bytearray()
        while len(chunks) < length:
            part = self.process.stdout.read(length - len(chunks))
            if not part: raise ValueError("Selected camera disconnected or stopped.")
            chunks.extend(part)
        return chunks

    def _receive(self):
        try:
            line = self.process.stdout.readline(4096)
            if not line:
                self._stderr_thread.join(timeout=.2)
                detail = self._stderr_tail
                raise ValueError(detail or "Selected camera failed to open; no fallback was attempted.")
            self.packets.put(json.loads(line))
            self.ready.wait()
            while not self.closed:
                metadata_length, = struct.unpack("<I", self._read_exact(4))
                if not 0 < metadata_length <= 4096:
                    raise ValueError("Native camera returned invalid frame metadata length.")
                metadata = json.loads(self._read_exact(metadata_length))
                width, height, length = (metadata.get(k) for k in ("width", "height", "length"))
                if any(type(v) is not int for v in (width, height, length)):
                    raise ValueError("Native camera returned invalid frame dimensions.")
                if not (0 < width <= 4096 and 0 < height <= 4096 and length == width * height * 4):
                    raise ValueError("Native camera returned an invalid frame packet.")
                if any(type(metadata.get(k)) not in (int, float) or not math.isfinite(metadata[k])
                       for k in ("capturedAt", "capturedMonotonic", "deliveredAt", "deliveredMonotonic")):
                    raise ValueError("Native camera returned invalid frame timestamps.")
                sequence = metadata.get("sequence")
                if type(sequence) is not int or sequence <= 0:
                    raise ValueError("Native camera returned invalid frame sequence.")
                pixels = self._read_exact(length)
                self.last_frame = time.monotonic()
                metadata.update(receivedAt=time.time(), receivedMonotonic=self.last_frame)
                with self._stats_lock:
                    if sequence <= self._stats["lastSequence"]:
                        raise ValueError("Native camera frame sequence did not advance.")
                    self._stats["nativeSequenceGaps"] += sequence - self._stats["lastSequence"] - 1
                    self._stats.update(received=self._stats["received"] + 1,
                                       lastSequence=sequence, lastReceivedAt=metadata["receivedAt"],
                                       transportMsMax=max(self._stats["transportMsMax"],
                                           (self.last_frame - metadata["deliveredMonotonic"]) * 1000))
                    self._stats["packetNativeCounters"] = {key: metadata.get(key) for key in
                        ("nativeDelivered", "nativeDropped", "nativeReplaced", "previousWriteMs")}
                    try:
                        self.packets.get_nowait()
                        self._stats["replaced"] += 1
                    except queue.Empty:
                        pass
                    self.packets.put_nowait((pixels, metadata))
        except Exception as error:
            if not self.closed: self.error = str(error)

    def _receive_diagnostics(self):
        try:
            while not self.closed:
                line = self.process.stderr.readline(4096).decode(errors="replace").strip()
                if not line: break
                if line.startswith("GOZ_NATIVE_DIAGNOSTICS "):
                    values = json.loads(line[len("GOZ_NATIVE_DIAGNOSTICS "):])
                    # Keep a fixed schema; never accumulate stderr text/history.
                    with self._stats_lock:
                        self._stats["nativeHeartbeat"] = {key: values.get(key) for key in
                            ("nativeDelivered", "nativeDropped", "nativeReplaced", "nativePending",
                             "nativeCallbackAgeS", "nativeWriteAgeS", "previousWriteMs", "nativeHeartbeatAt")}
                else:
                    self._stderr_tail = line[-700:]
        except (OSError, ValueError, TypeError):
            if not self.closed: self._stderr_tail = "Native diagnostic channel stopped."

    def read(self):
        ok, frame, metadata = self.read_timed()
        self.last_metadata = metadata
        return ok, frame

    def read_timed(self):
        """Convert only the latest consumed frame; the receiver just drains bytes."""
        if self.error: raise ValueError(self.error)
        if self.closed: return False, None, None
        try:
            pixels, metadata = self.packets.get(timeout=.25)
        except queue.Empty:
            with self._stats_lock: self._stats["readTimeouts"] += 1
            if self.error: raise ValueError(self.error)
            if time.monotonic() - self.last_frame > 10:
                raise ValueError("Selected camera delivered no frames for 10 seconds. Reconnect it and retry; no fallback was attempted.")
            return False, None, None
        import cv2
        import numpy as np
        started = time.monotonic()
        pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(metadata["height"], metadata["width"], 4)
        frame = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        converted = time.monotonic()
        with self._stats_lock:
            self._stats["consumed"] += 1
            self._stats["conversionMsMax"] = max(self._stats["conversionMsMax"], (converted-started)*1000)
        return True, frame, dict(metadata, readAt=time.time(), readMonotonic=converted)

    def diagnostics(self):
        with self._stats_lock:
            heartbeat = self._stats.get("nativeHeartbeat")
            stamp = heartbeat.get("nativeHeartbeatAt") if heartbeat else None
            age = max(0., time.time()-stamp) if type(stamp) in (float, int) else None
            return dict(self._stats, nativeHeartbeatAgeS=age, pending=self.packets.qsize(), queueCapacity=1,
                        frameAgeSeconds=time.monotonic()-self.last_frame, error=self.error)

    def release(self):
        if self.closed: return
        self.closed = True
        self.ready.set()
        self.process.stdin.close()
        try: self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=3)
        for stream in (self.process.stdout, self.process.stderr): stream.close()

    def isOpened(self):
        return not self.closed and self.process.poll() is None
