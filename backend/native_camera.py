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
                detail = self.process.stderr.read(4096).decode(errors="replace").strip()
                raise ValueError(detail or "Selected camera failed to open; no fallback was attempted.")
            self.packets.put(json.loads(line))
            self.ready.wait()
            import cv2
            import numpy as np
            while not self.closed:
                width, height, length = struct.unpack("<III", self._read_exact(12))
                if not (0 < width <= 4096 and 0 < height <= 4096 and length == width * height * 4):
                    raise ValueError("Native camera returned an invalid frame packet.")
                pixels = np.frombuffer(self._read_exact(length), dtype=np.uint8).reshape(height, width, 4)
                frame = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
                self.last_frame = time.monotonic()
                try: self.packets.get_nowait()
                except queue.Empty: pass
                self.packets.put_nowait(frame)
        except Exception as error:
            self.error = str(error)

    def read(self):
        if self.error: raise ValueError(self.error)
        if self.closed: return False, None
        try:
            return True, self.packets.get(timeout=2)
        except queue.Empty:
            if self.error: raise ValueError(self.error)
            if time.monotonic() - self.last_frame > 10:
                raise ValueError("Selected camera delivered no frames for 10 seconds. Reconnect it and retry; no fallback was attempted.")
            return False, None

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
