"""Run Gazekit's existing calibration/streaming in a camera-owning child."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import urllib.request
import socket
import subprocess
from types import SimpleNamespace

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def connected_cameras(camera_module):
    """Keep Continuity devices visible even before their first frame arrives."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["/usr/sbin/system_profiler", "SPCameraDataType", "-json"],
                                    capture_output=True, text=True, timeout=15)
            devices = sorted(json.loads(result.stdout).get("SPCameraDataType", []),
                             key=lambda d: d.get("spcamera_unique-id", ""))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            devices = []
        if devices:
            return [dict(index=i, name=d.get("_name", f"Camera {i}"),
                         deviceId=d.get("spcamera_unique-id")) for i, d in enumerate(devices)]
    return camera_module.list_cameras(max_index=5)


def choose_camera(ui, cv2, camera_module, screen_size):
    """Choose a real external camera through Gazekit's native UI."""
    win = ui.FullscreenWindow("gazekit-camera", screen_size())
    choices = []
    selected = None
    def scan():
        nonlocal choices
        img = win.canvas()
        ui.center_text(img, "Searching for your iPhone camera...", int(win.h * .4), .9)
        win.show(img)
        choices = sorted(connected_cameras(camera_module),
                         key=lambda c: "iphone" not in c['name'].lower())
    def clicked(event, x, y, flags, parameter):
        nonlocal selected
        if event == cv2.EVENT_LBUTTONDOWN:
            row = int((y - win.h * .32 + 24) // 46)
            if 0 <= row < len(choices) and abs(y - (win.h * .32 + row * 46)) < 24:
                selected = str(choices[row]['index'])
    cv2.setMouseCallback(win.name, clicked)
    try:
        scan()
        print("GOZ_CAMERA_WINDOW ready", flush=True)
        while True:
            img = win.canvas()
            ui.center_text(img, "SELECT YOUR GAZEKIT CAMERA", int(win.h * .18), 1.0)
            for i, camera in enumerate(choices):
                ui.center_text(img, f"{i + 1}. {camera['name']}", int(win.h * .32 + i * 46), .8,
                               ui.ACCENT if "iphone" in camera['name'].lower() else ui.WHITE)
            ui.center_text(img, "Click a camera or press its number; Enter selects iPhone", int(win.h * .65), .65)
            ui.center_text(img, "R: refresh cameras - Q / Esc: cancel", int(win.h * .71), .65)
            ui.center_text(img, "Continuity Camera: keep iPhone locked, rear cameras facing you", int(win.h * .79), .6)
            if not any("iphone" in c['name'].lower() for c in choices):
                ui.center_text(img, "iPhone not found: enable Continuity Camera or connect USB, then R", int(win.h * .86), .6)
            key = win.show(img)
            if key in (27, ord('q')):
                raise ValueError("Camera selection cancelled. Retry sensor setup to open Gazekit again.")
            if key in (ord('c'), ord('r')):
                scan()
            elif key in (10, 13):
                selected = next((str(c['index']) for c in choices if "iphone" in c['name'].lower()), None)
            elif ord('1') <= key < ord('1') + len(choices):
                selected = str(choices[key - ord('1')]['index'])
            if selected is not None:
                return selected
    finally:
        win.close()
        cv2.waitKey(1)


def configure_camera(camera, model, dataset):
    camera_config = model.parent / "camera.json"
    camera_config.write_text(json.dumps({"camera": str(camera)}))
    dataset.CONFIG_PATH = camera_config
    os.chdir(model.parent)


def saved_camera(cameras, device_id, name):
    matches = ([c for c in cameras if c.get("deviceId") == device_id] if device_id
               else [c for c in cameras if c.get("name") == name])
    if len(matches) != 1:
        raise ValueError("The saved eye calibration's camera is unavailable. Reconnect it, or remove eye calibration to choose another camera.")
    return str(matches[0]["index"])


def configure_alignment(stream, model, reuse):
    """Persist the first alignment, then apply it without a target window."""
    path = model.parent / "gaze_alignment.json"
    if reuse:
        try:
            values = json.loads(path.read_text())["coefficients"] if path.exists() else [1., 0., 1., 0.]
            if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Invalid alignment")
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("Saved gaze alignment is damaged. Remove eye calibration to train again.") from None
        ax, bx, ay, by = values
        original = stream.build_predictor
        def aligned_predictor(*args, **kwargs):
            predict, ridge, active = original(*args, **kwargs)
            def predict_saved(observation):
                point = predict(observation)
                return None if point is None else (ax * float(point[0]) + bx, ay * float(point[1]) + by)
            return predict_saved, ridge, active
        stream.build_predictor = aligned_predictor
        return False
    original = stream._quick_align
    def save_alignment(*args, **kwargs):
        values = original(*args, **kwargs)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"coefficients": [float(v) for v in values]}))
        temporary.chmod(0o600)
        temporary.replace(path)
        return values
    stream._quick_align = save_alignment
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "stream"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--camera", default="select")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report")
    parser.add_argument("--port", type=int, default=5590)
    parser.add_argument("--setup-id")
    parser.add_argument("--camera-device-id")
    parser.add_argument("--camera-name")
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()
    repo, model = Path(args.repo).resolve(), Path(args.model).resolve()
    sys.path.insert(0, str(repo))
    # Relative Gazekit assets/config always resolve to its checkout; calibration
    # recordings and trained models stay inside this GOZ attempt's data folder.
    os.chdir(repo)
    if args.command == "stream" and (args.camera_device_id or args.camera_name):
        from gazekit import camera as camera_module
        args.camera = saved_camera(connected_cameras(camera_module), args.camera_device_id, args.camera_name)
    if args.camera == "select":
        import cv2
        from gazekit import ui, camera as camera_module
        from gazekit.screen import screen_size
        args.camera = choose_camera(ui, cv2, camera_module, screen_size)
        print("GOZ_CAMERA " + json.dumps({"camera": args.camera}), flush=True)
    elif args.command == "stream":
        print("GOZ_CAMERA " + json.dumps({"camera": args.camera}), flush=True)
    landmarker = model.parent.parent / "face_landmarker.task"
    if not landmarker.is_file():
        existing = repo / "models" / "face_landmarker.task"
        if existing.is_file():
            landmarker = existing
        else:
            print("Downloading the official MediaPipe face landmark model…", flush=True)
            with urllib.request.urlopen(LANDMARKER_URL, timeout=60) as response:
                data = response.read(10 * 1024 * 1024 + 1)
            if not data or len(data) > 10 * 1024 * 1024:
                raise ValueError("Face landmark model download was invalid.")
            temporary = landmarker.with_suffix(".download")
            temporary.write_bytes(data)
            temporary.replace(landmarker)
    if not args.camera.isdigit():
        raise ValueError("Select a connected external camera, such as iPhone Continuity Camera.")
    camera = int(args.camera)
    from gazekit import dataset
    configure_camera(camera, model, dataset)
    if args.command == "calibrate":
        from gazekit.calibrate import run
        from gazekit import camera as camera_module
        camera_info = next((c for c in connected_cameras(camera_module) if str(c["index"]) == args.camera), {})
        report = run(camera_index=camera, model_out=str(model), dataset_root=str(model.parent / "dataset"), landmarker=str(landmarker))
        if not report or report.get("verdict") not in ("STABLE", "USABLE") or not model.is_file():
            raise ValueError("Gazekit calibration was cancelled or did not pass validation.")
        report.update(camera=args.camera, cameraName=camera_info.get("name"), cameraDeviceId=camera_info.get("deviceId"))
        Path(args.report).write_text(json.dumps(report, indent=2))
        Path(args.report).chmod(0o600)
    else:
        import gazekit.stream as stream
        # Additive UDP metadata binds samples to this validated calibration.
        # Adapt only this module's sender; Gazekit's camera sockets are unchanged.
        class TaggedSocket:
            def __init__(self, *values): self.sock = socket.socket(*values)
            def sendto(self, data, destination):
                sample = json.loads(data)
                sample["setupId"] = args.setup_id
                return self.sock.sendto(json.dumps(sample).encode(), destination)
            def close(self): self.sock.close()
        stream.socket = SimpleNamespace(socket=TaggedSocket, AF_INET=socket.AF_INET, SOCK_DGRAM=socket.SOCK_DGRAM)
        align = configure_alignment(stream, model, args.reuse_calibration)
        stream.run(camera_index=camera, model_path=str(model), port=args.port, align=align, landmarker=str(landmarker))


if __name__ == "__main__":
    main()
