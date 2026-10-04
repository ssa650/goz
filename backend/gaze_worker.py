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
import time
import hashlib
import importlib.metadata

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
CALIBRATION_SCHEMA = 2


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        os.chmod(temporary, 0o600)
        json.dump(value, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def model_schema(repo):
    """A changed feature extractor or regression format requires a new check."""
    digest = hashlib.sha256()
    for name in ("model.py", "tracker.py"):
        digest.update((Path(repo) / "gazekit" / name).read_bytes())
    digest.update(importlib.metadata.version("scikit-learn").encode())
    return digest.hexdigest()


def validate_compatibility(report, repo, profile, screen):
    if report.get("schemaVersion") != CALIBRATION_SCHEMA:
        raise ValueError("Legacy calibration has no verified viewer/camera/schema metadata. Preserve it and run a fresh calibration/check for this viewer.")
    if report.get("profileId") != profile:
        raise ValueError("Saved calibration belongs to another viewer profile. Select the matching GOZ_VIEWER_PROFILE or calibrate this viewer.")
    if report.get("modelSchema") != model_schema(repo):
        raise ValueError("Gazekit feature/model schema changed. Recalibrate before using this saved model.")
    if list(report.get("screen", [])) != list(screen):
        raise ValueError(f"Display geometry changed: saved {report.get('screen')}, current {list(screen)} screen points. Restore the display or recalibrate.")
    if not report.get("cameraDeviceId") and not report.get("cameraName"):
        raise ValueError("Saved calibration has no camera identity. Recalibrate with the intended camera.")


def alignment_values(model):
    path = model.parent / "gaze_alignment.json"
    if not path.exists():
        return (1., 0., 1., 0.)
    try:
        values = json.loads(path.read_text())["coefficients"]
        if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Invalid coefficients")
        return tuple(values)
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Saved gaze alignment is damaged. Run the supported alignment check or recalibrate.") from None


def load_predictor(model, screen):
    from gazekit.model import GazeModel, transform
    ridge = GazeModel.load(model)
    if tuple(ridge.screen_size) != tuple(screen):
        raise ValueError("Saved model display geometry does not match its metadata/current display.")
    # Gazekit clips predictions to screen edges. Preserve actual predictions:
    # outside-screen gaze must never be reassigned to a character at the edge.
    def predict(observation):
        if not observation.ok:
            return None
        point = ridge.pipe.predict(transform(observation.features))[0] + ridge.bias
        return point if all(math.isfinite(float(v)) for v in point) else None
    return predict, ridge


def stream_gaze(args, model, landmarker, screen):
    """Use Gazekit's tracker/filter with explicit capture time and no clamping."""
    import cv2
    from gazekit.camera import open_camera, read_mirrored
    from gazekit.tracker import FaceTracker
    from gazekit.filters import GazeSmoother
    from gazekit.live import BlinkGate
    predict, _ = load_predictor(model, screen)
    ax, bx, ay, by = alignment_values(model)
    cap = open_camera(int(args.camera))
    tracker = FaceTracker(str(landmarker))
    smoother, gate = GazeSmoother(), BlinkGate()
    started = last_frame = time.monotonic()
    sent = valid = 0
    last_xy = (screen[0] / 2, screen[1] / 2)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            while not args.seconds or time.monotonic() - started < args.seconds:
                frame = read_mirrored(cap)
                captured = time.time()  # Camera-read completion, before inference.
                if frame is None:
                    if time.monotonic() - last_frame > 10:
                        raise ValueError("Camera opened but delivered no frames for 10 seconds. Reconnect Continuity Camera or check macOS Camera permission.")
                    time.sleep(.01)
                    continue
                last_frame = time.monotonic()
                obs = tracker.process(frame)
                frozen = gate.update(obs)
                point = None if frozen else predict(obs)
                if point is not None:
                    last_xy = smoother.apply(ax * float(point[0]) + bx, ay * float(point[1]) + by, last_frame)
                sample = dict(t=captured, capturedAt=captured, sentAt=time.time(),
                              captureClock="unix-seconds", captureTiming="camera-read-complete",
                              coordinateSpace="screen-points", screenOrigin=[0, 0],
                              x=round(last_xy[0], 2), y=round(last_xy[1], 2), sw=screen[0], sh=screen[1],
                              valid=point is not None, face=bool(obs.ok), blink=bool(obs.ok and frozen),
                              yaw=float(obs.yaw), pitch=float(obs.pitch), setupId=args.setup_id,
                              profileId=args.profile)
                sender.sendto(json.dumps(sample, allow_nan=False).encode(), ("127.0.0.1", args.port))
                sent += 1
                valid += point is not None
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    print("GOZ_STREAM " + json.dumps(dict(samples=sent, valid=valid, elapsed=time.monotonic()-started)), flush=True)


def check_alignment(args, model, landmarker, screen):
    """Fresh probe targets are never training data; optional 3-target recenter."""
    import cv2
    from gazekit import ui
    from gazekit.camera import open_camera
    from gazekit.tracker import FaceTracker
    from gazekit.calibrate import validate, MARGINAL_FRAC
    from gazekit.live import _quick_align
    predict, ridge = load_predictor(model, screen)
    cap, tracker = open_camera(int(args.camera)), FaceTracker(str(landmarker))
    win = ui.FullscreenWindow("goz-gaze-check", screen)
    coefficients = alignment_values(model)
    try:
        if args.recenter:
            coefficients = _quick_align(win, cap, tracker, predict)
        ax, bx, ay, by = coefficients
        from gazekit.model import transform
        class Aligned:
            def predict(self, features):
                p = ridge.pipe.predict(transform(features))[0] + ridge.bias
                return (ax * float(p[0]) + bx, ay * float(p[1]) + by)
        error, points, _, _ = validate(win, cap, tracker, Aligned(), seed=None)
        passed = len(points) >= 4 and math.isfinite(error) and error / math.hypot(*screen) <= MARGINAL_FRAC
        report = dict(passed=passed, meanErrorPx=error if math.isfinite(error) else None,
                      validation=points, heldOut=True, checkedAt=time.time(), recentered=args.recenter)
        atomic_json(model.parent / "gaze_check.json", report)
        if not passed:
            raise ValueError("Fresh gaze check failed. Previous alignment/model preserved; recalibrate in the current seating/camera position.")
        if args.recenter:
            atomic_json(model.parent / "gaze_alignment.json", {"coefficients": list(coefficients), "validatedAt": time.time()})
        metadata_path = model.with_suffix(".report.json")
        metadata = json.loads(metadata_path.read_text())
        original_path = model.with_suffix(".initial-report.json")
        if not original_path.exists():
            atomic_json(original_path, metadata)
        fraction = error / math.hypot(*screen)
        metadata.update(verdict="STABLE" if fraction <= .045 else "USABLE", validation=points,
                        mean_error_px=round(error, 1), mean_error_frac_diag=round(fraction, 4),
                        validationHeldOut=True, checkedAt=report["checkedAt"], aligned=args.recenter)
        atomic_json(metadata_path, metadata)
        # A successful check can recover a rejected/orphaned managed attempt.
        # Registration uses the same manifest format as SensorSetup, and never
        # deletes the earlier model or initial validation report.
        if model.parent.parent.name == "calibration":
            directory = model.parent.parent.parent
            suffix = "" if args.profile == "default" else "-" + hashlib.sha256(args.profile.encode()).hexdigest()[:16]
            atomic_json(model.parent.parent / ("saved-gaze" + suffix + ".json"),
                        dict(version=2, profileId=args.profile, model=str(model.relative_to(directory)),
                             report=metadata, sha256=hashlib.sha256(model.read_bytes()).hexdigest(), savedAt=time.time()))
        print("GOZ_CHECK " + json.dumps(report), flush=True)
    finally:
        tracker.close()
        cap.release()
        win.close()
        cv2.destroyAllWindows()


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
    parser.add_argument("command", choices=("calibrate", "stream", "check", "inspect"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--camera", default="select")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report")
    parser.add_argument("--port", type=int, default=5590)
    parser.add_argument("--setup-id")
    parser.add_argument("--camera-device-id")
    parser.add_argument("--camera-name")
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--profile", default=os.getenv("GOZ_VIEWER_PROFILE", "default"))
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--recenter", action="store_true", help="Check a new alignment on independent probe targets before saving it")
    args = parser.parse_args()
    repo, model = Path(args.repo).resolve(), Path(args.model).resolve()
    sys.path.insert(0, str(repo))
    # Relative Gazekit assets/config always resolve to its checkout; calibration
    # recordings and trained models stay inside this GOZ attempt's data folder.
    os.chdir(repo)
    from gazekit.screen import screen_size
    screen = screen_size()
    if args.command != "calibrate":
        metadata = json.loads(model.with_suffix(".report.json").read_text())
        validate_compatibility(metadata, repo, args.profile, screen)
        if args.command == "stream" and metadata.get("verdict") not in ("STABLE", "USABLE"):
            raise ValueError("This model failed gaze validation. Run a passing fresh check or recalibrate before streaming it into the adaptive story.")
        args.camera_device_id = args.camera_device_id or metadata.get("cameraDeviceId")
        args.camera_name = args.camera_name or metadata.get("cameraName")
        if args.command == "inspect":
            _, ridge = load_predictor(model, screen)
            print("GOZ_MODEL_LOADED " + json.dumps(dict(screen=list(ridge.screen_size), profileId=args.profile, schemaVersion=CALIBRATION_SCHEMA)), flush=True)
            return
    if args.command in ("stream", "check") and (args.camera_device_id or args.camera_name):
        from gazekit import camera as camera_module
        args.camera = saved_camera(connected_cameras(camera_module), args.camera_device_id, args.camera_name)
    if args.camera == "select":
        import cv2
        from gazekit import ui, camera as camera_module
        args.camera = choose_camera(ui, cv2, camera_module, screen_size)
        print("GOZ_CAMERA " + json.dumps({"camera": args.camera}), flush=True)
    elif args.camera.isdigit():
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
        import gazekit.calibrate as calibration
        from gazekit import camera as camera_module
        camera_info = next((c for c in connected_cameras(camera_module) if str(c["index"]) == args.camera), {})
        metadata = dict(camera=args.camera, cameraName=camera_info.get("name"), cameraDeviceId=camera_info.get("deviceId"),
                        profileId=args.profile, schemaVersion=CALIBRATION_SCHEMA, modelSchema=model_schema(repo),
                        coordinateSpace="screen-points", validationHeldOut=True,
                        cameraConfiguration=dict(mirrored=True, requestedSize=[1920, 1080]), backend="ridge")
        original_validate = calibration.validate
        def held_out_validate(*values, **kwargs):
            error, points, _, _ = original_validate(*values, **kwargs)
            if len(points) < 4:
                raise ValueError("Gaze validation needs at least four fresh targets; too few usable observations were captured. Retry with steady lighting and face position.")
            # Gazekit otherwise folds these probes into its final fitted model.
            return error, points, None, None
        calibration.validate = held_out_validate
        original_save = calibration.GazeModel.save
        def save_atomically(instance, path, report=None):
            report = dict(report or {}, **metadata)
            temporary = Path(path).with_name("gaze_model-pending.pkl")
            original_save(instance, temporary, report)
            temporary.chmod(0o600)
            with temporary.open("rb") as saved_file:
                os.fsync(saved_file.fileno())
            temporary.replace(path)
            temporary.with_suffix(".report.json").unlink(missing_ok=True)
            atomic_json(Path(path).with_suffix(".report.json"), report)
            if report.get("verdict") in ("STABLE", "USABLE"):
                atomic_json(args.report, report)
                # Register before Gazekit's blocking results screen. Closing the
                # app after successful training must not lose the saved model.
                print("GOZ_CALIBRATION_SAVED " + json.dumps(report), flush=True)
            else:
                print("GOZ_CALIBRATION_REJECTED " + json.dumps(report), flush=True)
        calibration.GazeModel.save = save_atomically
        report = calibration.run(camera_index=camera, model_out=str(model), dataset_root=str(model.parent / "dataset"), landmarker=str(landmarker))
        if not report or report.get("verdict") not in ("STABLE", "USABLE") or not model.is_file():
            raise ValueError("Gazekit calibration was cancelled or did not pass validation.")
        report.update(metadata)
        atomic_json(args.report, report)
    else:
        if args.command == "check":
            check_alignment(args, model, landmarker, screen)
        else:
            stream_gaze(args, model, landmarker, screen)


if __name__ == "__main__":
    main()
