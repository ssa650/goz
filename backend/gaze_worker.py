"""Run Gazekit's existing calibration/streaming in a camera-owning child."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
import socket
from types import SimpleNamespace

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "stream"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report")
    parser.add_argument("--port", type=int, default=5590)
    parser.add_argument("--setup-id")
    args = parser.parse_args()
    repo, model = Path(args.repo).resolve(), Path(args.model).resolve()
    sys.path.insert(0, str(repo))
    # Relative Gazekit assets/config always resolve to its checkout; calibration
    # recordings and trained models stay inside this GOZ attempt's data folder.
    os.chdir(repo)
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
    camera = int(args.camera) if args.camera.isdigit() else args.camera
    if args.command == "calibrate":
        from gazekit import dataset
        # Label recordings for the explicit camera, without changing Gazekit's
        # global camera configuration or mixing webcam and phone domains.
        camera_config = model.parent / "camera.json"
        camera_config.write_text(json.dumps({"camera": args.camera}))
        dataset.CONFIG_PATH = camera_config
        from gazekit.calibrate import run
        report = run(camera_index=camera, model_out=str(model), dataset_root=str(model.parent / "dataset"), landmarker=str(landmarker))
        if not report or report.get("verdict") not in ("STABLE", "USABLE") or not model.is_file():
            raise ValueError("Gazekit calibration was cancelled or did not pass validation.")
        report.update(camera=args.camera)
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
        stream.run(camera_index=camera, model_path=str(model), port=args.port, align=True, landmarker=str(landmarker))


if __name__ == "__main__":
    main()
