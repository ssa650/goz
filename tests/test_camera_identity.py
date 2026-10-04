"""Camera regressions use fake native pipes/devices, never a physical camera."""
import asyncio
import io
import json
from pathlib import Path
import struct
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from backend import gaze_worker as worker, native_camera as native
from backend.sensor_setup import SensorSetup
from test_sensor_setup import sensors


PHONE = dict(index=7, name="iPhone Camera", deviceId="phone-uid")
OPENED = dict(name="Shayan's iPhone Camera", deviceId="phone-uid", verified=True, backend="avfoundation-uid", frameProtocol=2)


def frame_packet(pixels, sequence=1, **changes):
    now, mono = time.time(), time.monotonic()
    metadata = dict(width=2, height=1, length=len(pixels), sequence=sequence,
                    capturedAt=now-.03, capturedMonotonic=mono-.03,
                    deliveredAt=now, deliveredMonotonic=mono,
                    nativeDelivered=sequence, nativeDropped=0, nativeReplaced=0, previousWriteMs=0.)
    metadata.update(changes)
    encoded = json.dumps(metadata).encode()
    return struct.pack("<I", len(encoded)) + encoded + pixels


@pytest.fixture(autouse=True)
def no_native_helper(monkeypatch):
    monkeypatch.setattr(native, "helper_path", lambda: pytest.fail("Tests must never run native capture"))


class Pipe(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.stopped = threading.Event()

    def read(self, length=-1):
        result = super().read(length)
        if not result:
            self.stopped.wait(5)
        return result


def fake_capture(monkeypatch, opened=OPENED, packets=b""):
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=Pipe(json.dumps(opened).encode() + b"\n" + packets),
                              stderr=io.BytesIO(), poll=lambda: None)
    process.wait = lambda **kw: process.stdout.stopped.set()
    calls = []
    monkeypatch.setattr(native, "helper_path", lambda: "fake-helper")
    monkeypatch.setattr(native.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or process)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(COLOR_BGRA2BGR=1, cvtColor=lambda p, code: p[..., :3].copy()))
    return process, calls


def test_uid_capture_ignores_index_and_decodes_native_frame(monkeypatch):
    pixels = bytes([11, 22, 33, 255, 44, 55, 66, 255])
    process, calls = fake_capture(monkeypatch, packets=frame_packet(pixels))
    cap = native.NativeCapture(PHONE)
    try:
        assert calls == [["fake-helper", "capture", "phone-uid"]]
        assert cap.identity == OPENED
        ok, frame = cap.read()
        assert ok and frame.tolist() == [[[11, 22, 33], [44, 55, 66]]]
    finally:
        cap.release()
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("opened", [dict(OPENED, deviceId="mac-uid"), dict(OPENED, verified=False),
                                     dict(OPENED, backend="opencv-index"), dict(OPENED, name="")])
def test_wrong_opened_identity_releases_capture_and_never_falls_back(monkeypatch, opened):
    process, calls = fake_capture(monkeypatch, opened)
    with pytest.raises(ValueError, match="identity does not match.*preserved"):
        native.NativeCapture(PHONE)
    assert len(calls) == 1 and calls[0][-1] == "phone-uid" and process.stdin.closed


def test_native_disconnect_and_no_frames_are_explicit_errors(monkeypatch):
    process, _ = fake_capture(monkeypatch)
    cap = native.NativeCapture(PHONE)
    try:
        process.stdout.stopped.set()
        # The reader detects EOF; synchronize by reading with the normal timeout.
        with pytest.raises(ValueError, match="disconnected or stopped"):
            cap.read()
    finally:
        cap.release()
    cap = object.__new__(native.NativeCapture)
    cap.error, cap.closed, cap.last_frame = None, False, 0
    import queue
    cap.packets = queue.Queue()
    cap._stats_lock = threading.Lock()
    cap._stats = dict(readTimeouts=0)
    with pytest.raises(ValueError, match="no frames.*no fallback"):
        cap.read()


def test_missing_or_duplicate_name_never_resolves_to_macbook():
    mac = dict(index=0, name="FaceTime HD Camera", deviceId="mac")
    assert worker.resolve_camera([mac, PHONE], "phone-uid") is PHONE
    assert worker.resolve_camera([PHONE, mac], "iPhone Camera") is PHONE
    with pytest.raises(ValueError, match="unavailable"):
        worker.resolve_camera([mac], "iPhone Camera")
    with pytest.raises(ValueError, match="ambiguous"):
        worker.resolve_camera([PHONE, dict(PHONE, deviceId="other-phone")], "iPhone Camera")
    with pytest.raises(ValueError, match="unavailable"):
        worker.saved_camera([dict(PHONE, deviceId="replacement-phone")], "phone-uid", PHONE["name"])


def test_open_event_uses_native_input_name(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(native, "NativeCapture", lambda selected, **kw: calls.append(selected) or SimpleNamespace(identity=OPENED))
    args = SimpleNamespace(selected_camera=PHONE, camera="7")
    worker.open_selected_camera(args)
    event = json.loads(capsys.readouterr().out.split("GOZ_CAMERA ")[1])
    assert calls == [PHONE] and event["name"] == OPENED["name"]
    assert event["verified"] is True and event["deviceId"] == PHONE["deviceId"]


@pytest.mark.parametrize("command", ["stream", "check", "inspect"])
def test_old_calibration_is_rejected_before_camera_enumeration_or_capture(monkeypatch, tmp_path, command):
    package = ModuleType("gazekit")
    package.__path__ = []
    screen = ModuleType("gazekit.screen")
    screen.screen_size = lambda: (100, 80)
    monkeypatch.setitem(sys.modules, "gazekit", package)
    monkeypatch.setitem(sys.modules, "gazekit.screen", screen)
    model = tmp_path / "model.pkl"
    model.write_bytes(b"original-model")
    report = model.with_suffix(".report.json")
    report.write_text(json.dumps(dict(schemaVersion=2, cameraName="iPhone Camera", cameraDeviceId="phone-uid")))
    before = report.read_bytes()
    monkeypatch.setattr(worker, "connected_cameras", lambda: pytest.fail("Legacy metadata cannot be trusted"))
    monkeypatch.setattr(sys, "argv", ["worker", command, "--repo", str(tmp_path), "--model", str(model)])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="predates verified.*full eye recalibration"):
        worker.main()
    assert report.read_bytes() == before and model.read_bytes() == b"original-model"


def test_calibration_metadata_comes_from_opened_input_not_selector(monkeypatch, tmp_path):
    package = ModuleType("gazekit")
    package.__path__ = []
    calibration = ModuleType("gazekit.calibrate")
    dataset = ModuleType("gazekit.dataset")
    screen = ModuleType("gazekit.screen")
    screen.screen_size = lambda: (100, 80)
    calls = []
    class Model:
        def save(self, path, report=None):
            Path(path).write_bytes(b"new-model")
    def run(camera_index, model_out, **kwargs):
        calls.append(camera_index)
        calibration.open_camera(camera_index)
        report = dict(verdict="STABLE", screen=[100, 80])
        Model().save(model_out, report)
        return report
    calibration.run, calibration.GazeModel, calibration.validate = run, Model, lambda: None
    for name, module in [("gazekit", package), ("gazekit.calibrate", calibration),
                         ("gazekit.dataset", dataset), ("gazekit.screen", screen)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(worker, "connected_cameras", lambda: [PHONE])
    monkeypatch.setattr(worker, "model_schema", lambda repo: "schema")
    monkeypatch.setattr(native, "NativeCapture", lambda selected, **kw: SimpleNamespace(identity=OPENED))
    model = tmp_path / "attempt" / "model.pkl"
    model.parent.mkdir()
    (tmp_path / "face_landmarker.task").write_bytes(b"mock-asset")
    ready = model.parent / "ready.json"
    monkeypatch.setattr(sys, "argv", ["worker", "calibrate", "--repo", str(tmp_path), "--camera", "iPhone Camera",
                                    "--model", str(model), "--report", str(ready)])
    monkeypatch.chdir(tmp_path)
    worker.main()
    report = json.loads(ready.read_text())
    assert calls == ["phone-uid"]
    assert report["cameraName"] == OPENED["name"] and report["cameraDeviceId"] == "phone-uid"
    assert report["schemaVersion"] == 3 and report["cameraIdentityVerified"] is True
    assert json.loads(model.with_suffix(".report.json").read_text()) == report
    assert json.loads(dataset.CONFIG_PATH.read_text())["camera"] == "phone-uid"


@pytest.mark.asyncio
async def test_full_recalibration_failure_preserves_original_manifest_and_model(monkeypatch, tmp_path):
    setup = SensorSetup(sensors(), tmp_path)
    original = tmp_path / "calibration" / "old" / "model.pkl"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"old-model")
    setup.save_gaze(original, dict(verdict="STABLE", camera="0"))
    before = setup.saved_gaze_path.read_bytes()
    async def spawn(*args, **kwargs):
        reader = asyncio.StreamReader()
        reader.feed_data(b"Selected camera is unavailable; no fallback was attempted.\n")
        reader.feed_eof()
        async def wait(): return 1
        return SimpleNamespace(stdout=reader, returncode=1, wait=wait)
    setup.spawn = spawn
    await setup.recalibrate_gaze()
    await setup.task
    assert setup.phase == "failed" and not setup.gaze_calibrated
    assert setup.saved_gaze_path.read_bytes() == before and original.read_bytes() == b"old-model"
    assert "unavailable" in setup.error


@pytest.mark.asyncio
async def test_setup_surfaces_opened_device_name(monkeypatch, tmp_path):
    setup = SensorSetup(sensors(), tmp_path)
    async def spawn(*args, **kwargs):
        reader = asyncio.StreamReader()
        event = dict(camera="7", **OPENED, state="opened")
        reader.feed_data(("GOZ_CAMERA " + json.dumps(event) + "\n").encode())
        reader.feed_eof()
        return SimpleNamespace(stdout=reader, returncode=0)
    setup.spawn = spawn
    await setup.launch("gaze_calibration", ["fake"], tmp_path)
    await setup.readers["gaze_calibration"]
    assert setup.snapshot()["camera"]["name"] == OPENED["name"]
    assert setup.snapshot()["camera"]["verified"] is True
    assert OPENED["name"] in setup.message


@pytest.mark.asyncio
async def test_full_recalibration_api_retains_manifest_and_blocks_busy_generation(monkeypatch, tmp_path):
    from test_backend import harness
    async with harness(tmp_path) as (engine, adapter, client):
        setup = client._transport.app.state.sensors.setup
        model = setup.directory / "calibration" / "old" / "model.pkl"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"old-model")
        setup.save_gaze(model, dict(verdict="STABLE", camera="0"))
        before = setup.saved_gaze_path.read_bytes()
        calls = []
        async def start(fresh_gaze=False): calls.append(fresh_gaze)
        monkeypatch.setattr(setup, "start", start)
        response = await client.post("/api/sensors/gaze-recalibrate")
        assert response.status_code == 202 and calls == [True]
        assert response.json()["gaze"]["savedCalibration"]
        assert setup.saved_gaze_path.read_bytes() == before
        engine.new_job(dict(mode="text", prompt="Existing", duration=5, resolution="480P"))
        assert (await client.post("/api/sensors/gaze-recalibrate")).status_code == 409
        assert calls == [True] and not adapter.submissions
