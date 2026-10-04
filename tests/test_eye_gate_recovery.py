"""Evidence-based eye recovery, using synthetic observations and fake devices."""
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from backend import gaze_worker as worker
from backend.gaze_pipeline import (BlinkGate, HeldCoordinates, calibration_eye_evidence,
                                   observation_quality_reason, RECOVERY_MAX_GAP_S)
from test_gaze_pipeline import observation


def evidence(open_min=.267):
    return dict(enabled=True, reason="validated-calibration-open-lids", openMin=open_min)


def calibration(tmp_path):
    model = tmp_path / "gaze_model.pkl"
    metadata = dict(schemaVersion=3, verdict="STABLE", validationHeldOut=True,
                    cameraIdentityVerified=True, cameraBackend="avfoundation-uid",
                    cameraDeviceId="verified-uid", modelSchema="model-schema", profileId="viewer",
                    screen=[100, 80])
    path = tmp_path / "dataset" / "session_test" / "samples.jsonl"
    path.parent.mkdir(parents=True)
    rows = [dict(meta=True, camera="verified-uid", screen_size=[100, 80])]
    for tag, n in [("calib", 120), ("probe", 80)]:
        for i in range(n):
            rows.append(dict(tag=tag, target=[10+(i%4)*20, 40], blink=.24,
                             features=observation(openness=.27).features, yaw=0, pitch=0))
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return model, metadata, path


def test_calibration_baseline_is_measured_and_identity_bound(tmp_path):
    model, metadata, path = calibration(tmp_path)
    e = calibration_eye_evidence(model, metadata)
    assert e["enabled"] and e["openMin"] == .27
    assert e["calibrationSamples"] == 120 and e["probeSupport"] == 80
    assert len(e["datasetSha256"]) == 64
    assert not calibration_eye_evidence(model, dict(metadata, cameraDeviceId="other"))["enabled"]
    assert not calibration_eye_evidence(model, dict(metadata, schemaVersion=2))["enabled"]
    assert not calibration_eye_evidence(model, dict(metadata, validationHeldOut=False))["enabled"]
    assert not calibration_eye_evidence(model, dict(metadata, verdict="FAILED"))["enabled"]
    path.write_text("broken")
    assert not calibration_eye_evidence(model, metadata)["enabled"]


def test_insufficient_mixed_and_disagreeing_evidence_do_not_enable_recovery(tmp_path):
    model, metadata, path = calibration(tmp_path)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for row in rows:
        if row.get("tag") == "probe": row["features"][3] = .22
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    assert calibration_eye_evidence(model, metadata)["reason"] == "probe-open-evidence-disagrees"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows[:40]))
    assert not calibration_eye_evidence(model, metadata)["enabled"]
    other = path.parent.parent / "session_other" / "samples.jsonl"
    other.parent.mkdir(); other.write_text("{}\n")
    assert calibration_eye_evidence(model, metadata)["reason"] == "missing-or-mixed-calibration-dataset"


def test_ambiguous_scores_regain_gaze_only_with_sustained_calibrated_lids(tmp_path):
    g = BlinkGate(tmp_path / "missing", evidence())
    assert g.update(observation(.8, .03), 0.)
    for i in range(1, 60): assert g.update(observation(.24, .275), i/30)
    assert not g.update(observation(.24, .275), 2.)
    d = g.diagnostics(2.)
    assert d["reason"] == "calibrated-open-recovery" and d["uncertain"] and d["recovered"]
    assert d["recovery"]["count"] == 1 and d["thresholds"]["on"] == .28
    assert g.update(observation(.24, .26), 2+1/30)  # Evidence no longer meets baseline.
    # A long elapsed gate cannot replace a new continuous evidence window.
    assert g.update(observation(.24, .275), 10.)
    for i in range(1, 23): assert g.update(observation(.24, .275), 10+i/30)
    assert not g.update(observation(.24, .275), 10.8)


@pytest.mark.parametrize("obs,quality,fresh", [
    (observation(.8, .28), None, True),
    (observation(.05, .03), None, True),
    (observation(.24, .23), None, True),
    (observation(ok=False), None, True),
    (observation(float("nan")), None, True),
    (observation(.24, .275), "invalid-prediction", True),
    (observation(.24, .275), "lighting-out-of-range", True),
    (observation(.24, .275), None, False),
])
def test_elapsed_time_never_accepts_closure_absence_stale_or_low_quality(tmp_path, obs, quality, fresh):
    g = BlinkGate(tmp_path / "missing", evidence())
    g.update(observation(.8, .03), 0.)
    for i in range(1, 301):
        assert g.update(obs, i/30, fresh=fresh, quality_reason=quality)
    json.dumps(g.diagnostics(10.), allow_nan=False)


def test_interrupted_recovery_duplicate_time_and_delivery_gap_restart_evidence(tmp_path):
    g = BlinkGate(tmp_path / "missing", evidence())
    g.update(observation(.8, .03), 0.)
    for i in range(1, 60): g.update(observation(.24, .275), i/30)
    assert g.update(observation(.24, .275), 2. + RECOVERY_MAX_GAP_S)
    assert g.update(observation(.24, .275), 2. + RECOVERY_MAX_GAP_S)
    assert g.reason == "non-increasing-capture-time"
    for i in range(1, 23): assert g.update(observation(.24, .275), 2.15+i/30)
    assert not g.update(observation(.24, .275), 2.95)
    assert g.update(observation(.24, .275), 3.2)  # Recovered validity also needs cadence.


def test_missing_baseline_and_personal_blink_profiles_keep_strict_reopen(tmp_path):
    g = BlinkGate(tmp_path / "missing")
    g.update(observation(.8, .03), 0.)
    for i in range(1, 301): assert g.update(observation(.24, .275), i/30)
    p = tmp_path / "personal.json"
    p.write_text(json.dumps(dict(blink_on=.4, blink_off=.2, open_min=.12)))
    g = BlinkGate(p, evidence())
    assert not g.evidence["enabled"]
    g.update(observation(.8, .03), 0.)
    for i in range(1, 301): assert g.update(observation(.24, .275), i/30)


def test_each_eye_must_have_finite_openness(tmp_path):
    g = BlinkGate(tmp_path / "missing", evidence())
    o = observation(.05, .275); o.features[7] = float("nan")
    assert g.update(o, 0.) and g.reason == "invalid-eye-features"
    json.dumps(g.diagnostics(0.), allow_nan=False)


def test_observation_quality_rejects_bad_model_input_and_setup_conditions():
    assert observation_quality_reason(observation(), 2) is None
    for field, value, reason in [("yaw", 40, "head-pose-out-of-range"),
                                 ("pitch", float("nan"), "invalid-observation-quality"),
                                 ("brightness", 0, "lighting-out-of-range"),
                                 ("interocular_px", .001, "face-scale-out-of-range")]:
        o = observation(); setattr(o, field, value)
        assert observation_quality_reason(o, 2) == reason
    o = observation(); o.features[10] = float("nan")
    assert observation_quality_reason(o, 2) == "invalid-model-features"
    assert observation_quality_reason(SimpleNamespace(ok=True, features=[]), 2) == "invalid-model-features"


def test_invalid_coordinate_holds_expire_and_recovery_starts_current():
    c = HeldCoordinates((100, 80))
    assert c.select(None, 0.) == ((50., 40.), "unavailable", None)
    assert c.select((-20., 115.), 1.) == ((-20., 115.), "current", 0.)
    assert c.select(None, 1.2)[1] == "held-invalid"
    assert c.select(None, 1.251) == ((50., 40.), "unavailable", None)
    assert c.select((80., 60.), 20.) == ((80., 60.), "current", 0.)


@pytest.mark.parametrize("value", [None, float("nan"), [float("inf"), 2], [1], [None, 2]])
def test_malformed_prediction_never_counts_as_finite(value):
    assert not worker.finite_point(value)


def test_worker_recovery_quality_stale_and_hold_integration(monkeypatch, tmp_path):
    model, metadata, _ = calibration(tmp_path)
    class Clock:
        now = 100.
        def monotonic(self): return self.now
        def time(self): return self.now+1000
    clock = Clock()
    monkeypatch.setattr(worker, "time", clock)
    packets, prediction_frames, filter_inputs = [], [], []
    class Capture:
        identity = dict(deviceId="verified-uid")
        i = -1
        def read_timed(self):
            self.i += 1; clock.now += 1/30
            age = .8 if self.i == 180 else .01
            return True, np.zeros((1, 2, 3), dtype=np.uint8), dict(
                capturedMonotonic=clock.now-age, capturedAt=clock.time()-age,
                sequence=self.i+1, deliveredAt=clock.time(), receivedAt=clock.time(), readAt=clock.time())
        def diagnostics(self): return dict(queueCapacity=1)
        def release(self): pass
    cap = Capture()
    class Tracker:
        _ts_ms = 0
        def __init__(self, path): pass
        def process(self, frame):
            self._ts_ms += 33
            if cap.i == 130: clock.now += .6  # Frame becomes stale during inference.
            o = (observation() if cap.i < 5 else observation(.8, .03) if cap.i < 10
                 else observation(.24, .275, ok=not 150 <= cap.i < 160))
            if 140 <= cap.i < 150: o.brightness = 0
            return o
        def close(self): pass
    class Smoother:
        def apply(self, x, y, t):
            filter_inputs.append((cap.i, x, y)); return (x, y)
    class Sender:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def sendto(self, data, address): packets.append(json.loads(data))
    def predict(obs):
        prediction_frames.append(cap.i)
        return None if 100 <= cap.i < 120 else (70, 50)
    package = ModuleType("gazekit"); package.__path__ = []
    tracker = ModuleType("gazekit.tracker"); tracker.FaceTracker = Tracker
    filters = ModuleType("gazekit.filters"); filters.GazeSmoother = Smoother
    for name, module in [("gazekit", package), ("gazekit.tracker", tracker), ("gazekit.filters", filters)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(worker, "load_predictor", lambda *args: (predict, None))
    monkeypatch.setattr(worker, "open_selected_camera", lambda args: cap)
    monkeypatch.setattr(worker.socket, "socket", lambda *args: Sender())
    args = SimpleNamespace(seconds=8, setup_id="test", profile="viewer", port=5590,
                           calibration_metadata=metadata)
    worker.stream_gaze(args, model, "fake-landmarker", (100, 80))
    by_frame = {p["frameSequence"]-1: p for p in packets}
    assert all(by_frame[i]["valid"] for i in range(5))
    assert all(not by_frame[i]["valid"] for i in range(5, 60))
    assert any(by_frame[i]["valid"] and by_frame[i]["blinkDiagnostics"]["recovered"] for i in range(60, 100))
    assert all(not by_frame[i]["valid"] for i in range(100, 120))
    assert all(by_frame[i]["validityReason"] == "invalid-prediction" for i in range(100, 120))
    assert by_frame[130]["stale"] and by_frame[130]["validityReason"] == "stale-frame"
    assert all(not by_frame[i]["valid"] for i in range(140, 160))
    assert 180 not in by_frame
    assert not set(prediction_frames) & (set(range(5, 10)) | set(range(140, 160)) | {130, 180})
    assert by_frame[5]["coordinateState"] == "held-invalid"
    assert by_frame[20]["coordinateState"] == "unavailable" and by_frame[20]["x"] == 50
    assert any(p["valid"] for p in packets[-20:])
    assert all(p["heldCoordinateAgeS"] is None or p["heldCoordinateAgeS"] <= .25 for p in packets)
    assert all(p["t"] == p["capturedAt"] and p["captureTiming"] == "native-host-clock-pts" for p in packets)
