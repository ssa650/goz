"""Bounded gaze timing and blink state, with no camera/model dependencies."""
import json
import hashlib
import math
import threading
import time
from pathlib import Path

MAX_FRAME_AGE_S = .5
MAX_COORDINATE_HOLD_S = .25
RECOVERY_WAIT_S = 2.
RECOVERY_HOLD_S = .75
RECOVERY_MAX_GAP_S = .15


def calibration_eye_evidence(model, metadata):
    """Read only this validated model's UID-bound calibration observations.

    Open-target rows establish a lid baseline, not a closed-eye classifier or
    a probability of correctness. Missing/mixed/damaged evidence fails closed.
    No online learning and no modification of calibration/model artifacts.
    """
    disabled = dict(enabled=False, reason="missing-compatible-calibration-evidence")
    if (not isinstance(metadata, dict) or metadata.get("schemaVersion") != 3
            or metadata.get("verdict") not in ("STABLE", "USABLE")
            or metadata.get("validationHeldOut") is not True
            or metadata.get("cameraIdentityVerified") is not True
            or metadata.get("cameraBackend") != "avfoundation-uid"
            or not metadata.get("cameraDeviceId") or not metadata.get("modelSchema")):
        return disabled
    paths = list((Path(model).parent / "dataset").glob("session_*/samples.jsonl"))
    if len(paths) != 1:
        return dict(disabled, reason="missing-or-mixed-calibration-dataset")
    try:
        path = paths[0]
        if path.stat().st_size > 8 * 1024 * 1024:
            return dict(disabled, reason="oversized-calibration-dataset")
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines()]
        meta = rows[0]
        if (not meta.get("meta") or meta.get("camera") != metadata["cameraDeviceId"]
                or meta.get("screen_size") != metadata.get("screen")
                or any(row.get("meta") for row in rows[1:])):
            return dict(disabled, reason="calibration-dataset-identity-mismatch")
        groups = {tag: [] for tag in ("calib", "probe")}
        for row in rows[1:]:
            if row.get("tag") not in groups:
                continue
            features, score, target = row["features"], row["blink"], row["target"]
            if (len(features) != 14 or len(target) != 2
                    or not all(type(v) in (int, float) and math.isfinite(v)
                               for v in [*features, score, *target, row["yaw"], row["pitch"]])
                    or not 0 <= score < .35 or min(features[3], features[7]) <= .16 * 1.15
                    or max(features[3], features[7]) >= 1
                    or abs(row["yaw"]) >= 30 or abs(row["pitch"]) >= 30):
                continue
            groups[row["tag"]].append(row)
        if (len(groups["calib"]) < 100 or len(groups["probe"]) < 80
                or any(len({tuple(r["target"]) for r in group}) < 4 for group in groups.values())):
            return dict(disabled, reason="insufficient-calibration-open-evidence")
        # A low percentile covers measured target-dependent lid narrowing;
        # held-out probes must independently support it. Do not lower it using
        # ambiguous live samples. The percentile and timers are engineering
        # choices; their values are not claimed as measured blink accuracy.
        lids = sorted(min(r["features"][3], r["features"][7]) for r in groups["calib"])
        q = (len(lids)-1) * .01
        lo = math.floor(q)
        floor = lids[lo] + (lids[min(lo+1, len(lids)-1)]-lids[lo]) * (q-lo)
        support = sum(min(r["features"][3], r["features"][7]) >= floor for r in groups["probe"])
        if support / len(groups["probe"]) < .95:
            return dict(disabled, reason="probe-open-evidence-disagrees")
        return dict(enabled=True, reason="validated-calibration-open-lids", openMin=floor,
                    source="model-calibration-target-observations", datasetSha256=hashlib.sha256(data).hexdigest(),
                    calibrationSamples=len(lids), probeSamples=len(groups["probe"]),
                    probeSupport=support, observedOpenMin=lids[0], percentile=.01,
                    cameraDeviceId=metadata["cameraDeviceId"], profileId=metadata.get("profileId"),
                    modelSchema=metadata["modelSchema"])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(disabled, reason="invalid-calibration-open-evidence")


def observation_quality_reason(obs, frame_width):
    """Calibration's generous setup/pose bounds plus finite model features.

    Gazekit supplies no prediction confidence. These are measured observation
    checks, not a fabricated confidence score or a guarantee of gaze accuracy.
    """
    if not obs.ok:
        return "no-face"
    try:
        features = obs.features
        if len(features) != 14 or not all(math.isfinite(float(v)) for v in features):
            return "invalid-model-features"
        if not all(math.isfinite(float(v)) for v in (obs.yaw, obs.pitch, obs.brightness, obs.interocular_px)):
            return "invalid-observation-quality"
        if abs(obs.yaw) >= 30 or abs(obs.pitch) >= 30:
            return "head-pose-out-of-range"
        if not 40 <= obs.brightness <= 235:
            return "lighting-out-of-range"
        if not .03 <= obs.interocular_px / max(frame_width, 1) <= .40:
            return "face-scale-out-of-range"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "invalid-observation-quality"
    return None


class HeldCoordinates:
    """Finite wire coordinates, with a bounded lifetime for invalid holds."""
    def __init__(self, screen):
        self.center = (screen[0] / 2, screen[1] / 2)
        self.xy, self.at = self.center, None

    def expired(self, captured):
        return self.at is None or not 0 <= captured-self.at <= MAX_COORDINATE_HOLD_S

    def select(self, point, captured):
        if point is not None:
            self.xy, self.at = point, captured
            return self.xy, "current", 0.
        if not self.expired(captured):
            return self.xy, "held-invalid", captured-self.at
        self.xy, self.at = self.center, None
        return self.xy, "unavailable", None


class BlinkGate:
    """Strict closure gates with bounded, explicitly uncertain lid recovery.

    Generic onset/reopen thresholds and personalized blink profiles retain
    their meaning. Recovery needs calibrated open lids, sub-onset scores and
    continuous fresh, good observations; elapsed gate time alone never opens.
    """
    def __init__(self, profile_path="data/blink_profile.json", eye_evidence=None):
        self.on, self.off, self.open_min, self.hold = .28, .18, .16, .25
        self.profile = "generic"
        path = Path(profile_path)
        if path.exists():
            p = json.loads(path.read_text())
            values = [p[k] for k in ("blink_on", "blink_off", "open_min")]
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or not 0 <= values[1] < values[0] <= 1 or not 0 < values[2] < 1):
                raise ValueError("Invalid blink profile; repair the profile before streaming.")
            self.on, self.off, self.open_min = values
            self.profile = "personalized"
        self.frozen = False
        self.reopened_at = self.gated_at = self.last_at = None
        self.reason = "open"
        self.score = self.openness = None
        self.evidence = dict(eye_evidence or dict(enabled=False, reason="missing-compatible-calibration-evidence"))
        if self.profile == "personalized":
            self.evidence = dict(enabled=False, reason="personalized-blink-profile-takes-precedence")
        self.recovery_at = None
        self.recovered = self.uncertain = False
        self.recoveries = 0

    def reject(self, captured, reason):
        self.frozen, self.reopened_at, self.recovery_at = True, None, None
        self.recovered, self.uncertain, self.reason = False, True, reason
        if math.isfinite(captured):
            if self.gated_at is None: self.gated_at = captured
            if self.last_at is None or captured > self.last_at: self.last_at = captured
        return True

    def update(self, obs, captured, *, fresh=True, quality_reason=None):
        if not math.isfinite(captured):
            return self.reject(captured, "invalid-capture-time")
        if self.last_at is not None and captured <= self.last_at:
            return self.reject(captured, "non-increasing-capture-time")
        gap = self.last_at is not None and (captured <= self.last_at or captured-self.last_at > MAX_FRAME_AGE_S)
        recovery_gap = self.last_at is not None and captured-self.last_at > RECOVERY_MAX_GAP_S
        self.last_at = captured
        if gap or (recovery_gap and self.recovered):
            self.reject(captured, "capture-gap")
        if recovery_gap: self.recovery_at = None
        features = getattr(obs, "features", None)
        try:
            self.score = float(obs.blink) if obs.ok else None
            lids = [float(features[3]), float(features[7])] if obs.ok and features is not None else []
            self.openness = (min(lids) if all(math.isfinite(v) and 0 <= v < 1 for v in lids)
                             and lids else None)
        except (TypeError, ValueError, IndexError):
            self.score = self.openness = None
        if self.score is not None and not math.isfinite(self.score): self.score = None
        if not fresh: return self.reject(captured, "stale-frame")
        if quality_reason: return self.reject(captured, quality_reason)
        if not obs.ok:
            closed, opened, reason = True, False, "no-face"
        elif (self.openness is None or not math.isfinite(self.openness)
              or self.score is None or not math.isfinite(self.score)
              or not 0 <= self.score <= 1 or not 0 <= self.openness < 1):
            closed, opened, reason = True, False, "invalid-eye-features"
            self.openness = self.openness if self.openness is not None and math.isfinite(self.openness) else None
            self.score = self.score if self.score is not None and math.isfinite(self.score) else None
        else:
            score_closed, lid_closed = self.score > self.on, self.openness < self.open_min
            closed = score_closed or lid_closed
            opened = self.score < self.off and self.openness > self.open_min * 1.15
            reason = "blink-score" if score_closed else "eyelid-collapse" if lid_closed else "uncertain-eye-state"
        candidate = (not closed and self.evidence.get("enabled") is True
                     and self.score < self.on and self.openness >= self.evidence["openMin"])
        if not candidate: self.recovery_at = None
        self.uncertain = not opened
        if self.recovered and not opened and not candidate:
            return self.reject(captured, reason)
        if closed:
            return self.reject(captured, reason)
        elif self.frozen:
            if self.gated_at is None: self.gated_at = captured
            if not opened:
                self.reopened_at, self.reason = None, reason
            else:
                if self.reopened_at is None: self.reopened_at = captured
                self.reason = "reopen-hold"
                if captured-self.reopened_at >= self.hold:
                    self.frozen, self.recovered, self.reason = False, False, "open"
            if self.frozen and candidate:
                if self.recovery_at is None: self.recovery_at = captured
                if captured-self.gated_at >= RECOVERY_WAIT_S:
                    self.reason = "calibrated-open-hold"
                    if captured-self.recovery_at >= RECOVERY_HOLD_S:
                        self.frozen, self.recovered, self.reason = False, True, "calibrated-open-recovery"
                        self.recoveries += 1
        else:
            if opened: self.recovered = False
            self.reason = "calibrated-open-recovery" if self.recovered else "open"
        if self.frozen and self.gated_at is None: self.gated_at = captured
        if not self.frozen: self.gated_at = None
        return self.frozen

    def diagnostics(self, captured):
        return dict(reason=self.reason, score=self.score, openness=self.openness,
                    gatedSeconds=0. if self.gated_at is None else max(0., captured-self.gated_at),
                    thresholds=dict(on=self.on, off=self.off, openMin=self.open_min, reopenHoldS=self.hold),
                    profile=self.profile, uncertain=self.uncertain, recovered=self.recovered,
                    recovery=dict(**self.evidence, waitS=RECOVERY_WAIT_S, holdS=RECOVERY_HOLD_S,
                                  maxEvidenceGapS=RECOVERY_MAX_GAP_S, count=self.recoveries,
                                  evidenceSeconds=0. if self.recovery_at is None else max(0., captured-self.recovery_at)))


class StageDiagnostics:
    """Fixed stage names, cumulative counts and bounded latency summaries."""
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {key: 0 for key in ("frames", "processed", "sent", "valid", "noFace", "blink",
                                        "prolongedGate", "staleBefore", "staleAfter", "readTimeouts")}
        self.timings = {key: dict(count=0, totalMs=0., maxMs=0., lastMs=0.) for key in
                        ("modelLoad", "trackerLoad", "cameraOpen", "read", "inference", "prediction", "send",
                         "captureToRead", "captureToSend", "frameInterval")}
        self.stage, self.stage_at = "starting", time.monotonic()
        self.started, self.cpu_started = self.stage_at, time.process_time()
        self.last_frame = None
        self.eye = None

    def enter(self, stage):
        with self.lock: self.stage, self.stage_at = stage, time.monotonic()

    def count(self, key):
        with self.lock: self.counts[key] += 1

    def observe(self, key, seconds):
        with self.lock:
            row = self.timings[key]
            ms = max(0., seconds*1000)
            row.update(count=row["count"]+1, totalMs=row["totalMs"]+ms,
                       maxMs=max(row["maxMs"], ms), lastMs=ms)

    def frame(self, captured):
        if self.last_frame is not None: self.observe("frameInterval", captured-self.last_frame)
        self.last_frame = captured
        self.count("frames")

    def snapshot(self):
        with self.lock:
            return dict(counts=dict(self.counts), timings={k: dict(v) for k,v in self.timings.items()},
                        stage=self.stage, stageAgeS=time.monotonic()-self.stage_at,
                        wallS=time.monotonic()-self.started, cpuS=time.process_time()-self.cpu_started,
                        blink=dict(self.eye) if self.eye else None)

    def set_eye(self, eye):
        with self.lock: self.eye = eye


class DurableDiagnostics:
    """One overwritten snapshot every two seconds, including during inference stalls.

    Writes and fsync happen off the gaze processing thread. No raw pixels, no
    unbounded history and no per-frame disk/log work.
    """
    def __init__(self, path, snapshot, save):
        self.path, self.snapshot, self.save = path, snapshot, save
        self.stop = threading.Event()
        self.errors = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self):
        try:
            value = self.snapshot()
            value["diagnosticWriteErrors"] = self.errors
            self.save(self.path, value)
        except (OSError, ValueError):
            self.errors += 1

    def _run(self):
        self.write()
        while not self.stop.wait(2.): self.write()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)
        # If storage is stalled, do not launch a second competing fsync.
        if not self.thread.is_alive(): self.write()
