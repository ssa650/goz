"""Live viewer signals on one wall clock (unix seconds).

Gaze: `gazekit stream` UDP datagrams (gazekit docs/STREAM_PROTOCOL.md).
EEG:  Muse 2 via `muselsl stream` (Lab Streaming Layer), reduced to a
      exploratory beta/(alpha+theta) feature relative to a clean baseline.
Both have explicit simulators for rehearsal; the dashboard labels them SIM.
"""
import asyncio
import json
import math
import os
import random
import threading
import time
from collections import deque
from copy import deepcopy

import numpy as np

from .eeg_quality import (CausalEegFilter, MIN_CHANNELS, REASONS, MAX_INTERVAL_S,
                          UniqueCleanTime, channel_diagnostics, muse_metadata)
from .eeg_policy import MIN_CONFIDENCE, channel_eligibility, quality_failure

GAZE_PORT = 5590
EEG_RATE = 256
EEG_WINDOW_S = 2.0
EEG_STEP_S = 0.25
BASELINE_S = 60.0
ARTIFACT_UV = 150.0
EEG_FRESH_S = 3.0
EEG_CLOCK_INITIAL_S = 2.0
EEG_CLOCK_RETRY_S = 0.5
EEG_CLOCK_POLL_S = 0.25
EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


class GazeFeed:
    """Ring buffer of gaze samples; filled by UDP or the simulator."""

    def __init__(self, keep_s=180):
        self.samples = deque(maxlen=keep_s * 40)
        self.source, self.transport = "waiting", None
        self.last_rx = 0.0
        self.expected_setup_id = None
        self.loop_monitor = None
        self.diagnostics = dict(packets=0, accepted=0, malformed=0, rejected=0,
            setupMismatch=0, timestampRejected=0, frameSequenceGaps=0, udpSequenceGaps=0,
            interarrivalMsMax=0., sendToReceiptMsMax=0., captureToReceiptMsMax=0.,
            eventLoopLagMsMax=0., eventLoopLagMsLast=0., eventLoopLagOver500ms=0,
            lastReceivedAt=None)
        self._last_receipt_mono = None
        self._sequence_setup, self._last_sequence = None, None
        self._sent_sequence_setup, self._last_sent_sequence = None, None

    def packet(self, sample):
        received, mono = time.time(), time.monotonic()
        d = self.diagnostics
        d["packets"] += 1
        d["lastReceivedAt"] = received
        if self._last_receipt_mono is not None:
            d["interarrivalMsMax"] = max(d["interarrivalMsMax"], (mono-self._last_receipt_mono)*1000)
        self._last_receipt_mono = mono
        if (not isinstance(sample, dict) or any(type(sample.get(k)) not in (float, int)
                or not math.isfinite(sample[k]) for k in ("t", "x", "y"))
                or self.expected_setup_id and sample.get("setupId") != self.expected_setup_id):
            d["rejected"] += 1
            if isinstance(sample, dict) and self.expected_setup_id and sample.get("setupId") != self.expected_setup_id:
                d["setupMismatch"] += 1
            return False
        if any(key in sample and type(sample[key]) is not bool for key in ("valid", "face", "blink")):
            d["rejected"] += 1
            return False
        if any(k in sample and (type(sample[k]) not in (float, int) or not math.isfinite(sample[k]))
               for k in ("yaw", "pitch", "confidence")):
            d["rejected"] += 1
            return False
        if any(k in sample and (type(sample[k]) not in (float, int) or not math.isfinite(sample[k]))
               for k in ("capturedAt", "sentAt")):
            d["rejected"] += 1
            return False
        # Measure late packets before rejecting their evidence, so a backend
        # blackout remains attributable even when the entire backlog is stale.
        d["captureToReceiptMsMax"] = max(d["captureToReceiptMsMax"], max(0., received-sample["t"])*1000)
        if "sentAt" in sample:
            d["sendToReceiptMsMax"] = max(d["sendToReceiptMsMax"], max(0., received-sample["sentAt"])*1000)
        sent_sequence = sample.get("sentSequence")
        if type(sent_sequence) is int and sent_sequence > 0:
            if self._sent_sequence_setup == sample.get("setupId") and self._last_sent_sequence is not None:
                d["udpSequenceGaps"] += max(0, sent_sequence-self._last_sent_sequence-1)
            self._sent_sequence_setup, self._last_sent_sequence = sample.get("setupId"), sent_sequence
        if abs(sample["t"] - received) > 5 or self.samples and sample["t"] <= self.samples[-1]["t"]:
            d["rejected"] += 1
            d["timestampRejected"] += 1
            return False
        d["accepted"] += 1
        sequence = sample.get("frameSequence")
        if type(sequence) is int and sequence > 0:
            if self._sequence_setup == sample.get("setupId") and self._last_sequence is not None:
                d["frameSequenceGaps"] += max(0, sequence-self._last_sequence-1)
            self._sequence_setup, self._last_sequence = sample.get("setupId"), sequence
        sample = dict(sample, receivedAt=received, receiptDiagnostics=dict(d))
        self.add(sample)
        return True

    def add(self, sample):
        self.samples.append(sample)
        self.last_rx = time.time()

    def latest(self, max_age=0.5):
        if self.samples and time.time() - self.samples[-1]["t"] <= max_age:
            return self.samples[-1]
        return None

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s["t"] <= t1]

    def status(self):
        live = time.time() - self.last_rx < 2
        rate = len(self.window(time.time() - 2, time.time())) / 2
        sample = self.latest()
        return dict(source=self.source, live=live, hz=round(rate, 1),
                    trackingState="unavailable" if not self.samples else "stale" if sample is None else
                    "valid" if sample.get("valid") and sample.get("face", True) and not sample.get("blink", False) else "invalid",
                    coordinateSpace="screen-points", captureClock="unix-seconds",
                    inputDiagnostics=dict(self.diagnostics))

    async def listen(self, port=GAZE_PORT):
        feed = self

        class Protocol(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                try:
                    sample = json.loads(data)
                    if feed.packet(sample):
                        feed.source = "gazekit"
                except ValueError:
                    feed.diagnostics["malformed"] += 1

        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(Protocol, local_addr=("127.0.0.1", port))
        async def monitor():
            # Scheduling delay is distinct from UDP interarrival silence. Use
            # monotonic time; a long callback blackout is measured on resumption.
            while True:
                due = time.monotonic() + .25
                await asyncio.sleep(.25)
                lag = max(0., time.monotonic()-due)*1000
                self.diagnostics["eventLoopLagMsLast"] = lag
                self.diagnostics["eventLoopLagMsMax"] = max(self.diagnostics["eventLoopLagMsMax"], lag)
                self.diagnostics["eventLoopLagOver500ms"] += lag > 500
        self.loop_monitor = asyncio.create_task(monitor())

    def close(self):
        if self.loop_monitor:
            self.loop_monitor.cancel()
        if self.transport:
            self.transport.close()


def band_powers(window):
    """window: (n, channels) µV. Returns mean band powers over channels."""
    x = window - window.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / EEG_RATE)
    return {name: float(spectrum[(freqs >= lo) & (freqs < hi)].mean())
            for name, (lo, hi) in BANDS.items()}


class EegFeed:
    """Exploratory feature series [(t, ratio, z, artifact)] from raw EEG chunks."""

    def __init__(self):
        self.raw = deque(maxlen=int(EEG_RATE * 10))
        self.series = deque(maxlen=int(600 / EEG_STEP_S))
        self.source, self.state = "off", "off"
        self.last_eval = 0.0
        self.lock = threading.Lock()
        self.device_id = None
        self.calibration = None
        self.calibration_samples = []
        self.calibration_seconds = 60.0
        self.calibration_started = None
        self.quality_error = "Waiting for Muse EEG."
        self.connection_error = None
        self.last_sample_at = None
        self.samples_received = 0
        self.channel_quality = {}
        self.acquisition_phase = "off"
        self.filtered = deque(maxlen=self.raw.maxlen)
        self.filter = CausalEegFilter()
        self.clean_time = UniqueCleanTime()
        self.selected_channels = ()
        self.available_channels = ()
        self.calibration_identity = None
        self.channel_calibration_samples = []
        self.feature_history = deque(maxlen=self.series.maxlen)
        self.reject_reasons = []
        self.invalid_until = float("-inf")
        self.input_diagnostics = dict(duplicateSamples=0, outOfOrderSamples=0,
                                      conflictingSamples=0, invalidSamples=0, mismatchedChunks=0,
                                      gapCount=0, missingSamples=0, lastGapSeconds=0.0)
        self.stream_metadata = muse_metadata(None, None)
        self.metadata_error = False
        self.muse_diagnostics = dict(lastSampleAt=None, lastInterruption=None,
                                     readerState="off", retryCount=0, interruptionCount=0)
        self.muse_diagnostic_sink = None

    def muse_event(self, event, **fields):
        """Retain lifecycle evidence separately from live/calibration evidence."""
        with self.lock:
            fields.setdefault("lastSampleAt", self.muse_diagnostics["lastSampleAt"])
            if event == "reader_state":
                self.muse_diagnostics["readerState"] = fields.get("state")
            elif event == "interruption":
                self.muse_diagnostics["lastInterruption"] = dict(at=time.time(), **fields)
                self.muse_diagnostics["interruptionCount"] += 1
            sink = self.muse_diagnostic_sink
        if sink:
            sink(event, **fields)

    def _reset_feature_baseline(self):
        self.series.clear()
        self.feature_history.clear()
        self.calibration = None
        self.calibration_samples = []
        self.channel_calibration_samples = []
        self.clean_time = UniqueCleanTime()
        self.selected_channels = ()
        self.calibration_identity = None

    def _clear_current_evidence(self):
        """A packet gap breaks causal evidence, not the wearer's reference."""
        self.series.clear()
        self.feature_history.clear()
        self.clean_time.pause()
        self.filtered.clear()
        self.filter.reset()
        self.available_channels = ()
        self.reject_reasons = []
        self.invalid_until = float("-inf")
        self.raw.clear()
        self.last_eval = 0.0
        self.channel_quality = {}

    def _clear_baseline(self, reason="baseline_reset"):
        had_evidence = bool(self.calibration or self.raw or self.series or self.clean_time.seconds)
        if had_evidence:
            self.muse_diagnostics["baselineInvalidation"] = dict(at=time.time(), reason=reason)
            if self.muse_diagnostic_sink:
                self.muse_diagnostic_sink("baseline_invalidated", reason=reason,
                                          lastSampleAt=self.muse_diagnostics["lastSampleAt"])
        self._reset_feature_baseline()
        self._clear_current_evidence()

    def _identity(self):
        """Contact quality is transient; source, decoding and feature setup are not."""
        m = self.stream_metadata
        return (self.source, self.device_id, m.get("sourceId"), m.get("sampleRate"),
                m.get("units"), tuple(m.get("channelLabels", ())),
                tuple(m.get("excludedLabels", ())), m.get("valid"),
                EEG_RATE, EEG_WINDOW_S, ARTIFACT_UV, self.calibration_seconds)

    def _check_identity(self):
        if self.calibration_identity is not None and self.calibration_identity != self._identity():
            self._clear_baseline("source_identity_changed")
            self.calibration_started = None
            self.quality_error = "EEG source or calibration setup changed; a fresh baseline is required."

    def begin_calibration(self, seconds=60.0):
        with self.lock:
            self._clear_baseline("calibration_requested")
            self.calibration_seconds = seconds
            self.calibration_started = time.time()
            self.quality_error = "Collecting a fresh 2-second EEG window."

    def connected(self, identity, metadata=None):
        with self.lock:
            self.acquisition_phase = "waiting_for_samples"
            self.device_id = identity
            self.stream_metadata = metadata or muse_metadata(None, identity)
            self.metadata_error = not self.stream_metadata["valid"]
            self.connection_error = None
            self._clear_baseline("outlet_reconnected")
            self.last_sample_at = None
            self.calibration_started = time.time()
            self.quality_error = "Muse LSL outlet found; waiting for actual EEG samples."

    def reset_calibration(self):
        with self.lock:
            self._clear_baseline("calibration_reset")
            self.calibration_started = None
            self.quality_error = "EEG baseline reset; waiting for fresh samples."

    def disconnected(self):
        with self.lock:
            self.acquisition_phase = "disconnected"
            self.device_id = None
            self._clear_baseline("outlet_disconnected")
            self.last_sample_at = None
            self.calibration_started = None
            self.quality_error = "Muse EEG stream disconnected; waiting to reconnect."

    def push(self, samples, stamps):
        with self.lock:
            self._check_identity()
            if len(samples) != len(stamps):
                self.input_diagnostics["mismatchedChunks"] += 1
                self.quality_error = "Mismatched EEG samples and timestamps."
                self.reject_reasons = ["invalid_input"]
                self.invalid_until = (self.last_sample_at or time.time()) + EEG_WINDOW_S
                return
            for s, t in zip(samples, stamps):
                try:
                    x = np.asarray(s[:4], dtype=float)
                    t = float(t)
                    valid = len(x) == 4 and np.isfinite(x).all() and math.isfinite(t)
                except (TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    self.input_diagnostics["invalidSamples"] += 1
                    self.quality_error = "Invalid EEG samples."
                    self.reject_reasons = ["invalid_input"]
                    self.invalid_until = (self.last_sample_at or time.time()) + EEG_WINDOW_S
                    continue
                if self.last_sample_at is not None and t <= self.last_sample_at:
                    key = "duplicateSamples" if t == self.last_sample_at else "outOfOrderSamples"
                    self.input_diagnostics[key] += 1
                    if self.raw and t == self.raw[-1][0] and not np.array_equal(x, self.raw[-1][1]):
                        self.input_diagnostics["conflictingSamples"] += 1
                        self.invalid_until = self.last_sample_at + EEG_WINDOW_S
                        self.quality_error = "Conflicting EEG samples at one timestamp."
                        self.reject_reasons = ["invalid_input"]
                    continue
                if self.last_sample_at is not None and t - self.last_sample_at > MAX_INTERVAL_S:
                    interval = t - self.last_sample_at
                    self.input_diagnostics["gapCount"] += 1
                    self.input_diagnostics["missingSamples"] += max(1, round(interval * EEG_RATE) - 1)
                    self.input_diagnostics["lastGapSeconds"] = round(interval - 1 / EEG_RATE, 6)
                    # Brief packet loss requires fresh causal support. True raw
                    # silence retains the existing reconnect invalidation policy,
                    # even if nobody polled status during the interruption.
                    if interval >= EEG_FRESH_S:
                        self._clear_baseline("sample_gap_exceeded_freshness")
                        self.calibration_started = t
                        self.quality_error = "EEG samples stopped; collecting a fresh baseline."
                    else:
                        self._clear_current_evidence()
                        self.quality_error = "EEG sample gap; waiting for a fresh clean window."
                    self.reject_reasons = ["sample_gap"]
                if self.device_id and self.calibration is None and self.calibration_started is None:
                    self.calibration_started = t
                self.last_sample_at = t
                self.muse_diagnostics["lastSampleAt"] = t
                self.samples_received += 1
                self.connection_error = None
                if self.source == "muse":
                    self.state = "Muse EEG samples arriving"
                self.raw.append((t, x))
                self.filtered.append(self.filter.process(x[None, :])[0])
                if t - self.last_eval >= EEG_STEP_S:
                    self.last_eval = t
                    self._evaluate(t)

    def _evaluate(self, t):
        n = int(EEG_RATE * EEG_WINDOW_S)
        if len(self.raw) < n:
            return
        rows = list(self.raw)[-n:]
        window = np.array([s for _, s in rows], dtype=float)
        filtered = np.array(list(self.filtered)[-n:], dtype=float)
        stamps = np.array([tt for tt, _ in rows])
        self.channel_quality = channel_diagnostics(window, filtered, stamps, ARTIFACT_UV)
        available = tuple(i for i, name in enumerate(EEG_CHANNELS) if self.channel_quality[name]["usable"])
        self.available_channels = available
        # Lock the reference's channel mixture on the first accepted window.
        # Contact loss must not substitute a new mixture or erase calibration.
        selected = self.selected_channels or available
        reasons = []
        missing = [i for i in selected if i not in available]
        if missing or len(available) < MIN_CHANNELS:
            rejected = missing or range(len(EEG_CHANNELS))
            reasons = list(dict.fromkeys(reason for i in rejected
                for reason in self.channel_quality[EEG_CHANNELS[i]]["rejectReasons"]))
        if len(available) < MIN_CHANNELS:
            reasons.append("insufficient_channels")
        if t < self.invalid_until: reasons.append("invalid_input")
        if self.metadata_error: reasons.append("metadata")
        self.reject_reasons = reasons
        artifact = bool(reasons)
        self.quality_error = "; ".join(REASONS[reason] for reason in reasons)
        # Unusable rows retain a finite placeholder only for tuple compatibility;
        # artifact=True excludes them from every consumer and from the baseline.
        engagement, z = 0.0, 0.0
        if artifact:
            self.clean_time.pause()
        else:
            if not self.selected_channels:
                self.selected_channels = selected
                self.calibration_identity = self._identity()
            ratios = [band_powers(filtered[:, i:i+1]) for i in selected]
            values = np.array([bp["beta"] / max(bp["alpha"] + bp["theta"], 1e-9) for bp in ratios])
            engagement = float(np.median(values))
            if self.calibration is None:
                self.calibration_samples.append((t, engagement))
                self.channel_calibration_samples.append(values)
                clean_seconds = self.clean_time.add(stamps[0], t)
                if clean_seconds + 1e-6 >= self.calibration_seconds:
                    per_channel = np.array(self.channel_calibration_samples)
                    medians = np.median(per_channel, axis=0)
                    scales = np.maximum(np.median(np.abs(per_channel - medians), axis=0) * 1.4826, 1e-6)
                    median = float(np.median([value for _, value in self.calibration_samples]))
                    self.calibration = dict(deviceId=self.device_id, median=median,
                        scale=max(float(np.median(np.abs(np.array([v for _, v in self.calibration_samples]) - median))) * 1.4826, 1e-6),
                        channels=[EEG_CHANNELS[i] for i in selected],
                        channelMedians=medians.tolist(), channelScales=scales.tolist(),
                        cleanSeconds=round(clean_seconds, 6), calibratedAt=t)
                    self.calibration_started = None
            if self.calibration:
                z = float(np.median((values - self.calibration["channelMedians"]) / self.calibration["channelScales"]))
            else:
                base = [v for tt, v in self.feature_history if t - tt <= BASELINE_S]
                if len(base) >= 8:
                    medians = np.median(base, axis=0)
                    scales = np.maximum(np.median(np.abs(np.array(base) - medians), axis=0) * 1.4826, 1e-6)
                    z = float(np.median((values - medians) / scales))
            self.feature_history.append((t, values))
        self.series.append((t, engagement, float(np.clip(z, -5, 5)),
                            artifact or self.source == "muse" and self.calibration is None))

    def window(self, t0, t1):
        with self.lock:
            return [p for p in self.series if t0 <= p[0] <= t1]

    def latest(self):
        with self.lock:
            return self.series[-1] if self.series else None

    def status(self):
        now = time.time()
        with self.lock:
            self._check_identity()
            sample_age = None if self.last_sample_at is None else now - self.last_sample_at
            live = bool(sample_age is not None and -0.5 <= sample_age < EEG_FRESH_S)
            if sample_age is not None and sample_age >= EEG_FRESH_S:
                self._clear_baseline("samples_stale")
                self.calibration_started = None
                self.quality_error = "EEG samples stopped; a fresh clean baseline is required after reconnection."
            recent = [p for p in self.series if now - EEG_FRESH_S <= p[0] <= now + 0.5]
            usable = bool(live and recent and not recent[-1][3] and not self.quality_error
                          and (self.calibration or self.source == "sim"))
            confidence = (sum(not row[3] for row in recent) / len(recent) * len(self.selected_channels) / 4) if usable else 0.0
            state = ("simulated" if self.source == "sim" else "error" if self.connection_error else
                     "poor_signal" if live and self.channel_quality and self.quality_error else "streaming" if live else
                     "disconnected" if self.acquisition_phase == "disconnected" else
                     "stale" if sample_age is not None else "connecting" if self.device_id or self.source == "muse" else "disconnected")
            sampling = ("ready" if usable else "poor_signal" if state == "poor_signal" else
                        "recovering" if live and self.calibration else
                        "calibrating" if live and recent else "warming_up" if live else
                        "stale" if sample_age is not None else "waiting_for_samples")
            result = dict(source=self.source, state=self.connection_error or self.state, live=live, deviceId=self.device_id,
                    confidence=round(confidence, 3),
                    connectionState=state, modeLabel="EEG physiological observations available" if confidence else "EEG unavailable — gaze-only mode",
                    interpretation="Exploratory beta/(alpha+theta); not a validated emotion measure; cause and valence unknown",
                    calibrated=bool(self.calibration) and live, qualityError=self.quality_error,
                    calibrationRetained=bool(self.calibration), signalReady=bool(usable and confidence >= MIN_CONFIDENCE),
                    cleanSeconds=round(min(self.clean_time.seconds, self.calibration_seconds), 3),
                    targetSeconds=self.calibration_seconds, outletAvailable=bool(self.device_id),
                    samplingState=sampling, samplesReceived=self.samples_received,
                    acquisitionPhase=self.acquisition_phase,
                    acquisitionDiagnostics=deepcopy(self.muse_diagnostics),
                    lastSampleAt=self.last_sample_at, sampleAgeSeconds=None if sample_age is None else round(sample_age, 3),
                    channelQuality=dict(self.channel_quality), rejectReasons=list(self.reject_reasons),
                    selectedChannels=[EEG_CHANNELS[i] for i in self.selected_channels],
                    availableChannels=[EEG_CHANNELS[i] for i in self.available_channels],
                    qualityWarning=(f"{len(self.selected_channels)} of 4 clean channels; reduced confidence" if usable and len(self.selected_channels) < 4 else ""),
                    channelConfidenceCeiling=round(len(set(self.selected_channels) & set(self.available_channels)) / 4, 3),
                    excludedChannels=[name for i, name in enumerate(EEG_CHANNELS) if i not in self.available_channels],
                    streamMetadata=dict(self.stream_metadata), inputDiagnostics=dict(self.input_diagnostics),
                    featureWindowSeconds=EEG_WINDOW_S,
                    effectiveHistorySeconds=round(min(EEG_WINDOW_S, self.raw[-1][0] - self.raw[0][0] + 1 / EEG_RATE), 6) if self.raw else 0.0,
                    featureAgeSeconds=round(now - self.series[-1][0], 3) if self.series else None,
                    filter=dict(type="causal Butterworth SOS", bandHz=[1, 40], order=4,
                                notchHz=None, warmupSeconds=EEG_WINDOW_S, preservesRawGates=True),
                    qualityVersion="raw-validity-causal-sos-v1", cleanTimeMethod="unique accepted time intervals after first full clean window")
            # Presentation only: the feed still owns raw acceptance, confidence,
            # channel locking and accumulation. Never mutate the saved baseline.
            quality = dict(result, calibration=self.calibration)
            result.update(channel_eligibility(quality))
            result["channelPolicy"] = result["channel_policy"]
            if self.source == "muse":
                result["signalReady"] = bool(usable and quality_failure(quality) is None)
            if result["reduced_redundancy"]:
                result["qualityWarning"] = "Experimental two-clean-channel policy; reduced redundancy"
            return result


def run_muse(feed, stop):
    """Thread: resolve the muselsl EEG stream and push 4 channels forever."""
    feed.source = "muse"
    feed.muse_event("reader_state", state="running")
    try:
        import pylsl
        _consume_muse(feed, stop, pylsl)
    except Exception as error:
        feed.disconnected()
        feed.state = f"Muse connection error: {error}"
        feed.connection_error = feed.state
        feed.muse_event("interruption", reason="reader_error", detail=str(error)[:400])
    finally:
        feed.muse_event("reader_state", state="stopped" if stop.is_set() else "failed")


def _muse_clock_correction(inlet, stop, timeouts, budget):
    """Keep liblsl's estimator alive while waiting within a cancellable budget."""
    deadline = time.monotonic() + budget
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout = min(EEG_CLOCK_POLL_S, remaining)
        attempt_end = time.monotonic() + timeout
        try:
            correction = inlet.time_correction(timeout=timeout)
            return None if stop.is_set() or time.monotonic() > deadline else correction
        except timeouts:
            # Normally liblsl used the whole timeout. Also bound retries when a
            # provider returns early, without delaying cancellation.
            stop.wait(max(0.0, min(deadline, attempt_end) - time.monotonic()))
    return None


def _consume_muse(feed, stop, pylsl):
    address = os.getenv("GOZ_MUSE_ADDRESS", "").strip()
    pinned = "Muse" + address if address else None
    # Current pylsl exposes these RuntimeError subclasses in util, not at root.
    # Keep root support for older versions; never catch all RuntimeError here.
    providers = (pylsl, getattr(pylsl, "util", None))
    clock_timeouts = (TimeoutError,) + tuple(
        error for provider in providers
        if isinstance(error := getattr(provider, "TimeoutError", None), type))
    recoverable = (TimeoutError, OSError) + tuple(
        error for provider in providers for name in ("LostError", "TimeoutError")
        if isinstance(error := getattr(provider, name, None), type))
    retries = 0
    last_reason = None
    diagnostic = getattr(feed, "muse_event", lambda *args, **kwargs: None)

    def report(reason, detail):
        nonlocal last_reason
        if reason != last_reason:
            diagnostic("interruption", reason=reason, detail=detail)
            last_reason = reason

    def backoff():
        nonlocal retries
        if stop.is_set():
            return
        delay = min(4.0, .25 * 2 ** min(retries, 4))
        retries += 1
        if hasattr(feed, "muse_diagnostics"):
            with feed.lock:
                feed.muse_diagnostics["retryCount"] += 1
                feed.muse_diagnostics["retryDelaySeconds"] = delay
        # Short slices make cancellation responsive even for fake providers.
        while delay > 0 and not stop.is_set():
            step = min(.25, delay)
            stop.wait(step)
            delay -= step
    while not stop.is_set():
        feed.acquisition_phase = "resolving"
        feed.state = "searching for `muselsl stream`"
        try:
            streams = [s for s in pylsl.resolve_byprop("type", "EEG", timeout=1)
                       if s.name() == "Muse" and s.channel_count() == 5 and s.nominal_srate() == EEG_RATE
                       and (pinned is None or s.source_id() == pinned)]
        except recoverable as error:
            feed.disconnected()
            feed.state = f"Muse discovery interrupted: {error}; reconnecting"
            report("discovery_timeout", str(error)[:400])
            backoff()
            continue
        if stop.is_set():
            break
        if not streams:
            feed.quality_error = "No Muse LSL EEG outlet. Start the direct Muse bridge or retry sensor setup."
            report("no_outlet", feed.quality_error)
            backoff()
            continue
        if len(streams) > 1:
            raise ValueError("Multiple Muse streams found. Set GOZ_MUSE_ADDRESS to select your Muse 2.")
        pinned = streams[0].source_id()
        inlet = None
        try:
            inlet = pylsl.StreamInlet(streams[0], max_buflen=5, max_chunklen=12, recover=False)
            if stop.is_set():
                break
            metadata_info = inlet.info(timeout=.5) if hasattr(inlet, "info") else streams[0]
            feed.connected(pinned)
            metadata = muse_metadata(metadata_info, pinned)
            feed.metadata_error = not metadata["valid"]
            feed.stream_metadata = metadata
            feed.state = "Muse LSL outlet advertised; waiting for EEG samples"
            diagnostic("outlet_found")
            last_reason = None
            idle = time.time()
            initial_clock_attempt = True
            pending = None
            while not stop.is_set():
                chunk, stamps = pending if pending is not None else inlet.pull_chunk(timeout=0.5)
                if stop.is_set():
                    break
                if stamps:
                    now = time.time()
                    # Clock correction applies only to LSL-clock timestamps. Unix-clock
                    # external bridges are already in the application's wall-clock domain.
                    if abs(stamps[-1] - now) < 60:
                        converted = stamps
                    else:
                        if initial_clock_attempt:
                            feed.acquisition_phase = "synchronizing_clock"
                        budget = EEG_CLOCK_INITIAL_S if initial_clock_attempt else EEG_CLOCK_RETRY_S
                        initial_clock_attempt = False
                        pending = (chunk, stamps)
                        # Retain this chunk through bootstrap; a clock-only timeout
                        # must not recreate the inlet or reset its estimator.
                        correction = _muse_clock_correction(inlet, stop, clock_timeouts, budget)
                        if stop.is_set():
                            break
                        if correction is None:
                            feed.acquisition_phase = "waiting_for_clock"
                            feed.state = "Muse clock synchronization unavailable; retrying on the same inlet"
                            feed.quality_error = "Muse clock synchronization unavailable; EEG timestamps cannot be verified."
                            report("clock_unavailable", feed.quality_error)
                            idle = time.time()  # Data arrived; this is not a lost stream.
                            continue
                        pending = None
                        now = time.time()
                        offset = now - pylsl.local_clock()
                        converted = [s + correction + offset for s in stamps]
                    if stop.is_set():
                        break
                    fresh = [(c, s) for c, s in zip(chunk, converted)
                             if math.isfinite(s) and -0.5 <= now - s < EEG_FRESH_S]
                    if len(chunk) != len(converted) or not fresh:
                        feed.quality_error = "Stale or invalid Muse sample timestamps; waiting for fresh EEG."
                    else:
                        before = getattr(feed, "samples_received", None)
                        if getattr(feed, "quality_error", "").startswith((
                                "Muse clock synchronization unavailable", "Stale or invalid Muse sample timestamps")):
                            feed.quality_error = "Collecting a fresh 2-second EEG window."
                        feed.push([c[:4] for c, _ in fresh], [s for _, s in fresh])
                        if before is None or feed.samples_received > before:
                            feed.acquisition_phase = "streaming"
                            idle = now
                            retries = 0
                            last_reason = None
                if time.time() - idle > 5:
                    feed.state = "Muse LSL outlet has stopped delivering fresh EEG samples"
                    report("silent_outlet", feed.state)
                    break
        except recoverable as error:
            feed.state = f"Muse stream interrupted: {error}; reconnecting"
            report("stream_interrupted", str(error)[:400])
        finally:
            try:
                if inlet is not None:
                    inlet.close_stream()
            finally:
                feed.disconnected()
        # Avoid busy reconnection to an advertised outlet that no longer sends data.
        backoff()


class Simulator:
    """Rehearsal signals: gaze drifts between characters, lingering on the
    favourite; EEG engagement rises ~0.6 s after gaze lands on it."""

    def __init__(self, session, gaze, eeg, favourite_index=1, bias=0.7, seed=None):
        self.session, self.gaze, self.eeg = session, gaze, eeg
        self.favourite_index, self.bias = favourite_index, bias
        self.target, self.until, self.phase = None, 0.0, 0.0
        self.seed = seed if seed is not None else os.getenv("GOZ_SIM_SEED")
        self.random = random.Random(self.seed)
        self.noise = np.random.default_rng(None if self.seed is None else int.from_bytes(str(self.seed).encode(), "little") % (2**32))

    def _pick(self, boxes):
        if not boxes or self.random.random() < 0.12:
            return None
        names = list(boxes)
        fav = names[self.favourite_index] if self.favourite_index < len(names) else None
        return fav if fav and self.random.random() < self.bias else self.random.choice(names)

    async def run(self, simulate_gaze, simulate_eeg):
        if simulate_gaze:
            self.gaze.source = "sim"
        if simulate_eeg:
            self.eeg.source, self.eeg.state = "sim", "simulated"
        on_fav_since, last_eeg = None, time.time() - 0.25
        while True:
            now = time.time()
            boxes, rect = self.session.live_boxes()
            if simulate_gaze and rect:
                if now > self.until or (self.target and self.target not in boxes):
                    self.target, self.until = self._pick(boxes), now + self.random.uniform(0.8, 2.5)
                if self.target in boxes:
                    b = boxes[self.target]
                    nx, ny = (b[0] + b[2]) / 2, b[1] + 0.3 * (b[3] - b[1])
                else:
                    nx, ny = self.random.uniform(0.1, 0.9), self.random.uniform(0.1, 0.9)
                nx += self.random.gauss(0, 0.02)
                ny += self.random.gauss(0, 0.02)
                blink = self.random.random() < 0.01
                self.gaze.add(dict(t=now, x=rect["x"] + nx * rect["w"], y=rect["y"] + ny * rect["h"],
                                   valid=not blink, blink=blink, yaw=self.random.gauss(0, 3),
                                   pitch=self.random.gauss(0, 3), face=True))
            names = list(boxes)
            fav = names[self.favourite_index] if self.favourite_index < len(names) else None
            looking = fav is not None and self.session.live_target() == fav
            on_fav_since = (on_fav_since or now) if looking else None
            if simulate_eeg and now - last_eeg >= 0.25:
                if now - last_eeg > 3:
                    last_eeg = now - 0.25
                last_eeg += 0.25
                boost = 1.6 if on_fav_since and now - on_fav_since > 0.6 else 0.0
                self.phase += 0.25
                t = np.arange(int(EEG_RATE * 0.25)) / EEG_RATE + self.phase
                beta = (1.0 + boost) * np.sin(2 * math.pi * 20 * t)
                alpha = 2.0 * np.sin(2 * math.pi * 10 * t)
                theta = 1.5 * np.sin(2 * math.pi * 6 * t)
                chans = np.stack([beta + alpha + theta + self.noise.normal(0, 0.6, len(t)) for _ in range(4)], axis=1) * 10
                stamps = list(last_eeg - 0.25 + np.arange(len(t)) / EEG_RATE)
                self.eeg.push(chans.tolist(), stamps)
            await asyncio.sleep(1 / 30 if simulate_gaze else 0.25)
