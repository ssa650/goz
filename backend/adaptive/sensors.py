"""Live viewer signals on one wall clock (unix seconds).

Gaze: `gazekit stream` UDP datagrams (gazekit docs/STREAM_PROTOCOL.md).
EEG:  Muse 2 via `muselsl stream` (Lab Streaming Layer), reduced to a
      beta/(alpha+theta) engagement index z-scored against a rolling baseline.
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

import numpy as np

GAZE_PORT = 5590
EEG_RATE = 256
EEG_WINDOW_S = 2.0
EEG_STEP_S = 0.25
BASELINE_S = 60.0
ARTIFACT_UV = 150.0
EEG_FRESH_S = 3.0
EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


class GazeFeed:
    """Ring buffer of gaze samples; filled by UDP or the simulator."""

    def __init__(self, keep_s=180):
        self.samples = deque(maxlen=keep_s * 40)
        self.source, self.transport = "waiting", None
        self.last_rx = 0.0
        self.expected_setup_id = None

    def packet(self, sample):
        if (not isinstance(sample, dict) or any(type(sample.get(k)) not in (float, int)
                or not math.isfinite(sample[k]) for k in ("t", "x", "y"))
                or self.expected_setup_id and sample.get("setupId") != self.expected_setup_id):
            return False
        if any(key in sample and type(sample[key]) is not bool for key in ("valid", "face", "blink")):
            return False
        if any(k in sample and (type(sample[k]) not in (float, int) or not math.isfinite(sample[k]))
               for k in ("yaw", "pitch", "confidence")):
            return False
        if abs(sample["t"] - time.time()) > 5 or self.samples and sample["t"] <= self.samples[-1]["t"]:
            return False
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
                    coordinateSpace="screen-points", captureClock="unix-seconds")

    async def listen(self, port=GAZE_PORT):
        feed = self

        class Protocol(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                try:
                    sample = json.loads(data)
                    if feed.packet(sample):
                        feed.source = "gazekit"
                except ValueError:
                    pass

        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(Protocol, local_addr=("127.0.0.1", port))

    def close(self):
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
    """Engagement series [(t, engagement, z, artifact)] from raw EEG chunks."""

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

    def _clear_baseline(self):
        self.raw.clear()
        self.series.clear()
        self.last_eval = 0.0
        self.calibration = None
        self.calibration_samples = []
        self.channel_quality = {}

    def begin_calibration(self, seconds=60.0):
        with self.lock:
            self._clear_baseline()
            self.calibration_seconds = seconds
            self.calibration_started = time.time()
            self.quality_error = "Collecting a fresh 2-second EEG window."

    def connected(self, identity):
        with self.lock:
            self.device_id = identity
            self.connection_error = None
            self._clear_baseline()
            self.last_sample_at = None
            self.calibration_started = time.time()
            self.quality_error = "Muse LSL outlet found; waiting for actual EEG samples."

    def reset_calibration(self):
        with self.lock:
            self._clear_baseline()
            self.calibration_started = None
            self.quality_error = "EEG baseline reset; waiting for fresh samples."

    def disconnected(self):
        with self.lock:
            self.device_id = None
            self._clear_baseline()
            self.last_sample_at = None
            self.calibration_started = None
            self.quality_error = "Muse EEG stream disconnected; waiting to reconnect."

    def push(self, samples, stamps):
        with self.lock:
            for s, t in zip(samples, stamps):
                if len(s) < 4 or not np.isfinite(s[:4]).all() or not math.isfinite(t):
                    self.quality_error = "Invalid EEG samples."
                    continue
                t = float(t)
                if self.raw and t <= self.raw[-1][0]:
                    continue
                if self.last_sample_at is not None and t <= self.last_sample_at:
                    continue
                if self.last_sample_at is not None and t - self.last_sample_at > 1:
                    self._clear_baseline()
                    self.calibration_started = t
                    self.quality_error = "EEG sample gap; collecting a fresh baseline."
                if self.device_id and self.calibration is None and self.calibration_started is None:
                    self.calibration_started = t
                self.last_sample_at = t
                self.samples_received += 1
                self.connection_error = None
                if self.source == "muse":
                    self.state = "Muse EEG samples arriving"
                self.raw.append((t, s[:4]))
                if t - self.last_eval >= EEG_STEP_S:
                    self.last_eval = t
                    self._evaluate(t)

    def _evaluate(self, t):
        n = int(EEG_RATE * EEG_WINDOW_S)
        if len(self.raw) < n:
            return
        window = np.array([s for _, s in list(self.raw)[-n:]], dtype=float)
        spread, variation = np.ptp(window, axis=0), np.std(window, axis=0)
        spectrum = np.abs(np.fft.rfft((window - window.mean(axis=0)) * np.hanning(n)[:, None], axis=0)) ** 2
        frequencies = np.fft.rfftfreq(n, 1 / EEG_RATE)
        line_mask = ((frequencies >= 49) & (frequencies <= 51)) | ((frequencies >= 59) & (frequencies <= 61))
        line_fraction = spectrum[line_mask].sum(axis=0) / np.maximum(spectrum[frequencies >= 1].sum(axis=0), 1e-9)
        self.channel_quality = {name: dict(peakToPeakUV=round(float(spread[i]), 2),
            stdUV=round(float(variation[i]), 2), peakAbsUV=round(float(np.abs(window[:, i]).max()), 2),
            lineNoiseFraction=round(float(line_fraction[i]), 3)) for i, name in enumerate(EEG_CHANNELS)}
        problems = []
        if not 1.8 <= t - self.raw[-n][0] <= 2.2: problems.append("EEG packets missing or irregular")
        if spread.max() > ARTIFACT_UV: problems.append("movement or poor contact")
        if variation.min() < 0.05: problems.append("flat sensor channel")
        if np.abs(window).max() >= 950: problems.append("clipped sensor channel")
        if line_fraction.max() > 0.35: problems.append("50/60 Hz interference")
        artifact = bool(problems)
        self.quality_error = "; ".join(problems) if problems else ""
        bp = band_powers(window)
        engagement = bp["beta"] / max(bp["alpha"] + bp["theta"], 1e-9)
        if (self.calibration_started is not None and t >= self.calibration_started + EEG_WINDOW_S
                and not artifact and self.calibration is None):
            self.calibration_samples.append((t, engagement))
            if len(self.calibration_samples) * EEG_STEP_S >= self.calibration_seconds:
                values = np.array([value for _, value in self.calibration_samples])
                median = float(np.median(values))
                scale = max(float(np.median(np.abs(values - median))) * 1.4826, 1e-6)
                self.calibration = dict(deviceId=self.device_id, median=median, scale=scale,
                                        cleanSeconds=round(len(values) * EEG_STEP_S, 2), calibratedAt=t)
                self.calibration_started = None
        base = [e for (tt, e, _, a) in self.series if not a and t - tt <= BASELINE_S]
        if self.calibration:
            z = (engagement - self.calibration["median"]) / self.calibration["scale"]
        elif len(base) >= 8:
            median = float(np.median(base))
            mad = float(np.median(np.abs(np.array(base) - median))) * 1.4826 or 1e-6
            z = (engagement - median) / mad
        else:
            z = 0.0
        # Before a real-device baseline is ready, these rows are not response evidence.
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
            sample_age = None if self.last_sample_at is None else now - self.last_sample_at
            live = bool(sample_age is not None and -0.5 <= sample_age < EEG_FRESH_S)
            if sample_age is not None and sample_age >= EEG_FRESH_S:
                self._clear_baseline()
                self.calibration_started = None
                self.quality_error = "EEG samples stopped; a fresh clean baseline is required after reconnection."
            recent = [p for p in self.series if now - EEG_FRESH_S <= p[0] <= now + 0.5]
            usable = bool(live and recent and not recent[-1][3] and not self.quality_error
                          and (self.calibration or self.source == "sim"))
            confidence = sum(not row[3] for row in recent) / len(recent) if usable else 0.0
            state = ("simulated" if self.source == "sim" else "error" if self.connection_error else
                     "poor_signal" if live and self.channel_quality and self.quality_error else "streaming" if live else
                     "stale" if sample_age is not None else "connecting" if self.device_id or self.source == "muse" else "disconnected")
            sampling = ("ready" if usable else "poor_signal" if state == "poor_signal" else
                        "calibrating" if live and recent else "warming_up" if live else
                        "stale" if sample_age is not None else "waiting_for_samples")
            return dict(source=self.source, state=self.connection_error or self.state, live=live, deviceId=self.device_id,
                    confidence=round(confidence, 3),
                    connectionState=state, modeLabel="EEG physiological observations available" if confidence else "EEG unavailable — gaze-only mode",
                    interpretation="Physiological variation; cause and valence unknown",
                    calibrated=bool(self.calibration) and live, qualityError=self.quality_error,
                    cleanSeconds=round(min(len(self.calibration_samples) * EEG_STEP_S, self.calibration_seconds), 1),
                    targetSeconds=self.calibration_seconds, outletAvailable=bool(self.device_id),
                    samplingState=sampling, samplesReceived=self.samples_received,
                    lastSampleAt=self.last_sample_at, sampleAgeSeconds=None if sample_age is None else round(sample_age, 3),
                    channelQuality=dict(self.channel_quality))


def run_muse(feed, stop):
    """Thread: resolve the muselsl EEG stream and push 4 channels forever."""
    feed.source = "muse"
    try:
        import pylsl
        _consume_muse(feed, stop, pylsl)
    except Exception as error:
        feed.disconnected()
        feed.state = f"Muse connection error: {error}"
        feed.connection_error = feed.state


def _consume_muse(feed, stop, pylsl):
    address = os.getenv("GOZ_MUSE_ADDRESS", "").strip()
    pinned = "Muse" + address if address else None
    # Current pylsl exposes these RuntimeError subclasses in util, not at root.
    # Keep root support for older versions; never catch all RuntimeError here.
    providers = (pylsl, getattr(pylsl, "util", None))
    recoverable = (TimeoutError, OSError) + tuple(
        error for provider in providers for name in ("LostError", "TimeoutError")
        if isinstance(error := getattr(provider, name, None), type))
    while not stop.is_set():
        feed.state = "searching for `muselsl stream`"
        try:
            streams = [s for s in pylsl.resolve_byprop("type", "EEG", timeout=1)
                       if s.name() == "Muse" and s.channel_count() == 5 and s.nominal_srate() == EEG_RATE
                       and (pinned is None or s.source_id() == pinned)]
        except recoverable as error:
            feed.disconnected()
            feed.state = f"Muse discovery interrupted: {error}; reconnecting"
            if not stop.is_set():
                stop.wait(0.25)
            continue
        if stop.is_set():
            break
        if not streams:
            feed.quality_error = "No Muse LSL EEG outlet. Start the direct Muse bridge or retry sensor setup."
            continue
        if len(streams) > 1:
            raise ValueError("Multiple Muse streams found. Set GOZ_MUSE_ADDRESS to select your Muse 2.")
        pinned = streams[0].source_id()
        inlet = None
        try:
            inlet = pylsl.StreamInlet(streams[0], max_buflen=5, max_chunklen=12, recover=False)
            feed.connected(pinned)
            feed.state = "Muse LSL outlet advertised; waiting for EEG samples"
            idle = time.time()
            while not stop.is_set():
                chunk, stamps = inlet.pull_chunk(timeout=0.5)
                if stamps:
                    now = time.time()
                    # Clock correction applies only to LSL-clock timestamps. Unix-clock
                    # external bridges are already in the application's wall-clock domain.
                    if abs(stamps[-1] - now) < 60:
                        converted = stamps
                    else:
                        correction = inlet.time_correction(timeout=0.5)
                        now = time.time()
                        offset = now - pylsl.local_clock()
                        converted = [s + correction + offset for s in stamps]
                    if len(chunk) != len(converted) or any(not math.isfinite(s) or abs(s - now) > EEG_FRESH_S for s in converted):
                        feed.quality_error = "Stale or invalid Muse sample timestamps; waiting for fresh EEG."
                    else:
                        before = getattr(feed, "samples_received", None)
                        feed.push([c[:4] for c in chunk], converted)
                        if before is None or feed.samples_received > before:
                            idle = now
                if time.time() - idle > 5:
                    feed.state = "Muse LSL outlet has stopped delivering fresh EEG samples"
                    break
        except recoverable as error:
            feed.state = f"Muse stream interrupted: {error}; reconnecting"
        finally:
            try:
                if inlet is not None:
                    inlet.close_stream()
            finally:
                feed.disconnected()
        # Avoid busy reconnection to an advertised outlet that no longer sends data.
        if not stop.is_set():
            stop.wait(0.25)


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
