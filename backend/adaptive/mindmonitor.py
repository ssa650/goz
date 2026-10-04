"""Mind Monitor OSC: contact-gated alpha/beta observations, EMA (0.2).

Protocol/example: https://github.com/Enigma644/MindMonitorPython
This physiological ratio is not an enjoyment, attention or focus measurement.
"""
import math
import os
import socket
import time
from collections import deque
import statistics

from .sensors import EegFeed

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
FRESH_S = 3.0


def destination_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Select the default interface without transmitting a packet.
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return "your Mac's Wi-Fi IP"


class MindMonitorFeed(EegFeed):
    def __init__(self):
        super().__init__()
        self.source = "mindmonitor"
        self.state = "Waiting for Mind Monitor OSC."
        self.port = int(os.getenv("GOZ_MINDMONITOR_PORT", "5000"))
        self.destination = destination_ip()
        self.peer = os.getenv("GOZ_MINDMONITOR_PHONE_IP", "").strip() or None
        self.pinned_peer = self.peer
        self.last_rx = 0.0
        self.packets_received = 0
        self.raw_samples_received = 0
        self.contacts = [4] * 4
        self.contact_at = 0.0
        self.bands = {}
        self.used_at = 0.0
        self.active_channels = ()
        self.ema = None
        self.readings = 0
        self.baseline = []
        self.raw_recent = deque(maxlen=64)
        self.raw_at = 0.0
        self.artifact_until = 0.0
        self.label = "Waiting"
        self.quality_error = "Waiting for Mind Monitor OSC; connect Muse in the phone app and enable streaming."

    def _reset(self):
        self.bands.clear()
        self.used_at = 0.0
        self.ema = None
        self.readings = 0
        self.baseline = []
        self.calibration = None
        self.active_channels = ()
        self.label = "Waiting"
        self.series.clear()

    def reset_calibration(self):
        with self.lock:
            self._reset()
            self.series.clear()
            self.contact_at = 0.0

    def receive(self, sender, address, *values):
        now = time.time()
        with self.lock:
            if self.peer and not self.pinned_peer and now - self.last_rx > FRESH_S:
                self.peer = None  # Permit a phone to reconnect after a DHCP change.
                self._reset()
                self.device_id = None
                self.contacts, self.contact_at = [4] * 4, 0.0
                self.raw_recent.clear()
                self.raw_at, self.artifact_until = 0.0, 0.0
            if self.peer and sender != self.peer:
                return
            if address == "/muse/elements/horseshoe":
                if len(values) != 4 or any(type(v) not in (int, float) or v not in (1, 2, 3, 4) for v in values):
                    return
                if now - self.contact_at > FRESH_S or tuple(v == 1 for v in values) != tuple(v == 1 for v in self.contacts):
                    self._reset()
                self.peer = sender
                self.device_id = "MindMonitor@" + sender
                self.contacts = list(values)
                self.contact_at = now
                self.state = "Mind Monitor contact packets received; waiting for fresh brainwaves"
            elif address == "/muse/eeg":
                if len(values) < 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values[:4]):
                    return
                self.raw_at = now
                self.last_rx = now
                self.packets_received += 1
                self.raw_samples_received += 1
                self.raw_recent.append(values[:4])
                good = [i for i, contact in enumerate(self.contacts) if contact == 1]
                bad = any(abs(values[i]) >= 950 for i in good)
                if len(self.raw_recent) >= 16:
                    bad = bad or any(max(row[i] for row in self.raw_recent) - min(row[i] for row in self.raw_recent) > 150 for i in good)
                if bad:
                    self.artifact_until = now + 2
                if now < self.artifact_until:
                    self._reset()
                    self.quality_error = "EEG artifact: clipped signal or movement. Hold still and adjust contacts."
                return
            elif address in ("/muse/elements/blink", "/muse/elements/jaw_clench"):
                if values and values[0] == 1:
                    self.last_rx = now
                    self.packets_received += 1
                    self.artifact_until = now + 2
                    self._reset()
                    self.quality_error = "Blink/jaw movement artifact; waiting for a clean EEG window."
                return
            elif address in ("/muse/elements/alpha_absolute", "/muse/elements/beta_absolute"):
                if len(values) not in (1, 4):
                    self.quality_error = "Set OSC Stream Brainwaves to All Values or Average Only in Mind Monitor."
                    return
                # Non-finite values on bad electrodes are excluded below.
                if any(type(v) not in (int, float) for v in values):
                    return
                self.bands[address.split("/")[-1]] = (now, values)
            else:
                return
            self.last_rx = now
            self.packets_received += 1
            self._evaluate_osc(now)

    def _evaluate_osc(self, now):
        if self.state.startswith("Muse connection error:"):
            self.calibration = None
            self.quality_error = self.state
            return
        if now < self.artifact_until:
            self.quality_error = "EEG artifact: clipped signal or movement. Hold still and adjust contacts."
            return
        if now - self.contact_at > FRESH_S:
            self._reset()
            self.quality_error = (f"Waiting for OSC at {self.destination}:{self.port}. Connect Muse in Mind Monitor and start OSC streaming."
                                  if self.device_id is None else "No fresh contact status. Enable horseshoe/contact data in Mind Monitor.")
            return
        good = [i for i, value in enumerate(self.contacts) if value == 1]
        if not good:
            self._reset()
            self.quality_error = "No good contacts. Adjust forehead and ear sensors until at least one is good."
            return
        if any(band not in self.bands for band in ("alpha_absolute", "beta_absolute")):
            self.quality_error = "Waiting for both alpha and beta. Enable OSC Stream Brainwaves in Mind Monitor."
            return
        alpha_at, alpha = self.bands["alpha_absolute"]
        beta_at, beta = self.bands["beta_absolute"]
        if now - min(alpha_at, beta_at) > FRESH_S:
            self._reset()
            self.quality_error = "Brainwave stream stopped. Enable OSC streaming in Mind Monitor."
            return
        if len(alpha) != len(beta):
            self.quality_error = "Alpha/beta formats differ. Use the same OSC Brainwaves setting."
            return
        if len(alpha) == 4:
            good = [i for i in good if math.isfinite(alpha[i]) and math.isfinite(beta[i])]
            if not good:
                self.quality_error = "No finite alpha/beta values on good contacts."
                return
            a = sum(alpha[i] for i in good) / len(good)
            b = sum(beta[i] for i in good) / len(good)
        else:
            # Mind Monitor's Average Only values already select good contacts.
            a, b = alpha[0], beta[0]
        if not math.isfinite(a) or not math.isfinite(b) or not -12 <= a - b <= 12:
            self.quality_error = "Invalid alpha/beta power values. Check Mind Monitor signal quality."
            return
        channels = tuple(CHANNELS[i] for i in good)
        if channels != self.active_channels:
            self.ema, self.readings, self.calibration = None, 0, None
            self.baseline = []
            self.series.clear()
            self.active_channels = channels
        self.quality_error = ""
        # Require a new pair, rather than reusing an old beta with new alpha.
        if min(alpha_at, beta_at) <= self.used_at:
            return
        self.used_at = max(alpha_at, beta_at)
        ratio = 10 ** (a - b)
        self.ema = ratio if self.ema is None else 0.2 * ratio + 0.8 * self.ema
        self.readings += 1
        log_ratio = -math.log10(max(self.ema, 1e-12))
        self.label = "Physiological ratio warming up"
        if self.calibration is None:
            self.baseline.append(log_ratio)
        if self.readings >= 10 and self.calibration is None:
            median = statistics.median(self.baseline)
            scale = max(.1, 1.4826 * statistics.median(abs(value-median) for value in self.baseline))
            self.calibration = dict(method="alpha-beta-relative-baseline", deviceId=self.device_id,
                                    median=median, scale=scale, emaAlpha=0.2,
                                    channels=list(channels), readyAt=now)
        index = 0.0 if not self.calibration else max(-5., min(5., (log_ratio-self.calibration["median"])/self.calibration["scale"]))
        if self.calibration:
            self.label = "Beta/alpha above baseline" if index > 1 else "Beta/alpha below baseline" if index < -1 else "Beta/alpha near baseline"
        self.state = "Mind Monitor streaming physiological observations"
        self.series.append((now, 1 / max(self.ema, 1e-12), index, self.calibration is None))

    def status(self):
        now = time.time()
        with self.lock:
            self._evaluate_osc(now)
            live = bool(self.series and now - self.series[-1][0] < FRESH_S and not self.quality_error)
            fresh_packets = self.last_rx > 0 and now - self.last_rx < FRESH_S
            state = ("error" if self.state.startswith("Muse connection error:") else "streaming" if live else
                     "poor_signal" if fresh_packets and self.quality_error else "connecting" if fresh_packets else
                     "stale" if self.device_id else "disconnected")
            return dict(source=self.source, state=self.state, live=live, deviceId=self.device_id,
                        confidence=round(len(self.active_channels) / 4, 3) if live and self.calibration else 0.0,
                        calibrated=bool(self.calibration) and live, qualityError=self.quality_error,
                        cleanSeconds=0, targetSeconds=0, startupSamples=min(self.readings, 10), targetSamples=10,
                        contacts=dict(zip(CHANNELS, self.contacts)), goodChannels=list(self.active_channels),
                        alphaBetaRatio=self.ema, label=self.label, method="alpha-beta-relative-baseline",
                        connectionState=state, modeLabel="EEG physiological observations available" if live and self.calibration else "EEG unavailable — gaze-only mode",
                        timestampSource="OSC arrival time (capture clock unavailable)",
                        artifactCoverage="raw clipping/movement and contacts" if now-self.raw_at < FRESH_S else "contacts only; enable raw EEG OSC for artifact checks",
                        interpretation="Physiological variation; cause and valence unknown",
                        packetsReceived=self.packets_received, rawSamplesReceived=self.raw_samples_received,
                        contactAgeSeconds=round(now-self.contact_at, 3) if self.contact_at else None,
                        alphaAgeSeconds=round(now-self.bands['alpha_absolute'][0], 3) if 'alpha_absolute' in self.bands else None,
                        betaAgeSeconds=round(now-self.bands['beta_absolute'][0], 3) if 'beta_absolute' in self.bands else None,
                        oscDestination=self.destination, oscPort=self.port)


def run_mindmonitor(feed, stop):
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    dispatcher = Dispatcher()
    def handler(client, address, *values):
        feed.receive(client[0], address, *values)
    for name in ("horseshoe", "alpha_absolute", "beta_absolute", "blink", "jaw_clench"):
        dispatcher.map("/muse/elements/" + name, handler, needs_reply_address=True)
    dispatcher.map("/muse/eeg", handler, needs_reply_address=True)
    try:
        with BlockingOSCUDPServer(("0.0.0.0", feed.port), dispatcher) as server:
            server.timeout = 0.25
            feed.port = server.server_address[1]
            feed.state = f"Listening for Mind Monitor OSC on UDP {feed.port}"
            while not stop.is_set():
                server.handle_request()
    except Exception as error:
        feed.state = f"Muse connection error: Mind Monitor OSC: {error}"
        feed.quality_error = feed.state
    finally:
        feed.reset_calibration()
