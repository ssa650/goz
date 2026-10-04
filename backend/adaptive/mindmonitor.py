"""Fable-style Mind Monitor OSC: good contacts, alpha/beta, EMA (0.2).

Protocol/example: https://github.com/Enigma644/MindMonitorPython
Thresholds: https://github.com/StiopaPopa/fable/blob/main/backend/eeg.py
This is a heuristic focus index, not a personal calibration or probability.
"""
import math
import os
import socket
import time

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
        self.contacts = [4] * 4
        self.contact_at = 0.0
        self.bands = {}
        self.used_at = 0.0
        self.active_channels = ()
        self.ema = None
        self.readings = 0
        self.label = "Waiting"
        self.quality_error = "Waiting for Mind Monitor OSC; connect Muse in the phone app and enable streaming."

    def _reset(self):
        self.bands.clear()
        self.ema = None
        self.readings = 0
        self.calibration = None
        self.active_channels = ()
        self.label = "Waiting"

    def reset_calibration(self):
        with self.lock:
            self._reset()
            self.series.clear()
            self.contact_at = 0.0

    def receive(self, sender, address, *values):
        now = time.time()
        with self.lock:
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
                self.state = "Mind Monitor connected"
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
            self._evaluate_osc(now)

    def _evaluate_osc(self, now):
        if self.state.startswith("Muse connection error:"):
            self.calibration = None
            self.quality_error = self.state
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
            self.active_channels = channels
        self.quality_error = ""
        # Require a new pair, rather than reusing an old beta with new alpha.
        if min(alpha_at, beta_at) <= self.used_at:
            return
        self.used_at = max(alpha_at, beta_at)
        ratio = 10 ** (a - b)
        self.ema = ratio if self.ema is None else 0.2 * ratio + 0.8 * self.ema
        self.readings += 1
        self.label = "Focusing" if self.ema < 1.4 else "Relaxing" if self.ema > 2 else "Neutral"
        # Map Fable's thresholds to +1 (focus) and -1 (relax) for fusion.
        index = max(-5.0, min(5.0, (1.7 - self.ema) / 0.3))
        self.series.append((now, 1 / max(self.ema, 1e-12), index, False))
        if self.readings >= 10 and self.calibration is None:
            self.calibration = dict(method="fable-alpha-beta-ema", deviceId=self.device_id,
                                    focusBelow=1.4, relaxAbove=2.0, emaAlpha=0.2,
                                    channels=list(channels), readyAt=now)

    def status(self):
        now = time.time()
        with self.lock:
            self._evaluate_osc(now)
            live = bool(self.series and now - self.series[-1][0] < FRESH_S and not self.quality_error)
            return dict(source=self.source, state=self.state, live=live, deviceId=self.device_id,
                        calibrated=bool(self.calibration) and live, qualityError=self.quality_error,
                        cleanSeconds=0, targetSeconds=0, startupSamples=min(self.readings, 10), targetSamples=10,
                        contacts=dict(zip(CHANNELS, self.contacts)), goodChannels=list(self.active_channels),
                        alphaBetaRatio=self.ema, label=self.label, method="fable-alpha-beta-ema",
                        oscDestination=self.destination, oscPort=self.port)


def run_mindmonitor(feed, stop):
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    dispatcher = Dispatcher()
    def handler(client, address, *values):
        feed.receive(client[0], address, *values)
    for name in ("horseshoe", "alpha_absolute", "beta_absolute"):
        dispatcher.map("/muse/elements/" + name, handler, needs_reply_address=True)
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
