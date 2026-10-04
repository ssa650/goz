import socket
import threading
import time

import pytest
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

from backend.adaptive.mindmonitor import MindMonitorFeed, run_mindmonitor


def send_pair(feed, sender="127.0.0.1", alpha=None, beta=None):
    feed.receive(sender, "/muse/elements/alpha_absolute", *(alpha or [0, 0, 0, 0]))
    feed.receive(sender, "/muse/elements/beta_absolute", *(beta or [0, 0, 0, 0]))


def test_one_good_sensor_excludes_bad_channels_and_uses_fable_thresholds(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.adaptive.mindmonitor.time.time", lambda: clock[0])
    feed = MindMonitorFeed()
    feed.receive("phone", "/muse/elements/horseshoe", 4, 1, 4, 4)
    for _ in range(10):
        clock[0] += .1
        send_pair(feed, "phone", [float("nan"), 0, 9, 9], [float("inf"), 0, -9, -9])
    status = feed.status()
    assert status["calibrated"] and status["live"]
    assert status["goodChannels"] == ["AF7"]
    assert status["label"] == "Focusing" and status["alphaBetaRatio"] == 1
    assert feed.latest()[2] > 1 and status["targetSamples"] == 10
    # Contact quality changing on excluded sensors must not erase warmup.
    feed.receive("phone", "/muse/elements/horseshoe", 2, 1, 3, 4)
    assert feed.status()["calibrated"]
    # A different phone must not mix its waves into this stream.
    send_pair(feed, "another-phone", [4] * 4, [0] * 4)
    assert feed.status()["alphaBetaRatio"] == 1
    clock[0] += 4
    assert not feed.status()["live"] and not feed.status()["calibrated"]
    assert feed.calibration is None and feed.readings == 0


def test_contact_loss_and_recovery_require_new_paired_waves(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.adaptive.mindmonitor.time.time", lambda: clock[0])
    feed = MindMonitorFeed()
    for _ in range(10):
        clock[0] += .1
        send_pair(feed)
    assert not feed.status()["calibrated"]
    feed.receive("127.0.0.1", "/muse/elements/horseshoe", 1, 1, 1, 1)
    assert not feed.status()["calibrated"]
    for _ in range(10):
        clock[0] += .1
        send_pair(feed, alpha=[.5], beta=[0])
    assert feed.status()["label"] == "Relaxing"
    assert feed.status()["calibrated"]
    count = len(feed.series)
    for _ in range(20):
        clock[0] += .01
        feed.receive("127.0.0.1", "/muse/elements/alpha_absolute", .5)
    assert len(feed.series) == count  # Old beta cannot count repeatedly.
    feed.receive("127.0.0.1", "/muse/elements/horseshoe", 4, 4, 4, 4)
    assert not feed.status()["live"] and not feed.status()["calibrated"]
    assert "No good contacts" in feed.status()["qualityError"]


def test_stopped_brainwaves_cannot_be_kept_alive_by_contact_packets(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.adaptive.mindmonitor.time.time", lambda: clock[0])
    feed = MindMonitorFeed()
    feed.receive("phone", "/muse/elements/horseshoe", 1, 1, 1, 1)
    for _ in range(10):
        clock[0] += .1
        send_pair(feed, "phone")
    assert feed.status()["calibrated"]
    for _ in range(4):
        clock[0] += 1
        feed.receive("phone", "/muse/elements/horseshoe", 1, 1, 1, 1)
    assert not feed.status()["calibrated"]
    assert feed.readings == 0


def test_real_udp_osc_bundle_receiver_and_shutdown(monkeypatch):
    monkeypatch.setenv("GOZ_MINDMONITOR_PORT", "0")
    feed, stop = MindMonitorFeed(), threading.Event()
    thread = threading.Thread(target=run_mindmonitor, args=(feed, stop))
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not feed.port and time.monotonic() < deadline:
            time.sleep(.01)
        assert feed.port
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for _ in range(12):
                bundle = OscBundleBuilder(IMMEDIATELY)
                for name, values in [("horseshoe", [4, 1, 4, 4]), ("alpha_absolute", [9., 0., 9., 9.]), ("beta_absolute", [-9., 0., -9., -9.])]:
                    message = OscMessageBuilder(address="/muse/elements/" + name)
                    for value in values: message.add_arg(value)
                    bundle.add_content(message.build())
                sender.sendto(bundle.build().dgram, ("127.0.0.1", feed.port))
                time.sleep(.015)
        deadline = time.monotonic() + 2
        while not feed.status()["calibrated"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert feed.status()["calibrated"]
        assert feed.status()["goodChannels"] == ["AF7"]
    finally:
        stop.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not feed.status()["calibrated"]


@pytest.mark.parametrize("values", [[], [1, 2], [float("nan")] * 4, [0] * 4])
def test_invalid_contact_packets_do_not_connect(values):
    feed = MindMonitorFeed()
    feed.receive("phone", "/muse/elements/horseshoe", *values)
    assert feed.device_id is None and not feed.status()["calibrated"]


def test_busy_udp_port_reports_actionable_error(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        monkeypatch.setenv("GOZ_MINDMONITOR_PORT", str(occupied.getsockname()[1]))
        feed = MindMonitorFeed()
        run_mindmonitor(feed, threading.Event())
        assert "Muse connection error: Mind Monitor OSC:" in feed.status()["qualityError"]
        assert not feed.status()["live"]
