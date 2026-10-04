"""EEG acquisition regressions; synthetic signals here are test inputs only."""
import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from backend.adaptive.sensors import EegFeed, EEG_RATE, _consume_muse
from backend.adaptive.mindmonitor import MindMonitorFeed
from backend.sensor_setup import SensorSetup


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("backend.adaptive.sensors.time.time", lambda: now[0])
    return now


def signal(feed, clock, seconds, amplitude=8):
    start = clock[0]
    for block in range(int(seconds * 4)):
        stamps = start + block / 4 + np.arange(64) / EEG_RATE
        wave = amplitude * np.sin(2 * np.pi * 10 * stamps) + 3 * np.sin(2 * np.pi * 20 * stamps)
        clock[0] = float(stamps[-1])
        feed.push(np.stack([wave] * 4, axis=1).tolist(), stamps.tolist())
    clock[0] = start + seconds


def muse():
    feed = EegFeed()
    feed.source = "muse"
    feed.connected("Muse-device")
    return feed


def test_advertised_outlet_is_not_live_or_calibrated(clock):
    state = muse().status()
    assert state["outletAvailable"] and state["samplesReceived"] == 0
    assert state["samplingState"] == "waiting_for_samples"
    assert not state["live"] and not state["calibrated"] and state["confidence"] == 0
    assert "actual EEG samples" in state["qualityError"]


def test_numpy_timestamps_produce_json_serializable_readiness(clock):
    feed = muse()
    feed.source = "sim"
    feed.push([[1, 2, 3, 4]], [np.float64(clock[0])])
    state = feed.status()
    assert type(state["live"]) is bool
    assert type(state["calibrated"]) is bool
    json.dumps(state, allow_nan=False)


def test_raw_samples_warmup_baseline_and_channel_diagnostics(clock):
    feed = muse()
    signal(feed, clock, 1)
    state = feed.status()
    assert state["live"] and state["samplingState"] == "warming_up"
    assert state["samplesReceived"] == 256 and not state["calibrated"]
    signal(feed, clock, 2)
    state = feed.status()
    assert state["samplingState"] == "calibrating" and state["cleanSeconds"] > 0
    assert state["targetSeconds"] == 60 and state["confidence"] == 0
    assert all(row[3] for row in feed.window(0, clock[0]))
    assert set(state["channelQuality"]) == {"TP9", "AF7", "AF8", "TP10"}
    assert state["channelQuality"]["AF7"]["peakToPeakUV"] < 150


def test_stale_data_expires_baseline_and_reconnect_collects_fresh(clock):
    feed = muse()
    feed.begin_calibration(seconds=1)
    signal(feed, clock, 4)
    assert feed.status()["calibrated"]
    clock[0] += 4
    state = feed.status()
    assert state["samplingState"] == "stale" and state["confidence"] == 0
    assert not state["calibrated"] and not feed.calibration and not feed.series
    signal(feed, clock, 2)
    assert not feed.status()["calibrated"]
    signal(feed, clock, 2)
    assert feed.status()["calibrated"]


def test_latest_quality_rejection_disables_confidence_without_weakening_gate(clock):
    feed = muse()
    feed.begin_calibration(seconds=1)
    signal(feed, clock, 4)
    assert feed.status()["confidence"] > 0
    signal(feed, clock, 1, amplitude=120)
    state = feed.status()
    assert state["live"] and state["connectionState"] == "poor_signal"
    assert state["confidence"] == 0 and feed.latest()[3]
    assert "movement or poor contact" in state["qualityError"]
    assert state["channelQuality"]["TP9"]["peakToPeakUV"] > 150


def test_baseline_reset_removes_old_response_rows_and_invalid_samples(clock):
    feed = muse()
    feed.begin_calibration(seconds=1)
    signal(feed, clock, 4)
    assert feed.series
    feed.reset_calibration()
    feed.push([[float("nan")] * 4], [clock[0]])
    assert not feed.series and not feed.calibration
    assert feed.status()["confidence"] == 0


class Info:
    def __init__(self, identity="Muse-device", name="Muse"):
        self.identity, self.label = identity, name
    def name(self): return self.label
    def channel_count(self): return 5
    def nominal_srate(self): return 256
    def source_id(self): return self.identity


def lsl(inlet, resolver=None):
    return SimpleNamespace(StreamInlet=inlet, resolve_byprop=resolver or (lambda *a, **k: [Info()]),
                           local_clock=lambda: 123.0)


@pytest.mark.parametrize("unix", [False, True])
def test_clock_conversion_is_bounded_and_unix_ignores_lsl_correction(clock, unix):
    feed, stop, closed = muse(), threading.Event(), []
    class Inlet:
        def __init__(self, info, **kwargs):
            assert kwargs["recover"] is False and kwargs["max_buflen"] == 5
        def pull_chunk(self, timeout):
            return [[1, 2, 3, 4, 999]], [1000.0 if unix else 122.8]
        def time_correction(self, timeout):
            assert not unix and timeout == .25
            return .2
        def close_stream(self): closed.append(True)
    captured = []
    def push(samples, stamps):
        captured.append((samples, stamps))
        stop.set()
    feed.push = push
    _consume_muse(feed, stop, lsl(Inlet))
    assert captured == [([[1, 2, 3, 4]], [1000.0])]
    assert closed == [True] and feed.device_id is None


def test_missing_stream_and_cancel_does_not_open_an_inlet(clock):
    feed, stop = muse(), threading.Event()
    feed.disconnected()
    def resolve(*a, **kwargs):
        assert kwargs["timeout"] == 1
        stop.set()
        return []
    _consume_muse(feed, stop, lsl(lambda *a, **k: pytest.fail("Cancelled resolution opened an inlet"), resolve))
    assert not feed.status()["live"]


def test_lost_stream_reconnects_to_same_device_and_closes_each_inlet(clock):
    feed, stop, opened, closed = muse(), threading.Event(), [], []
    class LostError(Exception): pass
    class Inlet:
        def __init__(self, info, **kwargs):
            opened.append(info.source_id())
            self.number = len(opened)
        def pull_chunk(self, timeout):
            if self.number == 1:
                raise LostError("test loss")
            return [[1, 2, 3, 4]], [1000.0]
        def close_stream(self): closed.append(self.number)
    pylsl = lsl(Inlet, lambda *a, **k: [Info(), Info("Muse-other")])
    monkey_streams = [0]
    def resolve(*a, **k):
        monkey_streams[0] += 1
        return [Info()] if monkey_streams[0] == 1 else [Info(), Info("Muse-other")]
    original_push = feed.push
    def push(samples, stamps):
        original_push(samples, stamps)
        stop.set()
    feed.push = push
    pylsl.resolve_byprop, pylsl.LostError = resolve, LostError
    _consume_muse(feed, stop, pylsl)
    assert opened == ["Muse-device", "Muse-device"] and closed == [1, 2]
    assert feed.samples_received == 1 and not feed.calibration


def test_unexpected_inlet_error_still_cleans_up(clock):
    feed, stop, closed = muse(), threading.Event(), []
    class Inlet:
        def __init__(self, *a, **k): pass
        def pull_chunk(self, timeout): raise RuntimeError("unexpected")
        def close_stream(self): closed.append(True)
    with pytest.raises(RuntimeError, match="unexpected"):
        _consume_muse(feed, stop, lsl(Inlet))
    assert closed == [True] and feed.device_id is None


def test_no_samples_outlet_retries_and_cannot_become_ready(clock):
    feed, stop, closed = muse(), threading.Event(), []
    class Inlet:
        def __init__(self, *a, **k): pass
        def pull_chunk(self, timeout):
            clock[0] += 2
            return [], []
        def close_stream(self):
            closed.append(True)
            stop.set()
    _consume_muse(feed, stop, lsl(Inlet))
    assert closed == [True] and feed.samples_received == 0 and not feed.status()["calibrated"]


def test_stale_timestamp_chunk_is_not_counted(clock):
    feed, stop = muse(), threading.Event()
    class Inlet:
        def __init__(self, *a, **k): pass
        def pull_chunk(self, timeout):
            clock[0] += 6
            return [[1, 2, 3, 4]], [990.0]
        def close_stream(self): stop.set()
    _consume_muse(feed, stop, lsl(Inlet))
    assert feed.samples_received == 0


@pytest.mark.parametrize("multiple", [False, True])
def test_multiple_streams_require_selection_and_respect_address(clock, monkeypatch, multiple):
    feed, stop, opened = muse(), threading.Event(), []
    class Inlet:
        def __init__(self, info, **k): opened.append(info.source_id())
        def pull_chunk(self, timeout):
            stop.set()
            return [], []
        def close_stream(self): pass
    pylsl = lsl(Inlet, lambda *a, **k: [Info("Museone"), Info("Musetwo")])
    if multiple:
        monkeypatch.delenv("GOZ_MUSE_ADDRESS", raising=False)
        with pytest.raises(ValueError, match="Multiple Muse streams"):
            _consume_muse(feed, stop, pylsl)
    else:
        monkeypatch.setenv("GOZ_MUSE_ADDRESS", "two")
        _consume_muse(feed, stop, pylsl)
        assert opened == ["Musetwo"]


def pair(feed, clock, count):
    for _ in range(count):
        clock[0] += .1
        feed.receive("phone", "/muse/elements/alpha_absolute", 0.)
        feed.receive("phone", "/muse/elements/beta_absolute", 0.)


def test_osc_raw_alone_and_prebaseline_are_not_response_evidence(clock):
    feed = MindMonitorFeed()
    feed.receive("phone", "/muse/eeg", 1., 2., 3., 4.)
    state = feed.status()
    assert state["rawSamplesReceived"] == 1 and not state["calibrated"]
    assert state["confidence"] == 0 and not feed.series
    feed.receive("phone", "/muse/elements/horseshoe", 1, 1, 1, 1)
    pair(feed, clock, 9)
    assert feed.status()["confidence"] == 0 and all(row[3] for row in feed.series)
    pair(feed, clock, 1)
    assert feed.status()["calibrated"] and feed.status()["confidence"] == 1
    assert not feed.latest()[3]


def test_osc_invalid_packets_do_not_keep_dead_peer_alive(clock):
    feed = MindMonitorFeed()
    feed.receive("phone", "/muse/elements/horseshoe", 1, 1, 1, 1)
    pair(feed, clock, 10)
    clock[0] += 2
    feed.receive("phone", "/unknown", 1)
    feed.receive("phone", "/muse/elements/horseshoe", 0, 0, 0, 0)
    clock[0] += 2
    feed.receive("new-phone", "/muse/elements/horseshoe", 1, 1, 1, 1)
    assert feed.peer == "new-phone" and not feed.status()["calibrated"]
    assert not feed.raw_recent and feed.artifact_until == 0


@pytest.mark.asyncio
async def test_setup_timeout_includes_quality_and_clean_progress(tmp_path, clock):
    feed = muse()
    signal(feed, clock, 3, amplitude=120)
    setup = SensorSetup(SimpleNamespace(eeg=feed, gaze_mode="off"), tmp_path)
    setup.phase, setup.message = "calibrating_muse", "Collect baseline."
    with pytest.raises(ValueError, match="movement or poor contact.*Baseline: 0.0/60.0"):
        await setup.wait_for(lambda: False, -1)


@pytest.mark.asyncio
async def test_optional_retry_cancels_previous_discovery_and_preserves_bridge(tmp_path, clock):
    feed = muse()
    setup = SensorSetup(SimpleNamespace(eeg=feed, gaze_mode="off"), tmp_path)
    cancelled = []
    async def discover():
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)
    setup.muse_task = asyncio.create_task(discover())
    await asyncio.sleep(0)
    stopped = []
    async def stop_children(keep_muse=False): stopped.append(keep_muse)
    async def start(): pass
    setup.stop_children, setup.start = stop_children, start
    await setup.retry()
    assert cancelled == [True] and stopped == [True]


@pytest.mark.asyncio
async def test_explicit_connect_reuses_existing_bridge_during_reader_recovery(tmp_path, clock):
    feed = muse()
    feed.disconnected()
    sensors = SimpleNamespace(eeg=feed, gaze_mode="off", eeg_mode="muse", start_muse_reader=lambda: None)
    setup = SensorSetup(sensors, tmp_path)
    bridge = SimpleNamespace(returncode=None)
    setup.children["muse"] = bridge
    calls = []
    async def wait_for(predicate, timeout, child=None):
        calls.append(child)
        if child is None:
            raise ValueError("No outlet yet")
        feed.connected("Muse-device")
        if timeout == 240:
            return  # Calibration completion is independent from bridge ownership.
        assert predicate()
    async def launch(*a): pytest.fail("A second bridge must not be launched")
    setup.wait_for, setup.launch = wait_for, launch
    await setup.connect_muse()
    await setup.muse_task
    assert calls == [None, bridge, bridge]
    assert setup.manual_muse_state == "connected" and not feed.connection_error
