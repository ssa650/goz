"""Deterministic software replay only; no live devices, sockets or paid calls."""
import json
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pytest

from backend.adaptive.eeg_quality import (CausalEegFilter, UniqueCleanTime,
    channel_diagnostics, muse_metadata)
from backend.adaptive.sensors import EegFeed, EEG_RATE


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("backend.adaptive.sensors.time.time", lambda: now[0])
    return now


def replay(feed, clock, seconds, modify=None, block=64):
    start = clock[0]
    count = int(seconds * EEG_RATE)
    for offset in range(0, count, block):
        ts = start + np.arange(offset, min(count, offset + block)) / EEG_RATE
        wave = 8 * np.sin(2 * np.pi * 10 * ts) + 3 * np.sin(2 * np.pi * 20 * ts)
        x = np.stack([wave] * 4, axis=1)
        if modify: x = modify(x, ts)
        clock[0] = float(ts[-1])
        feed.push(x.tolist(), ts.tolist())
    clock[0] = start + seconds


def muse(seconds=1):
    feed = EegFeed()
    feed.source = "muse"
    feed.connected("Muse-test")
    feed.begin_calibration(seconds)
    return feed


def test_clean_replay_has_exact_diagnostics_and_compatible_tuple(clock):
    feed = muse()
    replay(feed, clock, 7)
    status = feed.status()
    assert status["calibrated"] and status["confidence"] == 1
    assert status["selectedChannels"] == ["TP9", "AF7", "AF8", "TP10"]
    q = status["channelQuality"]["AF7"]
    assert q["rawMinUV"] < 0 < q["rawMaxUV"]
    assert q["rawOffsetUV"] == pytest.approx(0, abs=.05)
    assert q["railFraction"] == 0 and q["hardClipFraction"] == 0
    assert not q["flat"] and not q["rejectReasons"] and q["gapCount"] == 0
    assert status["filter"]["notchHz"] is None and status["featureWindowSeconds"] == 2
    assert status["effectiveHistorySeconds"] == 2 and status["featureAgeSeconds"] < .3
    assert len(feed.latest()) == 4 and feed.latest()[3] is False
    json.dumps(status, allow_nan=False)


def test_slow_drift_separated_from_band_limited_amplitude(clock):
    feed = muse()
    def drift(x, ts): return x + 300 * np.sin(2 * np.pi * .1 * (ts - 1000))[:, None]
    replay(feed, clock, 10, drift)
    status = feed.status()
    assert status["calibrated"] and not status["qualityError"]
    q = status["channelQuality"]["TP9"]
    assert q["peakToPeakUV"] > 150 and q["filteredPeakToPeakUV"] < 150
    assert abs(q["rawOffsetUV"]) > 100


@pytest.mark.parametrize("hz", [50, 60])
def test_mains_is_rejected_on_raw_spectrum_even_if_filter_attenuates(clock, hz):
    feed = muse()
    replay(feed, clock, 5, lambda x, ts: x + 70 * np.sin(2 * np.pi * hz * ts)[:, None])
    q = feed.status()["channelQuality"]["TP9"]
    assert q["lineNoiseFraction"] > .35 and "line_noise" in q["rejectReasons"]
    assert not feed.calibration and not feed.calibration_samples and feed.latest()[3]
    assert feed.status()["confidence"] == 0


@pytest.mark.parametrize("bad_count, confidence", [(1, .75), (2, .5), (3, 0), (4, 0)])
def test_clean_channel_selection_reduces_confidence_or_rejects(clock, bad_count, confidence):
    feed = muse()
    def bad(x, ts):
        x[:, :bad_count] = 999.51171875
        return x
    replay(feed, clock, 7, bad)
    status = feed.status()
    assert status["confidence"] == confidence
    assert len(status["availableChannels"]) == 4 - bad_count
    assert len(status["selectedChannels"]) == (4 - bad_count if bad_count <= 2 else 0)
    assert status["channelQuality"]["TP9"]["railFraction"] == 1
    assert "raw_clip" in status["channelQuality"]["TP9"]["rejectReasons"]
    assert bool(feed.calibration) == (bad_count <= 2)
    assert feed.latest()[3] == (bad_count > 2)
    if bad_count <= 2:
        assert set(feed.calibration["channels"]) == set(status["selectedChannels"])


def test_contact_loss_retains_baseline_and_locks_calibrated_channels(clock):
    feed = muse(3)
    replay(feed, clock, 7)
    baseline = feed.calibration
    seconds = feed.clean_time.seconds
    assert feed.calibration and any(not row[3] for row in feed.series)
    def bad(x, ts):
        x[:, 0] = 999.51171875
        return x
    replay(feed, clock, 1, bad)
    status = feed.status()
    assert feed.calibration is baseline and feed.clean_time.seconds == seconds
    assert status["calibrationRetained"] and not status["signalReady"]
    assert status["confidence"] == 0 and feed.latest()[3]
    assert status["selectedChannels"] == ["TP9", "AF7", "AF8", "TP10"]
    assert status["availableChannels"] == ["AF7", "AF8", "TP10"]
    replay(feed, clock, 4, bad)
    assert feed.calibration is baseline and feed.latest()[3]
    for _ in range(24):  # allow causal filter transients to settle honestly
        replay(feed, clock, .25)
        if feed.status()['signalReady']:
            break
    assert len(feed.selected_channels) == 4
    assert feed.calibration is baseline and feed.status()['signalReady']
    assert not feed.latest()[3]


@pytest.mark.parametrize("value", [0, 999.51171875, -1000])
def test_all_flat_or_rails_never_fabricate_valid_output(clock, value):
    feed = muse()
    replay(feed, clock, 5, lambda x, ts: np.full_like(x, value))
    assert not feed.calibration_samples and not feed.calibration
    assert feed.status()["confidence"] == 0 and all(row[3] for row in feed.series)
    assert "raw_flat" in feed.status()["channelQuality"]["TP9"]["rejectReasons"]


def test_large_in_band_amplitude_threshold_stays_150(clock):
    feed = muse()
    replay(feed, clock, 5, lambda x, ts: x * 20)
    assert not feed.calibration_samples
    assert "band_amplitude" in feed.status()["channelQuality"]["TP9"]["rejectReasons"]


def test_duplicate_and_out_of_order_samples_do_not_add_duration(clock):
    feed = muse(60)
    replay(feed, clock, 5)
    seconds, accepted = feed.clean_time.seconds, feed.samples_received
    t, x = feed.raw[-1]
    feed.push([x.tolist()] * 1000, [t] * 1000)
    feed.push([x.tolist()], [t - .01])
    assert feed.samples_received == accepted and feed.clean_time.seconds == seconds
    status = feed.status()
    assert status["inputDiagnostics"]["duplicateSamples"] == 1000
    assert status["inputDiagnostics"]["outOfOrderSamples"] == 1


def test_conflicting_duplicate_and_mismatched_chunk_fail_closed(clock):
    feed = muse()
    replay(feed, clock, 4)
    t, x = feed.raw[-1]
    feed.push([(x + 1).tolist()], [t])
    assert feed.status()["confidence"] == 0
    replay(feed, clock, 1)
    assert feed.latest()[3] and "invalid_input" in feed.status()["rejectReasons"]
    feed.push([[1, 2, 3, 4]], [])
    assert feed.status()["inputDiagnostics"]["mismatchedChunks"] == 1
    assert feed.status()["confidence"] == 0


def test_single_dropped_sample_retains_baseline_but_invalidates_current_response(clock):
    feed = muse()
    replay(feed, clock, 4)
    assert feed.calibration
    baseline = feed.calibration
    clock[0] += 1 / EEG_RATE
    replay(feed, clock, .25)
    status = feed.status()
    assert feed.calibration is baseline and not feed.series and status["confidence"] == 0
    assert status['calibrationRetained'] and not status['signalReady']
    assert "sample_gap" in status["rejectReasons"]
    assert status["effectiveHistorySeconds"] == .25
    replay(feed, clock, 4)
    assert feed.calibration is baseline and feed.status()['signalReady']


def test_stateful_filter_is_invariant_to_chunk_boundaries_and_has_no_lookahead():
    ts = np.arange(4 * EEG_RATE) / EEG_RATE
    raw = np.stack([100 + 8 * np.sin(2 * np.pi * 10 * ts)] * 4, axis=1)
    whole = CausalEegFilter().process(raw)
    incremental = CausalEegFilter()
    chunked = np.concatenate([incremental.process(raw[i:i+12]) for i in range(0, len(raw), 12)])
    assert np.allclose(whole, chunked, atol=1e-11)
    altered = raw.copy()
    altered[512:] += 100
    assert np.array_equal(CausalEegFilter().process(altered)[:512], whole[:512])
    incremental.reset()
    assert np.allclose(incremental.process(raw), whole)


def test_chunk_size_does_not_change_feature_or_unique_clean_seconds(clock):
    first = muse(60)
    replay(first, clock, 8, block=12)
    clock[0] = 1000
    second = muse(60)
    replay(second, clock, 8, block=64)
    assert np.allclose(first.series, second.series)
    assert first.clean_time.seconds == second.clean_time.seconds


def test_sixty_unique_seconds_not_sixty_overlapping_windows(clock):
    feed = muse(60)
    replay(feed, clock, 61)
    assert not feed.calibration and 58 < feed.status()["cleanSeconds"] < 60
    replay(feed, clock, 2)
    assert feed.calibration and 60 <= feed.calibration["cleanSeconds"] <= 60.25
    assert len(feed.calibration_samples) >= 241
    frozen = dict(feed.calibration)
    replay(feed, clock, 1)
    assert feed.calibration == frozen


def test_unique_interval_union_never_counts_overlap_or_unobserved_gap():
    coverage = UniqueCleanTime()
    assert coverage.add(0, 2) == 0  # startup is conservative
    for end in np.arange(2.25, 62.25, .25): coverage.add(end - 2, end)
    assert coverage.seconds == 60
    coverage.add(100, 102)
    assert coverage.seconds == 62
    assert coverage.add(100, 102) == 62


def test_window_timing_diagnostics_detect_missing_samples():
    ts = np.arange(512) / EEG_RATE
    raw = np.stack([8 * np.sin(2 * np.pi * 10 * ts)] * 4, axis=1)
    ts[256:] += 12 / EEG_RATE
    quality = channel_diagnostics(raw, CausalEegFilter().process(raw), ts, 150)
    assert quality["AF7"]["gapCount"] == 1 and quality["AF7"]["missingSamples"] == 12
    assert "sample_gap" in quality["AF7"]["rejectReasons"]


class XmlInfo:
    def __init__(self, labels=None, units="microvolts", identity="Muse-test"):
        root = Element("info")
        SubElement(root, "source_id").text = identity
        channels = SubElement(SubElement(root, "desc"), "channels")
        for label in labels or ["TP9", "AF7", "AF8", "TP10", "Right AUX"]:
            channel = SubElement(channels, "channel")
            SubElement(channel, "label").text = label
            SubElement(channel, "unit").text = units
        self.xml = tostring(root, encoding="unicode")
    def as_xml(self): return self.xml


@pytest.mark.parametrize("units", ["microvolts", "uV", "µV", "μV"])
def test_metadata_validates_units_labels_identity_and_excludes_aux(units):
    meta = muse_metadata(XmlInfo(units=units), "Muse-test")
    assert meta["valid"] and meta["verification"] == "verified"
    assert meta["excludedLabels"] == ["Right AUX"] and not meta["hardwarePacketCounterAvailable"]


@pytest.mark.parametrize("info", [XmlInfo(units="volts"), XmlInfo(identity="Muse-other"),
    XmlInfo(labels=["AF7", "TP9", "AF8", "TP10", "Right AUX"]),
    XmlInfo(labels=["TP9", "AF7", "AF8", "TP10", "unknown"])])
def test_metadata_mismatch_rejects_features_even_for_clean_wave(clock, info):
    feed = muse()
    feed.connected("Muse-test", muse_metadata(info, "Muse-test"))
    replay(feed, clock, 5)
    assert feed.status()["streamMetadata"]["verification"] == "rejected"
    assert feed.status()["rejectReasons"] == ["metadata"]
    assert feed.status()["confidence"] == 0 and not feed.calibration
    assert all(row[3] for row in feed.series)


def test_oversampled_timestamps_cannot_masquerade_as_two_seconds(clock):
    feed = muse()
    stamps = clock[0] + np.arange(2048) / (EEG_RATE * 4)
    wave = 8 * np.sin(2 * np.pi * 10 * np.arange(2048) / EEG_RATE)
    clock[0] = stamps[-1]
    feed.push(np.stack([wave] * 4, axis=1).tolist(), stamps.tolist())
    assert not feed.calibration and feed.latest()[3]
    assert "sample_gap" in feed.status()["rejectReasons"]
    assert feed.status()["channelQuality"]["TP9"]["irregularIntervals"] == 511
