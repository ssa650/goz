"""Clock bootstrap regressions: deterministic LSL inputs, no devices or outlets."""
from types import SimpleNamespace

import numpy as np
import pylsl.util as lsl_errors
import pytest

from backend.adaptive.sensors import EegFeed, EEG_RATE, _consume_muse
from test_eeg_acquisition import Info


class Clock:
    wall = 1_700_000_000.0

    def __init__(self):
        self.elapsed = 0.0
        self.cancelled = False
        self.waits = []

    def is_set(self):
        return self.cancelled

    def set(self):
        self.cancelled = True

    def wait(self, seconds):
        assert 0 <= seconds <= .25
        self.waits.append(seconds)
        self.elapsed += seconds
        return self.cancelled

    def time(self):
        return self.wall + self.elapsed

    def monotonic(self):
        return self.elapsed

    def local_clock(self):
        return 100.0 + self.elapsed


@pytest.fixture
def rig(monkeypatch):
    import backend.adaptive.sensors as module
    clock = Clock()
    monkeypatch.setattr(module, 'time', clock)
    monkeypatch.delenv('GOZ_MUSE_ADDRESS', raising=False)
    feed = EegFeed()
    feed.source = 'muse'
    return clock, feed


def api(clock, inlet):
    return SimpleNamespace(util=lsl_errors, StreamInlet=inlet,
                           resolve_byprop=lambda *a, **k: [Info()], local_clock=clock.local_clock)


def reader(clock, feed, ready_after, cancel_after=None, bad_correction=None):
    opened, closed, calls, chunks, snapshots = [], [], [], [], []
    class Inlet:
        def __init__(self, info, **kwargs):
            opened.append(self)
            self.started = None
        def pull_chunk(self, timeout):
            assert timeout == .5
            clock.elapsed += 12 / EEG_RATE
            stamps = clock.local_clock() - .2 - np.arange(11, -1, -1) / EEG_RATE
            wave = 8 * np.sin(2 * np.pi * 10 * stamps) + 3 * np.sin(2 * np.pi * 20 * stamps)
            chunk = np.stack([wave] * 4 + [np.zeros(12)], axis=1).tolist()
            chunks.append((chunk, stamps.tolist()))
            return chunks[-1]
        def time_correction(self, timeout):
            assert 0 < timeout <= .25
            if self.started is None:
                self.started = clock.elapsed
            calls.append((self, clock.elapsed, timeout))
            remaining = self.started + ready_after - clock.elapsed
            clock.elapsed += min(timeout, max(0, remaining))
            if cancel_after is not None and clock.elapsed >= cancel_after:
                clock.set()
                return .2  # A result returning after Disconnect must not be used.
            if clock.elapsed < self.started + ready_after:
                if clock.elapsed >= 6:
                    snapshots.append(feed.status())
                    clock.set()
                raise lsl_errors.TimeoutError('clock estimate not available')
            return .2 if bad_correction is None else bad_correction
        def close_stream(self):
            closed.append(self)
    original = feed.push
    accepted = []
    def push(samples, stamps):
        accepted.append((samples, stamps))
        original(samples, stamps)
        snapshots.append(feed.status())
        if feed.samples_received >= 48:
            clock.set()
    feed.push = push
    return Inlet, opened, closed, calls, chunks, snapshots, accepted


def test_liblsl_640ms_bootstrap_keeps_inlet_and_original_realistic_chunk(rig):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, .128 + 8 * .064)
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened
    assert all(call[0] is opened[0] for call in calls)
    assert calls[2][1] - calls[0][1] == pytest.approx(.5)
    assert accepted[0][0] == [row[:4] for row in chunks[0][0]]
    assert accepted[0][1] == pytest.approx([clock.wall + stamp - 100 + .2 for stamp in chunks[0][1]])
    assert feed.samples_received == 48
    assert snapshots[0]['samplesReceived'] == 12 and snapshots[0]['live']
    assert snapshots[0]['sampleAgeSeconds'] == pytest.approx(.640)
    assert all(not state['calibrated'] and state['confidence'] == 0 for state in snapshots)
    assert all(b > a for a, b in zip(accepted[-1][1], accepted[-1][1][1:]))
    assert feed.device_id is None and feed.last_sample_at is None


def test_delayed_estimate_retries_same_pending_chunk_after_initial_two_second_budget(rig):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, 2.3)
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened
    assert feed.samples_received == 48
    assert accepted[0][0] == [row[:4] for row in chunks[0][0]]
    assert snapshots[0]['sampleAgeSeconds'] == pytest.approx(2.3)
    assert calls[8][1] - calls[0][1] == pytest.approx(2.0)


def test_unavailable_clock_is_honest_without_reopening_or_unix_fallback(rig):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, 100)
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened and len(chunks) == 1
    assert feed.samples_received == 0 and not accepted
    assert clock.elapsed < 6.3
    assert snapshots[0]['acquisitionPhase'] == 'waiting_for_clock'
    assert 'timestamps cannot be verified' in snapshots[0]['qualityError']
    assert not snapshots[0]['live'] and not snapshots[0]['calibrated'] and snapshots[0]['confidence'] == 0


@pytest.mark.parametrize('cancel_after', [.25, .75, 2.25])
def test_late_clock_result_after_cancellation_never_pushes_and_closes_once(rig, cancel_after):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, 100, cancel_after)
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened
    assert feed.samples_received == 0 and not accepted and feed.device_id is None
    assert clock.elapsed <= cancel_after + .25


@pytest.mark.parametrize('correction', [float('nan'), float('inf'), -10., 1.5])
def test_invalid_stale_or_future_corrected_timestamps_are_not_accepted(rig, correction):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, .64, bad_correction=correction)
    original_pull = inlet.pull_chunk
    def pull(self, timeout):
        result = original_pull(self, timeout)
        if len(chunks) > 2:
            clock.elapsed += 6
        return result
    inlet.pull_chunk = pull
    original_close = inlet.close_stream
    def close(self):
        snapshots.append(feed.status())
        original_close(self)
        clock.set()
    inlet.close_stream = close
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened
    assert feed.samples_received == 0 and not accepted
    assert 'Stale or invalid' in snapshots[-1]['qualityError']
    assert not snapshots[-1]['live'] and snapshots[-1]['confidence'] == 0


def test_estimate_after_freshness_limit_drops_only_stale_pending_samples(rig):
    clock, feed = rig
    inlet, opened, closed, calls, chunks, snapshots, accepted = reader(clock, feed, 3.3)
    _consume_muse(feed, clock, api(clock, inlet))
    assert len(opened) == 1 and closed == opened
    assert len(chunks) == 5 and len(accepted) == 4 and feed.samples_received == 48
    assert accepted[0][0] == [row[:4] for row in chunks[1][0]]
    assert snapshots[0]['sampleAgeSeconds'] == 0


@pytest.mark.parametrize('phase', ['open', 'pull'])
def test_cancellation_after_blocking_open_or_pull_cleans_up_without_samples(rig, phase):
    clock, feed = rig
    opened, closed = [], []
    class Inlet:
        def __init__(self, *a, **k):
            opened.append(self)
            if phase == 'open':
                clock.set()
        def pull_chunk(self, timeout):
            clock.set()
            return [[1, 2, 3, 4, 5]], [clock.time()]
        def close_stream(self):
            closed.append(self)
    _consume_muse(feed, clock, api(clock, Inlet))
    assert len(opened) == 1 and closed == opened
    assert feed.samples_received == 0 and feed.device_id is None


def test_early_clock_timeouts_are_rate_limited_and_cancellable(rig):
    clock, feed = rig
    calls, closed = [], []
    class Inlet:
        def __init__(self, *a, **k): pass
        def pull_chunk(self, timeout):
            return [[1, 2, 3, 4, 5]], [clock.local_clock()]
        def time_correction(self, timeout):
            calls.append(timeout)
            if len(calls) == 10:
                clock.set()
            raise lsl_errors.TimeoutError('immediate provider timeout')
        def close_stream(self): closed.append(True)
    _consume_muse(feed, clock, api(clock, Inlet))
    assert len(calls) == 10 and sum(clock.waits) == pytest.approx(2.5)
    assert all(timeout == .25 for timeout in calls)
    assert closed == [True] and feed.samples_received == 0


def test_mixed_buffer_preserves_fresh_samples_without_accepting_invalid_timestamps(rig):
    clock, feed = rig
    closed, captured = [], []
    class Inlet:
        def __init__(self, *a, **k): pass
        def pull_chunk(self, timeout):
            # Delayed buffered data can straddle the freshness boundary.
            ages = [3.1, 3., 2.9, .1, 0., -.6, float('nan')]
            return [[index] * 5 for index in range(len(ages))], [clock.local_clock() - age for age in ages]
        def time_correction(self, timeout): return 0.
        def close_stream(self): closed.append(True)
    original = feed.push
    def push(samples, stamps):
        original(samples, stamps)
        captured.append((samples, stamps, feed.status()))
        clock.set()
    feed.push = push
    _consume_muse(feed, clock, api(clock, Inlet))
    assert len(captured) == 1 and closed == [True]
    assert captured[0][0] == [[index] * 4 for index in (2, 3, 4)]
    assert captured[0][1] == pytest.approx([clock.wall - age for age in (2.9, .1, 0.)])
    assert feed.samples_received == 3 and captured[0][2]['live']
    assert captured[0][2]['confidence'] == 0
