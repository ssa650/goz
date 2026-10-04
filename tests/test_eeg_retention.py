"""Offline raw replay: a saved reference is distinct from fresh viewing evidence."""
from copy import deepcopy
import time
from types import SimpleNamespace

import pytest

from backend.adaptive import eeg_policy
from backend.adaptive.cumulative_eeg_policy import CumulativeEEGHistory, compatibility_key
from backend.adaptive.eeg_calibration import calibration_status
from backend.adaptive.sensors import EEG_RATE
from backend.sensor_setup import SensorSetup
from test_eeg_quality import clock, muse, replay
from test_sensor_setup import sensors


def snapshot(feed):
    return dict(feed.status(), calibration=deepcopy(feed.calibration or {}))


def clipped(count):
    def modify(x, ts):
        x[:, :count] = 999.51171875
        return x
    return modify


@pytest.fixture
def completed(clock, calibrated_snapshot):
    feed = muse(60)
    feed.__dict__.update(deepcopy(calibrated_snapshot[0]))
    clock[0] = calibrated_snapshot[1]
    q = snapshot(feed)
    assert feed.calibration['cleanSeconds'] >= 60
    assert calibration_status(q)['ready'], {key: q[key] for key in (
        'live', 'calibrated', 'confidence', 'qualityError', 'connectionState', 'sampleAgeSeconds')}
    return feed


@pytest.fixture(scope='module')
def calibrated_snapshot():
    """Replay actual 60s once; isolate every test's reference and current buffers."""
    feed = muse(60)
    replay_clock = [1000.]
    # Also flush the conservative recent-feature confidence window after completion.
    replay(feed, replay_clock, 65)
    return deepcopy({key: value for key, value in vars(feed).items() if key != 'lock'}), replay_clock[0]


@pytest.mark.parametrize('bad_count', [1, 3, 4])
def test_completed_reference_survives_bad_signal_without_cue_then_recovers(clock, completed, bad_count):
    feed = completed
    baseline = feed.calibration
    frozen = deepcopy(baseline)
    key = compatibility_key(snapshot(feed))
    count = len(feed.calibration_samples)
    seconds = feed.clean_time.seconds
    replay(feed, clock, 1, clipped(bad_count))
    q = snapshot(feed)
    assert q['live'] and q['calibrated'] and q['calibrationRetained']
    assert not q['signalReady'] and q['confidence'] == 0 and feed.latest()[3]
    assert q['selectedChannels'] == frozen['channels']
    assert len(q['availableChannels']) == 4 - bad_count
    assert compatibility_key(q) == key
    assert feed.calibration is baseline and baseline == frozen
    assert feed.clean_time.seconds == seconds and len(feed.calibration_samples) == count
    guided = calibration_status(q)
    assert guided['state'] == 'signal_weak' and guided['calibrationRetained']
    assert not guided['ready'] and not guided['signalReady']
    assert guided['progress'] == 1 and guided['remainingCleanSeconds'] == 0
    assert 'Playback' in guided['reason'] and 'paused' not in guided['reason']
    observation = eeg_policy.observe(feed.window(clock[0]-4, clock[0]), q, (clock[0]-4, clock[0]))
    assert not observation['eligible'] and eeg_policy.delivery_cue(eeg_policy.decide(observation)) == ''
    received = q['samplesReceived']
    replay(feed, clock, 6)
    q = snapshot(feed)
    assert q['samplesReceived'] > received and q['signalReady'] and q['confidence'] > 0
    assert calibration_status(q)['ready'] and not feed.latest()[3]
    assert feed.calibration is baseline and baseline == frozen and compatibility_key(q) == key
    assert feed.clean_time.seconds == seconds and len(feed.calibration_samples) == count


def test_packet_gap_retains_reference_requires_full_fresh_support(clock, completed):
    feed = completed
    baseline = feed.calibration
    clock[0] += .25
    replay(feed, clock, .25)
    q = snapshot(feed)
    assert feed.calibration is baseline and q['calibrationRetained']
    assert not q['signalReady'] and q['confidence'] == 0 and not feed.series
    assert q['effectiveHistorySeconds'] == .25
    assert calibration_status(q)['calibrationRetained'] and not calibration_status(q)['ready']
    replay(feed, clock, 3)
    assert feed.calibration is baseline and feed.status()['signalReady']


def test_incomplete_progress_keeps_only_compatible_clean_seconds(clock):
    feed = muse(60)
    replay(feed, clock, 8)
    seconds = feed.clean_time.seconds
    samples = deepcopy(feed.calibration_samples)
    channels = feed.selected_channels
    replay(feed, clock, 1, clipped(1))
    assert feed.clean_time.seconds == seconds and feed.calibration_samples == samples
    assert feed.selected_channels == channels and not feed.calibration
    assert calibration_status(snapshot(feed))['state'] == 'paused'
    # The first recovered full window anchors duration, without bridging artifacts.
    for _ in range(32):
        previous = feed.clean_time.seconds
        replay(feed, clock, .25)
        if not feed.quality_error:
            assert feed.clean_time.seconds == previous == seconds
            break
    else:
        pytest.fail('Clean calibrated channels did not recover')
    replay(feed, clock, 1)
    assert seconds < feed.clean_time.seconds <= seconds + 1
    seconds = feed.clean_time.seconds
    clock[0] += 1 / EEG_RATE
    replay(feed, clock, .25)
    assert feed.clean_time.seconds == seconds and feed.calibration_samples[:len(samples)] == samples
    assert not feed.calibration and not feed.series


def test_additional_clean_channels_do_not_change_an_existing_subset_reference(clock):
    feed = muse(3)
    replay(feed, clock, 7, clipped(1))
    baseline = feed.calibration
    assert baseline['channels'] == ['AF7', 'AF8', 'TP10']
    replay(feed, clock, 6)
    assert feed.calibration is baseline and feed.selected_channels == (1, 2, 3)
    assert feed.status()['availableChannels'] == ['TP9', 'AF7', 'AF8', 'TP10']
    assert feed.status()['signalReady'] and feed.status()['confidence'] == .75


@pytest.mark.parametrize('change', ['reset', 'recalibrate', 'disconnect', 'reconnect', 'device',
                                   'source', 'metadata', 'config', 'silence_polled', 'silence_unpolled'])
def test_genuine_invalidation_requires_fresh_reference(clock, completed, change):
    feed = completed
    if change == 'reset': feed.reset_calibration()
    elif change == 'recalibrate': feed.begin_calibration(60)
    elif change == 'disconnect': feed.disconnected()
    elif change == 'reconnect': feed.connected('Muse-test')
    elif change == 'device': feed.connected('Muse-other')
    elif change == 'source': feed.source = 'sim'
    elif change == 'metadata': feed.stream_metadata['channelLabels'] = ['AF7', 'TP9', 'AF8', 'TP10']
    elif change == 'config': feed.calibration_seconds = 90
    elif change == 'silence_polled': clock[0] = feed.last_sample_at + 3
    elif change == 'silence_unpolled':
        clock[0] += 3
        replay(feed, clock, .25)
    q = snapshot(feed)
    assert feed.calibration is None and q['confidence'] == 0
    assert not q['calibrationRetained'] and not calibration_status(q)['ready']
    assert q['cleanSeconds'] == 0 and compatibility_key(q) is None


def test_saved_reference_cannot_certify_bad_sustained_viewing_rows(clock, completed):
    feed = completed
    h = CumulativeEEGHistory('session')
    start = clock[0]
    h.start_clip('clip', start)
    key = compatibility_key(snapshot(feed))
    # Give collector one current feature before the artifact, then attest bad rows.
    replay(feed, clock, 3)
    row = feed.latest()
    h.collect('session', 'clip', [row], snapshot(feed), observed_at=clock[0])
    accepted = dict(h._active['rows'])
    replay(feed, clock, 1, clipped(4))
    row = feed.latest()
    h.collect('session', 'clip', [row], snapshot(feed), observed_at=clock[0])
    assert h._active['key'] == key and all(h._active['rows'][t] == r for t, r in accepted.items())
    assert h._active['rows'][row[0]][3] is True
    result = h.compare([], snapshot(feed), (start, clock[0]))
    assert not result['eligible'] and result['action'] == 'keep'
    assert feed.calibration is not None


def test_setup_allows_saved_weak_reference_but_preserves_gaze_and_initial_calibration_gates(tmp_path, monkeypatch):
    from test_cumulative_eeg_policy import quality
    sn = sensors()
    sn.eeg.source = 'muse'
    q = quality(calibrationRetained=True, signalReady=False, confidence=0,
                qualityError='clipped sensor channel', connectionState='poor_signal')
    sn.eeg.calibration = deepcopy(q.pop('calibration'))
    monkeypatch.setattr(sn.eeg, 'status', lambda: deepcopy(q))
    setup = SensorSetup(sn, tmp_path)
    setup.required = True; setup.phase = 'ready'; setup.gaze_calibrated = True
    setup.children['gaze'] = SimpleNamespace(returncode=None)
    sn.gaze.add(dict(t=time.time(), valid=True, face=True))
    result = setup.snapshot()
    assert result['generationReady'] and result['eegCalibration']['calibrationRetained']
    assert not result['eegCalibration']['signalReady'] and 'continue' in result['message']
    for field in ('calibrated', 'calibrationRetained'):
        q[field] = False
    sn.eeg.calibration = None
    assert not setup.snapshot()['generationReady']
    sn.eeg_gaze_only = True
    assert setup.snapshot()['generationReady']
    sn.eeg_gaze_only = False
    q.update(calibrated=True, calibrationRetained=True)
    sn.eeg.calibration = quality()['calibration']
    q['live'] = False
    assert not setup.snapshot()['generationReady']
    q['live'] = True; q['deviceId'] = 'another-device'
    assert not setup.snapshot()['generationReady']
    q['deviceId'] = 'device'
    setup.children['muse'] = SimpleNamespace(returncode=1)
    assert not setup.snapshot()['generationReady']
    del setup.children['muse']
    sn.gaze.samples.clear()
    assert not setup.snapshot()['generationReady']
    sn.gaze.add(dict(t=time.time(), valid=False, face=False))
    assert not setup.snapshot()['generationReady']
    sn.gaze.add(dict(t=time.time(), valid=True, face=True))
    setup.phase = 'connecting_muse'
    assert not setup.snapshot()['generationReady']
    setup.phase = 'ready'
    setup.gaze_calibrated = False
    assert not setup.snapshot()['generationReady']
    setup.gaze_calibrated = True; setup.children['gaze'].returncode = 1
    assert not setup.snapshot()['generationReady']
