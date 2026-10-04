from copy import deepcopy
from backend.adaptive.eeg_calibration import calibration_status
from test_cumulative_eeg_policy import quality


def test_real_baseline_ready_and_sixty_wall_seconds_do_not_fake_readiness():
    assert calibration_status(quality())['ready']
    q=quality();q['calibration']['cleanSeconds']=59.9
    assert not calibration_status(q)['ready']
    q=quality(calibrated=False)
    assert not calibration_status(q)['ready']


def test_invalid_signal_pauses_and_does_not_mutate_progress_or_baseline():
    q=quality(cleanSeconds=22.,qualityError='raw artifact',calibrated=False)
    before=deepcopy(q);r=calibration_status(q)
    assert r['state']=='paused' and r['cleanSeconds']==22. and q==before
    assert 'eyes open' in r['instructions'] and 'breathe naturally' in r['instructions']
    assert r['targetSeconds']==60. and r['startAction']=='start_or_recalibrate'


def test_channel_change_stale_and_unsupported_mindmonitor_are_not_ready():
    for q in (quality(selectedChannels=['TP9','AF8']),quality(sampleAgeSeconds=4.),quality(source='mindmonitor')):
        assert not calibration_status(q)['ready']
    r=calibration_status(quality(source='mindmonitor'))
    assert not r['supported'] and r['state']=='unsupported' and 'gaze-only' in r['reason']


def test_missing_signal_and_clean_progress_are_explicit():
    assert calibration_status({})['state']=='unsupported'
    r=calibration_status(quality(cleanSeconds=12.,calibrated=False))
    assert r['state']=='collecting' and r['progress']==.2 and not r['ready']
