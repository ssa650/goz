"""Offline reduced-redundancy eligibility; no physiological validity claim."""
from copy import deepcopy

import pytest

from backend.adaptive import director, eeg_policy, fusion, profile
from backend.adaptive.cumulative_eeg_policy import CumulativeEEGHistory
from backend.adaptive.eeg_calibration import calibration_status
from backend.clip_settings import build_h3_request
from test_cumulative_eeg_policy import collect, prompt, quality, rows
from test_eeg_quality import clock, muse, replay


def two_clean_quality(**overrides):
    q = quality(confidence=.5, channelPolicy='two_clean',
        availableChannels=['AF7', 'AF8'], channelConfidenceCeiling=.5,
        qualityVersion='raw-validity-causal-sos-v1',
        channelQuality={c: dict(usable=c in ('AF7', 'AF8'),
            rejectReasons=[] if c in ('AF7', 'AF8') else ['raw_clip'])
            for c in eeg_policy.MUSE_CHANNELS})
    q.update(overrides)
    return q


def sustained(z=1., q=None):
    q = two_clean_quality() if q is None else q
    h = CumulativeEEGHistory('session')
    for clip, start, value, duration in [('one', 100., 0., 15),
            ('two', 116., z, 15), ('three', 132., z, 3.5)]:
        h.start_clip(clip, start)
        collect(h, clip, start, value, duration=duration, q=q)
    return h


def compare(h, q=None):
    return h.compare([], two_clean_quality() if q is None else q, (132., 135.5))


@pytest.mark.parametrize('z,state,cue', [(1., 'above_prior_clips',
    'EEG DELIVERY TRIAL: Slightly quicken the existing gestures in the interior of this clip. Keep camera staging readable and retain all scripted actions, dialogue wording, cast and outcome. Preserve the required first and last frame compositions.'),
    (-1., 'below_prior_clips',
    'EEG DELIVERY TRIAL: Let the existing gestures unfold slightly more slowly in the interior of this clip. Keep camera staging steady and retain all scripted actions, dialogue wording, cast and outcome. Preserve the required first and last frame compositions.')])
def test_two_clean_sustained_opposites_have_exact_bounded_provider_prompts(z, state, cue):
    q = two_clean_quality()
    before = deepcopy(q)
    r = compare(sustained(z, q), q)
    p = prompt(r)
    assert r['eligible'] and r['state'] == state and r['confidence'] == .5
    expected = 'Pat and Bob wave.\n\nADAPTIVE SCENE DIRECTION (interior shot/delivery priority): ' + cue
    assert p['video_prompt'] == expected and p['video_prompt'].count('EEG DELIVERY TRIAL') == 1
    assert p['decision']['applied_actions'] == [r['action']]
    model, payload = build_h3_request(dict(mode='frames', prompt=p['video_prompt'], duration=15,
        resolution='480P', seed=42, promptExpansionMode='disabled'),
        dict(start='https://fal.media/first.png', end='https://fal.media/last.png'))
    neutral_model, neutral_payload = build_h3_request(dict(mode='frames', prompt='Pat and Bob wave.',
        duration=15, resolution='480P', seed=42, promptExpansionMode='disabled'),
        dict(start='https://fal.media/first.png', end='https://fal.media/last.png'))
    assert payload['prompt'] == expected and model == neutral_model
    assert {k: v for k, v in payload.items() if k != 'prompt'} == {
        k: v for k, v in neutral_payload.items() if k != 'prompt'}
    assert r['selected_channels'] == ['AF7', 'AF8']
    assert r['channel_coverage'] == r['channel_confidence_ceiling'] == r['confidence_threshold'] == .5
    assert r['reduced_redundancy'] and r['channel_policy'] == 'two_clean'
    assert 'not scientific probability' in r['confidence_basis']
    assert r['current_window'][1] - r['current_window'][0] == 12
    assert r['reference_window'][1] <= r['current_window'][0] - 2
    assert r['reference_coverage_s'] >= 12
    assert all(abs(d) >= r['change_threshold'] for d in r['persistence_block_deltas'])
    assert q == before


@pytest.mark.parametrize('failure', ['strict', 'unknown_mode', 'one_channel', 'no_diagnostics',
    'no_version', 'unusable', 'rejected', 'raw_error', 'missing_available', 'lost_locked_channel',
    'bad_ceiling', 'lower_confidence', 'no_calibration', 'under60', 'bad_scale', 'changed_identity',
    'changed_channels', 'uncalibrated', 'stale', 'not_live', 'bad_state', 'mindmonitor', 'sim'])
def test_half_confidence_never_bypasses_identity_or_quality(failure):
    q = two_clean_quality()
    if failure == 'strict': q['channelPolicy'] = 'strict'
    if failure == 'unknown_mode': q['channelPolicy'] = 'anything'
    if failure == 'one_channel': q['selectedChannels'] = ['AF7']
    if failure == 'no_diagnostics': q.pop('channelQuality')
    if failure == 'no_version': q.pop('qualityVersion')
    if failure == 'unusable': q['channelQuality']['AF8']['usable'] = False
    if failure == 'rejected': q['channelQuality']['AF8']['rejectReasons'] = ['band_amplitude']
    if failure == 'raw_error': q['qualityError'] = 'sample gap'
    if failure == 'missing_available': q.pop('availableChannels')
    if failure == 'lost_locked_channel': q['availableChannels'] = ['TP9', 'AF7']
    if failure == 'bad_ceiling': q['channelConfidenceCeiling'] = .75
    if failure == 'lower_confidence': q['confidence'] = .49
    if failure == 'no_calibration': q.pop('calibration')
    if failure == 'under60': q['calibration']['cleanSeconds'] = 59.9
    if failure == 'bad_scale': q['calibration']['channelScales'][0] = 0
    if failure == 'changed_identity': q['deviceId'] = 'other'
    if failure == 'changed_channels': q['selectedChannels'] = ['AF8', 'AF7']
    if failure == 'uncalibrated': q['calibrated'] = False
    if failure == 'stale': q['sampleAgeSeconds'] = 3
    if failure == 'not_live': q['live'] = False
    if failure == 'bad_state': q['connectionState'] = 'poor_signal'
    if failure in ('mindmonitor', 'sim'): q['source'] = failure
    # Reject at collection as well as comparison: a later good snapshot cannot
    # retrospectively certify features collected under bad signal/identity.
    rejected = compare(sustained(q=q))
    result = compare(sustained(), q)
    for r in (result, rejected):
        assert not r['eligible'] and r['action'] == 'keep'
        assert prompt(r)['video_prompt'] == 'Pat and Bob wave.'
    assert eeg_policy.quality_failure(q) is not None
    assert not calibration_status(q)['ready']
    assert not fusion.analyze([], rows(100., 1., 3.5), [], [],
        eeg_quality=q, eeg_window=(100., 103.5))['eeg_policy']['eligible']


def test_recovered_channels_preserve_locked_pair_but_never_replace_it():
    q = two_clean_quality(availableChannels=list(eeg_policy.MUSE_CHANNELS))
    assert eeg_policy.quality_failure(q) is None
    assert compare(sustained(q=q), q)['eligible']
    q['availableChannels'].remove('AF8')
    assert eeg_policy.quality_failure(q) is not None
    assert not compare(sustained(), q)['eligible']


def test_no_reference_burst_gaps_and_latest_artifact_still_abstain():
    q = two_clean_quality()
    h = CumulativeEEGHistory('session'); h.start_clip('first', 100.)
    collect(h, 'first', 100., 1., q=q)
    r = h.compare([], q, (100., 115.))
    assert not r['eligible'] and r['reference_count'] == 0 and r['action'] == 'keep'
    h = sustained()
    h.collect('session', 'three', [(135.5, .6, 1., True)], q, observed_at=135.5)
    assert not compare(h)['eligible']
    h = CumulativeEEGHistory('session'); h.start_clip('first', 100.)
    h.collect('session', 'first', [(103.5, .6, 1., False)] * 1000, q, observed_at=103.5)
    r = h.compare([], q, (100., 103.5))
    assert not r['eligible'] and r['valid_span_s'] == 0
    h = sustained()
    for t in list(h._reference[1]['rows']):
        if 120 < t < 128: h._reference[1]['rows'][t] = (t, .6, 1., True)
    assert not compare(h)['eligible']


def test_current_gaze_pacing_keeps_priority_with_reduced_redundancy():
    r = compare(sustained())
    a = dict(eeg_policy=r, characters={}, valid_gaze_s=0, comparison_s=0)
    p, _ = profile.update(profile.new_profile(['Pat', 'Bob']), a)
    hint = profile.decide(p, a)
    hint['pacing'] = 'slower'
    plan = director.template(dict(premise='Pat and Bob wave.'), hint, 15, ['Pat', 'Bob'])
    assert not plan['decision']['eeg_applied']
    assert plan['decision']['eeg_policy']['suppressed_by'] == 'gaze_readability_pacing'
    assert 'EEG DELIVERY TRIAL' not in plan['video_prompt']


def test_reduced_redundancy_retention_requires_new_persistent_evidence():
    q = two_clean_quality()
    h = sustained()
    assert compare(h)['action'] == 'faster_pacing'
    collect(h, 'three', 132., 1.5, q=q)
    h.start_clip('four', 148.)
    collect(h, 'four', 148., 1.5, duration=3.5, q=q)
    r = h.compare([], q, (148., 151.5))
    assert r['eligible'] and r['change_threshold'] == pytest.approx(.6 * r['entry_threshold'])
    h.start_clip('five', 164.)
    r = h.compare([], q, (164., 167.5))
    assert not r['eligible'] and h._last_state is None


@pytest.mark.parametrize('invalid', [(1., True), (5., False), (float('nan'), False)])
def test_reduced_threshold_does_not_relax_feature_artifact_or_clipping_guards(invalid):
    q = two_clean_quality()
    z, artifact = invalid
    evidence = [(t, v, z, artifact) for t, v, _, _ in rows(100., duration=3.5)]
    observation = fusion.analyze([], evidence, [], [], eeg_quality=q,
        eeg_window=(100., 103.5))['eeg_policy']
    assert not observation['eligible'] and not eeg_policy.delivery_cue(observation)
    h = sustained()
    h.collect('session', 'three', [(135.5, .6, z, artifact)], q, observed_at=135.5)
    assert not compare(h)['eligible']


def test_environment_opt_in_strict_rollback_and_snapshot_mode(monkeypatch):
    q = two_clean_quality(); q.pop('channelPolicy')
    monkeypatch.delenv('GOZ_EEG_CHANNEL_POLICY', raising=False)
    assert eeg_policy.quality_failure(q) is not None
    monkeypatch.setenv('GOZ_EEG_CHANNEL_POLICY', 'two_clean')
    assert eeg_policy.quality_failure(q) is None
    q['channelPolicy'] = 'strict'
    assert eeg_policy.quality_failure(q) is not None
    q.pop('channelPolicy')
    monkeypatch.setenv('GOZ_EEG_CHANNEL_POLICY', 'strict')
    assert eeg_policy.quality_failure(q) is not None


def test_mindmonitor_keeps_its_existing_strict_quality_semantics():
    q = two_clean_quality(source='mindmonitor', artifactCoverage='raw clipping/movement and contacts',
        contactAgeSeconds=.1, alphaAgeSeconds=.1, betaAgeSeconds=.1)
    assert eeg_policy.channel_eligibility(q)['confidence_threshold'] == .6
    assert eeg_policy.quality_failure(q) is not None
    q['confidence'] = .75
    assert eeg_policy.quality_failure(q) is None
    q['artifactCoverage'] = 'contacts only'
    assert eeg_policy.quality_failure(q) is not None


def test_real_raw_two_channel_calibration_readiness_and_retention(clock, monkeypatch):
    monkeypatch.setenv('GOZ_EEG_CHANNEL_POLICY', 'two_clean')
    feed = muse(60)
    def excluded(x, ts):
        x[:, [0, 3]] = 999.51171875
        return x
    replay(feed, clock, 30, excluded)
    q = dict(feed.status(), calibration=deepcopy(feed.calibration))
    assert not q['calibrated'] and not q['signalReady'] and not calibration_status(q)['ready']
    replay(feed, clock, 35, excluded)
    baseline = deepcopy(feed.calibration)
    q = dict(feed.status(), calibration=deepcopy(feed.calibration))
    assert q['confidence'] == .5 and q['signalReady'] and calibration_status(q)['ready']
    assert q['selectedChannels'] == ['AF7', 'AF8'] and q['excludedChannels'] == ['TP9', 'TP10']
    assert q['channel_policy'] == 'two_clean' and q['reduced_redundancy']
    assert q['confidence_threshold'] == .5 and 'reduced redundancy' in q['qualityWarning']
    monkeypatch.setenv('GOZ_EEG_CHANNEL_POLICY', 'strict')
    strict = dict(feed.status(), calibration=deepcopy(feed.calibration))
    assert strict['confidence'] == .5 and not strict['signalReady']
    assert not calibration_status(strict)['ready'] and feed.calibration == baseline
    monkeypatch.setenv('GOZ_EEG_CHANNEL_POLICY', 'two_clean')
    def lost(x, ts):
        x[:, [0, 2, 3]] = 999.51171875
        return x
    replay(feed, clock, 1, lost)
    q = dict(feed.status(), calibration=deepcopy(feed.calibration))
    assert q['calibrationRetained'] and not q['signalReady'] and q['confidence'] == 0
    assert q['selectedChannels'] == ['AF7', 'AF8'] and q['availableChannels'] == ['AF7']
    assert not calibration_status(q)['ready'] and feed.calibration == baseline
    replay(feed, clock, 6, excluded)
    assert feed.status()['signalReady'] and feed.calibration == baseline
