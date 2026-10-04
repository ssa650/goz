"""Offline sustained controller tests; not physiological validation."""
from copy import deepcopy

import pytest
from backend.adaptive import director, profile
from backend.adaptive.cumulative_eeg_policy import CumulativeEEGHistory, VERSION


def quality(**overrides):
    q=dict(source='muse',deviceId='device',selectedChannels=['AF7','AF8'],live=True,
        calibrated=True,confidence=.9,qualityError='',connectionState='streaming',sampleAgeSeconds=.02,
        cleanSeconds=60., calibration=dict(deviceId='device',channels=['AF7','AF8'],cleanSeconds=60.,
            channelMedians=[.5,.5],channelScales=[.1,.1],calibratedAt=90.))
    q.update(overrides)
    return q


def rows(start,z=0.,duration=15):
    return [(start+i*.25,.6,z,False) for i in range(int(duration/.25)+1)]


def collect(h,clip,start,z=0.,duration=15,q=None,bad=None,bursts=1):
    for row in rows(start,z,duration):
        if bad and bad(row[0]-start): row=(*row[:3],True)
        h.collect('session',clip,[row]*bursts,q or quality(),observed_at=row[0])


def history(z=1.,bad=None):
    h=CumulativeEEGHistory('session');h.start_clip('one',100.)
    collect(h,'one',100.)
    h.start_clip('two',116.);collect(h,'two',116.,z,bad=bad)
    h.start_clip('three',132.);collect(h,'three',132.,z,duration=3.5)
    return h


def comparison(h,z=1.,q=None,start=132.):
    return h.compare(rows(start,z,3.5),q or quality(),(start,start+3.5))


def prompt(policy):
    a=dict(eeg_policy=policy,characters={},valid_gaze_s=0,comparison_s=0)
    p,_=profile.update(profile.new_profile(['Pat','Bob']),a)
    hint=profile.decide(p,a)
    return director.template(dict(premise='Pat and Bob wave.'),hint,15,['Pat','Bob'])


def test_first_two_opening_decisions_abstain_without_disjoint_history():
    h=CumulativeEEGHistory('session');h.start_clip('one',100.)
    collect(h,'one',100.,2.,duration=3.5)
    first=comparison(h,2.,start=100.)
    assert first['action']=='keep' and first['delta'] is None
    assert prompt(first)['video_prompt']=='Pat and Bob wave.'
    collect(h,'one',100.,2.)
    h.start_clip('two',116.);collect(h,'two',116.,2.,duration=3.5)
    second=comparison(h,2.,start=116.)
    assert not second['eligible'] and second['delta'] is None


def test_opposite_sustained_shifts_change_exact_prompt_and_log_partition():
    for z,cue in [(1.,'quicken'),(-1.,'slowly')]:
        r=comparison(history(z),z);p=prompt(r)
        assert r['eligible'] and r['delta']==pytest.approx(z)
        assert cue in p['video_prompt'] and p['decision']['eeg_applied']
        assert r['policy']==VERSION and 'v2' in VERSION
        assert r['current_window'][1]-r['current_window'][0]==12.
        assert r['reference_window'][1] <= r['current_window'][0]-2.
        assert r['reference_coverage_s']>=12.
        assert r['current_effective_count']<=6 and r['reference_effective_count']<=7
        assert r['evidence_age_s']<=5 and r['confidence_basis'].startswith('feed')
        assert r['coverage']['current_fraction']>=.8
        assert all(abs(d)>=r['change_threshold'] for d in r['persistence_block_deltas'])


def test_tiny_change_no_longer_triggers_or_calls_a_classifier():
    r=comparison(history(.01),.01)
    assert r['eligible'] and r['action']=='keep' and r['change_threshold']>=.35
    assert prompt(r)['video_prompt']=='Pat and Bob wave.'


@pytest.mark.parametrize('failure',['missing','stale','uncalibrated','channel','baseline','identity','under60','latest_artifact'])
def test_invalid_current_or_identity_abstains(failure):
    h=history();q=quality()
    if failure=='missing':h._active['rows'].clear()
    if failure=='stale':q['sampleAgeSeconds']=4.
    if failure=='uncalibrated':q['calibrated']=False
    if failure=='channel':q['selectedChannels']=['TP9','AF8']
    if failure=='baseline':q['calibration']['calibratedAt']=110.
    if failure=='identity':q.pop('calibration')
    if failure=='under60':q['calibration']['cleanSeconds']=59.9
    if failure=='latest_artifact':h.collect('session','three',[(135.5,.6,1.,True)],q,observed_at=135.5)
    r=comparison(h,q=q)
    assert not r['eligible'] and r['action']=='keep' and r['delta'] is None
    assert 'EEG DELIVERY TRIAL' not in prompt(r)['video_prompt']


def test_insufficient_coverage_gaps_and_transients_do_not_change_prompt():
    for bad in (lambda t:True,lambda t:5<t<10):
        r=comparison(history(bad=bad))
        assert not r['eligible'] and r['action']=='keep'
    # Only the latest 3.5s changes; previous sustained window is neutral.
    h=history(0.);r=comparison(h,2.)
    assert r['action']=='keep'


def test_reference_freeze_caller_mutation_and_future_rows():
    h=history();r=comparison(h);snapshot=deepcopy(r)
    assert h.collect('session','two',[(131.,.6,-4.,False)],quality(),observed_at=135.)==0
    assert h.collect('other','three',[(135.5,.6,-4.,False)],quality(),observed_at=135.5)==0
    assert h.collect('session','three',[(140.,.6,-4.,False)],quality(),observed_at=135.5)==0
    r['reference_clip_ids'].append('fake')
    collect(h,'three',132.,-2.,duration=5.)
    assert comparison(h)==snapshot


def test_session_reset_and_changed_baseline_cannot_reuse_history():
    h=history();h.reset('new');h.start_clip('first',200.)
    assert not comparison(h,start=200.)['eligible']
    h=history();q=quality();q['calibration']['calibratedAt']=130.
    assert not comparison(h,q=q)['eligible']


def test_packet_duplicates_never_inflate_duration_or_independent_counts():
    h=history();r=comparison(h)
    burst=history();burst.collect('session','three',[(135.5,.6,1.,False)]*1000,quality(),observed_at=135.5)
    other=comparison(burst)
    assert other['current_effective_count']==r['current_effective_count']
    assert other['reference_count']==r['reference_count']
    assert other['coverage']==r['coverage']


def test_noise_threshold_scales_up_with_variable_reference():
    h=CumulativeEEGHistory('session');h.start_clip('one',100.)
    for i,row in enumerate(rows(100.)):
        h.collect('session','one',[(row[0],.6,.5 if i%2 else -.5,False)],quality(),observed_at=row[0])
    h.start_clip('two',116.);collect(h,'two',116.,.4)
    h.start_clip('three',132.);collect(h,'three',132.,.4,duration=3.5)
    r=comparison(h,.4)
    assert r['noise_floor']>comparison(history(.4),.4)['noise_floor']
    assert r['change_threshold']>.35 and r['action']=='keep'


def test_hysteresis_requires_current_persistence_and_invalid_resets():
    h=history(1.);first=comparison(h)
    assert first['action']=='faster_pacing'
    collect(h,'three',132.,1.5)
    h.start_clip('four',148.);collect(h,'four',148.,1.5,duration=3.5)
    retained=comparison(h,1.5,start=148.)
    # Retained threshold has explicit noise scaling and needs new evidence.
    assert retained['change_threshold']==pytest.approx(.6*retained['entry_threshold'])
    assert retained['eligible']
    h.start_clip('five',164.)
    assert not comparison(h,start=164.)['eligible'] and h._last_state is None


def test_result_ttl_and_raw_stale_gate_are_independent():
    h=history();h._active['rows'].clear()
    # Current source is live, but no viewing ledger refresh cannot reuse a tail.
    r=comparison(h)
    assert not r['eligible'] and 'fresh viewing' in r['reason']


@pytest.mark.parametrize('bad',[None,[['bad']],['AF7'],['AF7','AF7']])
def test_malformed_channel_identity_abstains_without_raising(bad):
    q=quality();q['calibration']['channels']=bad
    assert not comparison(history(),q=q)['eligible']
