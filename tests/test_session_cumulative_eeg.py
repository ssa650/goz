"""Actual session tick/freeze/prompt integration, offline mocked generation."""
import asyncio
from copy import deepcopy
import json
import time

import pytest

from backend.adaptive import tracking_diagnostics
from test_adaptive_deadline import clock, setup, tick
from test_cumulative_eeg_policy import quality
from test_eeg_two_clean_policy import two_clean_quality
from test_adaptive import png
from backend.frames import verify_image


def ready_quality(session, monkeypatch, q=None):
    q=quality() if q is None else q
    session.eeg.source='muse'
    session.eeg.calibration=deepcopy(q.pop('calibration'))
    monkeypatch.setattr(session.eeg,'status',lambda:deepcopy(q))
    return q


@pytest.mark.asyncio
@pytest.mark.parametrize('z,cue',[(1.,'quicken'),(-1.,'slowly')])
@pytest.mark.parametrize('channel_mode', ['strict', 'two_clean'])
async def test_sustained_prior_history_reaches_next_prompt_without_delaying_gaze_deadline(tmp_path,monkeypatch,clock,z,cue,channel_mode):
    s,first,engine,submitted=setup(tmp_path,monkeypatch,clock)
    ready_quality(s,monkeypatch, two_clean_quality() if channel_mode == 'two_clean' else quality(channelPolicy='strict'))
    s.max_scenes=4
    def boundary(clip,path):
        s.boundary_frames[clip['id']]=verify_image(png(),'mock-actual-last-frame.png')
        clip['boundaryFrameStatus']='ready'
        return None
    monkeypatch.setattr(s,'prepare_boundary',boundary)
    origin=s.started
    for scene,value in ((0,0.),(1,z)):
        clip=s.clips[scene]
        start=origin+16*scene
        for i in range(61):
            clock[0]=start+i*.25
            s.eeg.series.append((clock[0],.6,value,False))
            s.tick(scene,i*.25,True,{},clock[0],clip_id=clip['id'])
            if i==14:
                await asyncio.wait_for(s.task,5)
                policy=submitted[-1][0]['engagementDecision']['eeg_policy']
                assert not policy['eligible'] and policy['action']=='keep'
                assert 'EEG DELIVERY TRIAL' not in submitted[-1][0]['prompt']
                assert submitted[-1][0]['observationMs']==3500
                frozen=deepcopy(clip['frozenEvidence']['eegPolicy'])
        assert clip['frozenEvidence']['eegPolicy']==frozen
        s.ended(scene)
    third=s.clips[2]
    start=origin+32
    for i in range(15):
        clock[0]=start+i*.25
        s.eeg.series.append((clock[0],.6,z,False))
        s.tick(2,i*.25,True,{},clock[0],clip_id=third['id'])
    frozen3=deepcopy(third['frozenEvidence']['eegPolicy'])
    await asyncio.wait_for(s.task,5)
    policy=submitted[-1][0]['engagementDecision']['eeg_policy']
    assert policy['eligible'] and policy['delta']*z>0
    assert policy['current_window'][1]-policy['current_window'][0]==12
    assert policy['reference_window'][1]<=policy['current_window'][0]-2
    assert policy['reference_coverage_s']>=12
    assert policy['coverage']['current_fraction']>=.8
    assert cue in submitted[-1][0]['prompt']
    assert submitted[-1][0]['observationMs']==3500 and len(submitted)==3
    if channel_mode == 'two_clean':
        assert policy['confidence'] == policy['confidence_threshold'] == .5
        await engine.trace_journal.flush()
        saved = engine.trace_journal.read(s.id)
        eligible = [r['evidence']['eeg'] for r in saved if r['evidence']['eeg']['eligible']]
        assert eligible and eligible[-1]['selected_channels'] == ['AF7', 'AF8']
        assert eligible[-1]['channel_policy'] == 'two_clean'
        assert eligible[-1]['channel_coverage'] == eligible[-1]['confidence_threshold'] == .5
        assert eligible[-1]['reduced_redundancy'] and eligible[-1]['channel_eligibility_reason']
        assert eligible[-1]['coverage']['prior_clean_s'] >= 12
    clock[0]=start+4
    s.eeg.series.append((clock[0],.6,-z,False))
    s.tick(2,4,True,{},clock[0],clip_id=third['id'])
    assert third['frozenEvidence']['eegPolicy']==frozen3
    await tracking_diagnostics.for_clip(s,third).flush()
    logs=tracking_diagnostics.read(tmp_path,s.id)
    assert any(e['event']=='eeg_cumulative_frozen' and e['policy']['eligible'] for e in logs)
    assert s.public()['eegRunMode']=='cumulative_prior_clips'
    assert json.loads((s.dir/'session.json').read_text())['eegRunMode']=='cumulative_prior_clips'
    await engine.close()


@pytest.mark.asyncio
async def test_pause_hold_and_delayed_ticks_cannot_collect_earlier_rows_with_current_quality(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock,final=True)
    ready_quality(s,monkeypatch)
    tick(s,clock,0)
    s.eeg.series.append((s.started+2,.6,0.,False))
    tick(s,clock,2,playing=False)
    assert s.eeg_history._active['rows']=={}
    # The report is one second old; current quality cannot certify its samples.
    clock[0]=s.started+3
    s.tick(0,2,True,{},s.started+2,clip_id=clip['id'])
    assert s.eeg_history._active['rows']=={}
    s.status='stopped'
    clock[0]=s.started+3.25
    s.eeg.series.append((clock[0],.6,0.,False))
    s.tick(0,3.25,True,{},clock[0],clip_id=clip['id'])
    assert s.eeg_history._active['rows']=={}
    await engine.close()


@pytest.mark.asyncio
async def test_quality_snapshot_does_not_mutate_feed_calibration_and_mode_is_explicit(tmp_path,monkeypatch,clock):
    s,_,engine,_=setup(tmp_path,monkeypatch,clock,eeg_run_mode='baseline')
    ready_quality(s,monkeypatch)
    before=deepcopy(s.eeg.calibration)
    copied=s.eeg_quality_snapshot()
    copied['calibration']['channelMedians'][0]=99
    assert s.eeg.calibration==before
    assert s.public()['eegRunMode']=='baseline'
    assert json.loads((s.dir/'session.json').read_text())['eegRunMode']=='baseline'
    await engine.close()
