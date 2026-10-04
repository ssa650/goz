"""Bounded session integration; fake generation, no app/device/provider calls."""
import asyncio
from copy import deepcopy
import json

import pytest
from backend.adaptive import local_tracker
from test_adaptive_deadline import clock, setup, tick
from test_cumulative_eeg_policy import quality


class Worker:
    def __init__(self):
        self.requested = 0
        self.closed = False
    def request_close(self):
        self.requested += 1
    async def close(self):
        await asyncio.sleep(0)
        self.closed = True


@pytest.mark.asyncio
async def test_initial_warm_is_before_generation_and_same_worker_reaches_every_clip(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    s.tracker='color'; worker=Worker();entered=asyncio.Event();release=asyncio.Event();calls=[]
    async def warm(provider,**kw):
        calls.append(('warm',provider));kw['on_diagnostic'](dict(diagnostic_event='warming',provider=provider))
        entered.set();await release.wait()
        kw['on_diagnostic'](dict(diagnostic_event='warm_ready',provider=provider,worker_pid=123))
        return worker
    async def generate(*args,**kw):
        assert s.local_worker is worker;calls.append(('generation',None))
    async def detect(path,names,**kw):
        assert kw['warm_worker'] is worker
        assert kw['provider']=='color' and kw['session_id']==s.id
        calls.append(('track',kw['clip_id']));return []
    monkeypatch.setattr(local_tracker,'prewarm_local_tracker',warm)
    monkeypatch.setattr(local_tracker,'detect_local',detect)
    monkeypatch.setattr(s,'make_scene',generate)
    await s.start();await asyncio.wait_for(entered.wait(),2)
    assert calls==[('warm','color')] and s.public()['trackingReadiness']['state']=='warming'
    release.set();await asyncio.wait_for(s.task,2)
    assert calls[:2]==[('warm','color'),('generation',None)]
    await s.track('one.mp4',None,None,clip_id=clip['id'])
    second=dict(id='second',jobId='generation2',index=1,status='ready');s.clips.append(second)
    await s.track('two.mp4',None,None,clip_id='second')
    assert calls[-2:]==[('track','source'),('track','second')]
    assert sum(c[0]=='warm' for c in calls)==1
    s.stop();await asyncio.wait_for(s.local_worker_close_task,2)
    assert worker.closed and worker.requested==1 and s.public()['trackingReadiness']['state']=='closed'
    await engine.close()


@pytest.mark.asyncio
async def test_stop_during_initial_warm_never_starts_generation(tmp_path,monkeypatch,clock):
    s,_,engine,_=setup(tmp_path,monkeypatch,clock);entered=asyncio.Event();cancelled=asyncio.Event();submitted=[]
    async def warm(*args,**kw):
        entered.set()
        try: await asyncio.Future()
        finally: cancelled.set()
    async def generate(*args,**kw):submitted.append(True)
    monkeypatch.setattr(local_tracker,'prewarm_local_tracker',warm)
    monkeypatch.setattr(s,'make_scene',generate)
    await s.start();await asyncio.wait_for(entered.wait(),2)
    s.stop();await asyncio.wait_for(cancelled.wait(),2)
    assert s.task.cancelled() and not submitted and s.tracking_readiness['state']=='cancelled'
    await engine.close()


@pytest.mark.asyncio
async def test_warm_failure_does_not_spawn_cold_tracking_or_block_generation(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock);s.tracker='color';generated=[]
    async def warm(*args,**kw):raise RuntimeError('offline warm failure')
    async def generate(*args,**kw):generated.append(True)
    async def detect(*args,**kw):pytest.fail('cold playback tracking must not start')
    monkeypatch.setattr(local_tracker,'prewarm_local_tracker',warm)
    monkeypatch.setattr(local_tracker,'detect_local',detect)
    monkeypatch.setattr(s,'make_scene',generate)
    await s.start();await asyncio.wait_for(s.task,2)
    assert generated and s.tracking_readiness['state']=='failed'
    assert await s.track('one.mp4',None,None,clip_id=clip['id'])==[]
    assert 'no cold worker' in clip['localTrackingError']
    s.stop();await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal',['failed','finished'])
async def test_terminal_paths_request_cooperative_close_without_wait(tmp_path,monkeypatch,clock,terminal):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock,final=True);worker=Worker();s.local_worker=worker
    s.tracking_readiness={'state':'ready'}
    if terminal=='failed':s.fail(RuntimeError('generation failed'))
    else:
        clip.update(analysisStarted=True,analysis={'samples':1});s.ended(0)
    assert s.status==terminal and worker.requested==1
    await asyncio.wait_for(s.local_worker_close_task,2)
    assert worker.closed
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['baseline','cumulative_prior_clips'])
async def test_frozen_gaze_only_abstains_both_eeg_policies_and_retains_gaze_focus(tmp_path,monkeypatch,clock,mode):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock,eeg_run_mode=mode,eeg_gaze_only=True)
    q=quality();s.eeg.calibration=deepcopy(q['calibration']);s.eeg.source='muse'
    monkeypatch.setattr(s.eeg,'status',lambda:deepcopy(q))
    feed_before=deepcopy(s.eeg.calibration)
    boxes={'Ana':[.05,.1,.4,.9],'Bea':[.6,.1,.95,.9]}
    clip['track']=[dict(t=i/8,valid_until=(i+1)/8,boxes=boxes,clip_id=clip['id'],session_id=s.id) for i in range(29)]
    for i in range(71):
        clock[0]=s.started+i/20
        if i%5==0:s.eeg.series.append((clock[0],.6,1.5,False))
        s.gaze.add(dict(t=clock[0],x=75,y=50,valid=True,face=True,confidence=.9,yaw=0))
        tick(s,clock,i/20)
    frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,5)
    decision=submitted[0][0]['engagementDecision']
    assert decision['focus']=='Bea' and not decision['eeg_applied']
    assert not decision['eeg_policy']['eligible'] and decision['eeg_policy']['action']=='keep'
    assert 'PRIMARY SHOT: Bea' in submitted[0][0]['prompt'] and 'EEG DELIVERY TRIAL' not in submitted[0][0]['prompt']
    assert frozen['quality']['live'] is False and frozen['quality']['confidence']==0
    assert frozen['quality']['measuredLive'] is True and frozen['quality']['measuredConfidence']==.9
    s.eeg.calibration['channelMedians'][0]=99
    assert clip['frozenEvidence']==frozen and q['calibration']==feed_before
    assert s.public()['eegGazeOnly'] is True and json.loads((s.dir/'session.json').read_text())['eegGazeOnly'] is True
    other,_,other_engine,_=setup(tmp_path/'other',monkeypatch,clock,eeg_run_mode=mode)
    monkeypatch.setattr(other.eeg,'status',lambda:deepcopy(q))
    assert other.eeg_quality_snapshot()['live'] is True and other.eeg_quality_snapshot()['confidence']==.9
    assert not other.eeg_gaze_only
    await other_engine.close();await engine.close()
