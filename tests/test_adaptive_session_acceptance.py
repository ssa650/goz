"""Session lifecycle and failure-path tests; uses local media, not live evidence."""
import asyncio
import time

import pytest
from backend.adaptive import director
from backend.adaptive.session import AdaptiveSession
from backend.adaptive.sensors import GazeFeed, EegFeed
from backend.demo import DemoAdapter
from backend.engine import Engine


def session(engine):
    return AdaptiveSession(engine,GazeFeed(),EegFeed(),engine.directory,'SpongeBob and Patrick open a box.',
        [{'name':'SpongeBob'},{'name':'Patrick'}],5,'480P',None)


@pytest.mark.asyncio
async def test_duplicate_slow_generation_and_future_queue_are_bounded(tmp_path, monkeypatch):
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s=session(engine)
    original=engine.run_job
    entered,release=asyncio.Event(),asyncio.Event()
    calls=[]
    async def slow(job,images):
        calls.append(job['id']);entered.set();await release.wait();return await original(job,images)
    monkeypatch.setattr(engine,'run_job',slow)
    first=asyncio.create_task(s.make_scene({},None,None,[]))
    await entered.wait()
    await asyncio.gather(*(s.make_scene({},None,None,[]) for _ in range(5)))
    assert len(s.clips)==1 and len(calls)==1
    release.set();await first
    await s.make_scene({},None,None,[])
    assert len(s.clips)==1, 'cannot fill queue before current media is watched'
    s.clips[0]['status']='playing'
    await s.make_scene({},None,None,[])
    await s.make_scene({},None,None,[])
    assert len(s.clips)==2 and len(calls)==2
    assert len(s.readiness_samples)==2
    assert s.clips[1]['decisionId']==engine.jobs[s.clips[1]['jobId']]['decisionId']
    await engine.close()


@pytest.mark.asyncio
async def test_stop_drops_late_generation_and_old_detection(tmp_path, monkeypatch):
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s=session(engine)
    entered,release=asyncio.Event(),asyncio.Event()
    async def late(job,images):
        entered.set();await release.wait();job.update(status='completed')
    monkeypatch.setattr(engine,'run_job',late)
    task=asyncio.create_task(s.make_scene({},None,None,[]));await entered.wait();s.stop();release.set();await task
    assert s.status=='stopped' and not s.clips[0].get('url') and not s.story['scenes']
    replacement=session(engine)
    assert replacement.id!=s.id and replacement.clips==[] and replacement.live_gaze() is None
    await engine.close()


@pytest.mark.asyncio
async def test_detection_is_not_in_media_readiness_critical_path(tmp_path,monkeypatch):
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s=session(engine)
    await s.make_scene({},None,None,[])
    clip=s.clips[0]
    engine.adapter.demo=False
    entered,release=asyncio.Event(),asyncio.Event()
    async def slow(*args):
        entered.set();await release.wait();return [dict(t=0,boxes={})]
    monkeypatch.setattr(s,'track',slow)
    s.start_detection(clip,clip['path'],None,None,5)
    await entered.wait()
    assert clip['status']=='ready' and clip['detectionStatus']=='processing'
    s.stop();release.set();await asyncio.gather(*s.detection_tasks,return_exceptions=True)
    assert clip['detectionStatus']=='processing', 'cancelled detections must not mutate session'
    await engine.close()


@pytest.mark.asyncio
async def test_clip_identity_and_seek_epoch_are_retained(tmp_path):
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s=session(engine);await s.make_scene({},None,None,[])
    with pytest.raises(ValueError,match='identity'):
        s.tick(0,0,True,{},time.time(),clip_id='old')
    s.tick(0,0,True,dict(x=0,y=0,w=100,h=100),time.time(),clip_id=s.clips[0]['id'],epoch=7)
    assert s.clips[0]['ticks'][-1]['epoch']==7
    assert s.clips[0]['ticks'][-1]['sessionId']==s.id
    await engine.close()


@pytest.mark.asyncio
async def test_delayed_old_clip_tick_cannot_revert_current_clip(tmp_path):
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s=session(engine);await s.make_scene({},None,None,[])
    s.clips[0]['status']='playing'
    await s.make_scene({},None,None,[])
    rect=dict(x=0,y=0,w=100,h=100)
    s.tick(1,0,True,rect,time.time(),clip_id=s.clips[1]['id'])
    s.tick(0,1,True,rect,time.time(),clip_id=s.clips[0]['id'])
    assert s.playing==1 and not s.clips[0]['ticks']
    await engine.close()


def policy_trigger_session(engine, clock):
    """Real fusion/policy data with a controlled clock; no generation requested."""
    s=session(engine)
    s.duration=20
    s.started=clock[0]
    s.max_scenes=1  # Real adapt() analyzes this final scene without making a paid request.
    boxes={'SpongeBob':[.05,.2,.4,.9],'Patrick':[.6,.2,.95,.9]}
    clip=dict(id='trigger-clip',index=0,status='playing',duration=20,ticks=[],
              playStartedAt=clock[0],detectionStatus='ready',
              track=[dict(t=i/4,boxes=boxes) for i in range(81)],plan={'beats':[]})
    s.clips=[clip]
    s.playing=0
    return s,clip


def captured_tick(s, clock, relative, x=50, y=5):
    clock[0]=s.started+relative
    s.gaze.add(dict(t=clock[0],x=x,y=y,valid=True,face=True,confidence=1,blink=False,yaw=0))
    s.tick(0,relative,True,dict(x=0,y=0,w=100,h=100),clock[0],clip_id=s.clips[0]['id'])


@pytest.mark.asyncio
async def test_comparative_visibility_does_not_extend_deadline_for_late_dwell(tmp_path,monkeypatch):
    from copy import deepcopy
    from backend.adaptive import fusion, profile
    clock=[1_800_000_000.0]
    monkeypatch.setattr(time,'time',lambda:clock[0])
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s,clip=policy_trigger_session(engine,clock)
    original=deepcopy(s.profile)
    # Both actors are observable, but the viewer makes only brief, evenly
    # divided looks at either actor; most valid gaze stays on background.
    for i in range(63):
        phase=(i%20)/20
        x=20 if phase<.15 else 80 if phase<.3 else 50
        captured_tick(s,clock,i/20,x,50 if x!=50 else 5)
    analysis=fusion.analyze(fusion.label(list(s.gaze.samples),clip['ticks'],clip['track']),[],clip['track'],s.names)
    assert analysis['comparison_s']>=profile.MIN_COMPARISON_S
    assert max(c['comparison_dwell_s'] for c in analysis['characters'].values())<profile.MIN_DWELL_S
    assert not clip.get('analysisStarted') and s.task is None
    assert s.profile==original, 'previewing must not learn or increment real profile clips'

    # Sustained Patrick gaze starts too late to qualify by the deadline.
    for i in range(63,161):
        captured_tick(s,clock,i/20,80,50)
        if clip.get('analysisStarted'):
            break
    assert clip.get('analysisStarted')
    assert s.profile==original, 'only adapt(), not the synchronous preview, learns'
    await s.task
    assert clip['frozenEvidence']['end']-s.started==pytest.approx(3.5)
    assert s.profile['policy']['focus'] is None, 'late dwell cannot extend the frozen window'
    assert s.profile['clips']==1 and not engine.jobs
    await engine.close()


@pytest.mark.asyncio
async def test_deadline_can_commit_balanced_when_no_action_ever_becomes_valid(tmp_path,monkeypatch):
    clock=[1_800_000_000.0]
    monkeypatch.setattr(time,'time',lambda:clock[0])
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s,clip=policy_trigger_session(engine,clock)
    for i in range(70):
        captured_tick(s,clock,i/20)  # Valid background, no target preference.
    assert not clip.get('analysisStarted')
    captured_tick(s,clock,3.5)
    assert clip.get('analysisStarted')
    await s.task
    assert s.profile['policy']['focus'] is None
    assert s.profile['characters']=={'SpongeBob':.5,'Patrick':.5}
    assert not engine.jobs
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('readability',['pacing','dialogue'])
async def test_deadline_supports_readability_with_preserved_absolute_evidence(tmp_path,monkeypatch,readability):
    clock=[1_800_000_000.0]
    monkeypatch.setattr(time,'time',lambda:clock[0])
    engine=Engine(DemoAdapter(tmp_path/'demo'),tmp_path,poll_seconds=.01)
    s,clip=policy_trigger_session(engine,clock)
    if readability=='dialogue':
        clip['plan']['beats']=[dict(t0=0,t1=2,dialogue=True,speaker='SpongeBob'),
                              dict(t0=2,t1=20,dialogue=False,speaker=None)]
    for i in range(121):
        outside=readability=='pacing' or i<40
        captured_tick(s,clock,i/20,-20 if outside else 50,5)
        if clip.get('analysisStarted'):
            break
    assert clip.get('analysisStarted'), 'readability adjustments must not wait for a character preference'
    assert s.profile['clips']==0
    await s.task
    assert clip['analysis']['valid_gaze_s']<=3.5
    assert s.profile[readability]<0, 'adequate absolute readability evidence fits the opening window'
    assert s.profile['policy']['focus'] is None
    assert not engine.jobs
    await engine.close()


@pytest.mark.asyncio
async def test_continuation_readiness_includes_extraction_composition_and_engine(tmp_path,monkeypatch):
    """Known stage delays must all reach the scheduling estimate and saved job."""
    from pathlib import Path
    from test_backend import FakeAdapter
    from backend.adaptive import director
    clock=[100.0]
    monkeypatch.setattr(time,'perf_counter',lambda:clock[0])
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.01)
    s=session(engine)
    s.clips=[dict(id='source',index=0,status='watched',duration=20,ticks=[],track=[],
                  path='source.mp4',plan=dict(beats=[]),endedAt=time.time(),detectionStatus='ready')]
    s.story['scenes']=[dict(title='Opening',summary='SpongeBob and Patrick find a box.')]
    async def extract(path):
        from backend.frames import verify_image
        from test_adaptive import png
        assert str(path) in ('source.mp4', 'validated.mp4')
        clock[0]+=3.6 if str(path)=='source.mp4' else .2
        return verify_image(png())
    original_write=director.write_scene
    async def compose(*args,**kwargs):
        clock[0]+=.4
        return await original_write(*args,**kwargs)
    async def generate(job,images):
        clock[0]+=6
        job.update(status='completed',generationInput={'prompt':job['prompt']})
    async def media(job):
        clock[0]+=.5
        return Path('validated.mp4')
    monkeypatch.setattr(engine,'extractor',extract)
    monkeypatch.setattr(director,'write_scene',compose)
    monkeypatch.setattr(engine,'run_job',generate)
    monkeypatch.setattr(engine,'media_path',media)
    monkeypatch.setattr(s,'start_detection',lambda *args:None)
    await s.adapt(0)
    assert s.status=='running',s.error
    continuation=s.clips[1]
    job=engine.jobs[continuation['jobId']]
    assert continuation['frameExtractionMs']==pytest.approx(3600)
    assert continuation['compositionMs']==pytest.approx(400)
    assert continuation['continuationReadyMs']==pytest.approx(10700)
    assert continuation['generatedS']==pytest.approx(10.7)
    assert job['frameExtractionMs']==pytest.approx(3600)
    assert job['continuationReadyMs']==pytest.approx(10700)
    assert continuation['timing']['continuationReadyMs']==pytest.approx(10700)
    assert s.readiness_samples==[10.7]
    assert s.analyze_at({'duration':20})==pytest.approx(3.5)
    assert 'continuationStarted' not in job, 'process-local monotonic timestamps must not be persisted'
    await engine.close()
