"""Free deadline/lifecycle regressions. No hardware or remote generation calls."""
import asyncio
from copy import deepcopy
from pathlib import Path
import time

import httpx
import pytest

from backend.adaptive import director, tracks
from backend.adaptive.sensors import GazeFeed, EegFeed
from backend.adaptive.session import AdaptiveSession, OBSERVATION_SECONDS
from backend.engine import Engine
from backend.frames import verify_image
from test_adaptive import png
from test_backend import FakeAdapter


@pytest.fixture
def clock(monkeypatch):
    clock = [1_800_000_000.0]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    return clock


def setup(tmp_path, monkeypatch, clock, *, final=False):
    engine = Engine(FakeAdapter(), tmp_path, poll_seconds=.001)
    s = AdaptiveSession(engine, GazeFeed(), EegFeed(), tmp_path, 'Two explorers find a box.',
                        [dict(name='Ana'), dict(name='Bea')], 15, '480P', verify_image(png()))
    s.max_scenes = 1 if final else 2
    clip = dict(id='source', sessionId=s.id, index=0, status='ready', duration=15,
                ticks=[], track=[], detectionStatus='processing', path='source.mp4', plan=dict(beats=[]))
    s.clips = [clip]
    s.story['scenes'] = [dict(title='Opening', summary='The explorers see a box.')]
    s.boundary_frames[clip['id']] = verify_image(png(), 'actual-last-frame.png')
    clip['boundaryFrameStatus'] = 'ready'
    submitted = []
    async def generate(job, images):
        submitted.append((deepcopy(job), images))
        job.update(status='completed', apiStartedAt=time.time()*1000)
    async def media(job):
        return Path('successor.mp4')
    monkeypatch.setattr(engine, 'run_job', generate)
    monkeypatch.setattr(engine, 'media_path', media)
    monkeypatch.setattr(s, 'start_detection', lambda *args: None)
    return s, clip, engine, submitted


def tick(s, clock, elapsed, playing=True):
    clock[0] = s.started + elapsed
    s.tick(0, elapsed, playing, dict(x=0,y=0,w=100,h=100), clock[0], clip_id='source')


@pytest.mark.asyncio
@pytest.mark.parametrize('detection_status', ['processing', 'unavailable', 'ready'])
async def test_deadline_submits_with_missing_evidence_before_end(tmp_path, monkeypatch, clock, detection_status):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    clip['detectionStatus'] = detection_status
    tick(s, clock, 0)
    tick(s, clock, OBSERVATION_SECONDS-.01)
    assert not clip.get('analysisStarted') and not submitted
    tick(s, clock, OBSERVATION_SECONDS)
    await asyncio.wait_for(s.task, .5)
    assert len(submitted) == 1 and clip['status'] == 'playing' and 'endedAt' not in clip
    assert s.clips[1]['status'] == 'ready'
    decision = submitted[0][0]['engagementDecision']
    assert decision['focus'] is None and decision['action'] == 'keep'
    assert 'Insufficient' in ' '.join(decision['reasons'])
    assert clip['analysis']['eeg_available'] is False
    assert decision['observationWindow']['end'] - decision['observationWindow']['start'] == 3.5
    job = engine.jobs[s.clips[1]['jobId']]
    assert job['observationToSubmitMs'] == 3500 and job['observationToReadyMs'] == 3500
    assert job['observationMs'] == 3500
    assert not engine.adapter.submissions
    await engine.close()


@pytest.mark.asyncio
async def test_loading_time_does_not_consume_playback_window(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    created = s.started
    clock[0] += 60  # Opening generation/download/preload took a minute.
    playback_start = clock[0]
    s.tick(0, 0, False, {}, clock[0], clip_id='source')
    assert not clip.get('analysisStarted')
    s.tick(0, 0, True, {}, clock[0]+.01, clip_id='source')
    playback_start += .01
    clock[0] = playback_start + 3.49
    s.tick(0, 3.49, True, {}, clock[0], clip_id='source')
    assert not submitted and not clip.get('analysisStarted')
    clock[0] = playback_start + 3.5
    s.tick(0, 3.5, True, {}, clock[0], clip_id='source')
    await s.task
    window = submitted[0][0]['observationWindow']
    assert window['start'] == playback_start and window['start'] >= created+60
    assert window['end'] - window['start'] == 3.5
    await engine.close()


@pytest.mark.asyncio
async def test_freeze_is_synchronous_and_late_evidence_cannot_change_decision(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    tick(s, clock, 0)
    tick(s, clock, 3.5)
    frozen = deepcopy(clip['frozenEvidence'])
    # Complete detection and receive delayed samples before adapt() gets CPU.
    clip.update(track=[dict(t=0,boxes={'Bea':[0,0,1,1]})], detectionStatus='ready')
    s.gaze.add(dict(t=s.started+2,x=50,y=50,valid=True,confidence=1))
    tick(s, clock, 8)
    await s.task
    assert clip['frozenEvidence'] == frozen
    assert clip['analysis']['samples'] == 0 and submitted[0][0]['engagementDecision']['focus'] is None
    assert submitted[0][0]['observationWindow']['end'] == s.started+3.5
    # Repeated end/tick/adapt events never add another paid job.
    s.ended(0); s.ended(0)
    await s.adapt(0)
    assert len(submitted) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_available_partial_valid_gaze_can_drive_focus_at_deadline(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    boxes = {'Ana':[0,.2,.4,1], 'Bea':[.6,.2,1,1]}
    clip['track'] = [dict(t=i/4, boxes=boxes, clip_id='source', session_id=s.id) for i in range(15)]
    for i in range(71):
        clock[0] = s.started+i/20
        s.gaze.add(dict(t=clock[0],x=80,y=50,valid=True,face=True,confidence=1,blink=False,yaw=0))
        tick(s, clock, i/20)
    assert clip['detectionStatus'] == 'processing'
    await s.task
    assert submitted[0][0]['engagementDecision']['focus'] == 'Bea'
    assert clip['analysis']['comparison_s'] >= 2
    assert s.profile['characters']['Bea'] > s.profile['characters']['Ana']
    await engine.close()


@pytest.mark.asyncio
async def test_stale_sensors_and_crossclip_boxes_have_neutral_fallback(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    clip['track'] = [dict(t=i/4,boxes={'Ana':[0,0,1,1]},clip_id='wrong',session_id='old') for i in range(15)]
    s.gaze.add(dict(t=s.started-1,x=50,y=50,valid=True,confidence=1))
    s.eeg.series.append((s.started-1,1,20,False))
    tick(s, clock, 0); tick(s, clock, 3.5)
    await s.task
    assert clip['frozenEvidence']['track'] == []
    assert clip['analysis']['valid_gaze_s'] == 0 and not clip['analysis']['eeg_available']
    assert submitted[0][0]['engagementDecision']['focus'] is None
    await engine.close()


@pytest.mark.asyncio
async def test_short_clip_end_is_bounded_fallback_and_duplicate_end_is_safe(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    clip['duration'] = 2
    tick(s, clock, 0); tick(s, clock, 1.9)
    clock[0] = s.started+2
    s.ended(0); s.ended(0)
    await s.task
    assert len(submitted) == 1 and submitted[0][0]['observationWindow']['end'] == s.started+2
    await engine.close()


@pytest.mark.asyncio
async def test_scene_change_cancels_obsolete_planning_and_releases_owner(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    entered, release = asyncio.Event(), asyncio.Event()
    async def compose(story, profile, decision, duration):
        entered.set(); await release.wait()
        return director.template(story, decision, duration, s.names), 'test'
    monkeypatch.setattr(director, 'write_scene', compose)
    tick(s, clock, 0); tick(s, clock, 3.5)
    old = s.task
    await entered.wait()
    s.clips.append(dict(id='new',index=1,status='ready',duration=15,ticks=[],track=[]))
    clock[0] += .01
    s.tick(1,0,True,{},clock[0],clip_id='new')
    await asyncio.gather(old, return_exceptions=True)
    assert old.cancelled() and not submitted and not s.generation_lock.locked()
    release.set()
    assert s.status == 'running' and s.playing == 1
    await engine.close()


@pytest.mark.asyncio
async def test_stop_cancels_pending_adaptation_and_late_completion_is_not_playable(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted = setup(tmp_path, monkeypatch, clock)
    entered, release = asyncio.Event(), asyncio.Event()
    async def pending(job, images):
        submitted.append(job); entered.set(); await release.wait(); job.update(status='completed')
    monkeypatch.setattr(engine, 'run_job', pending)
    tick(s, clock, 0); tick(s, clock, 3.5)
    old = s.task
    await entered.wait()
    s.stop(); release.set()
    await asyncio.gather(old, return_exceptions=True)
    await asyncio.sleep(0)
    assert s.status == 'stopped' and old.cancelled() and not s.generation_lock.locked()
    assert not s.clips[1].get('url') and len(submitted) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_partial_detection_publishes_before_slow_frame_and_verifier(tmp_path, monkeypatch):
    import backend.adaptive.identity as identity
    frame0, frame2 = dict(t=0,jpeg=b'a',width=100,height=100,shot=0,cut=False), dict(t=2,jpeg=b'b',width=100,height=100,shot=0,cut=False)
    monkeypatch.setattr(tracks, 'sample_scene_frames', lambda *a:[frame0,frame2])
    box = [0,.1,.3,.9]
    def parse(*args):
        return {'Ana':box}, [dict(identity='Ana',identity_status='verified_reference',box=box)]
    monkeypatch.setattr(tracks, 'parse_regions', parse)
    release, observed = asyncio.Event(), asyncio.Event()
    updates = []
    async def handler(request):
        if 'Yg==' in request.content.decode():
            await release.wait()
        return httpx.Response(200,json={'results':{'bboxes':[]}})
    # Supply a local transport without requesting fixture-mode identity behavior.
    original_client = httpx.AsyncClient
    monkeypatch.setattr(tracks.httpx, 'AsyncClient', lambda **kwargs:original_client(**{**kwargs,'transport':httpx.MockTransport(handler)}))
    async def verify(*args):
        await release.wait()
        return None, {'status':'unavailable'}
    monkeypatch.setattr(identity,'verify_frames',verify)
    def progress(result):
        updates.append(deepcopy(result))
        if result and result[0]['boxes']:
            observed.set()
    work = asyncio.create_task(tracks.detect_characters('unused', [dict(name='Ana')], 'test', directory=tmp_path,
        clip_id='source',session_id='session',on_progress=progress))
    await asyncio.wait_for(observed.wait(),.5)
    assert not work.done() and updates[-1][0]['boxes'] == {'Ana':box}
    assert updates[-1][0]['clip_id'] == 'source'
    release.set()
    result = await work
    assert len(result) == 2


@pytest.mark.asyncio
async def test_pending_caption_identity_is_not_published_as_verified(tmp_path, monkeypatch):
    import backend.adaptive.identity as identity
    monkeypatch.setattr(tracks,'sample_scene_frames',lambda *a:[dict(t=0,jpeg=b'a',width=100,height=100,shot=0,cut=False)])
    monkeypatch.setattr(tracks,'parse_regions',lambda *a:({'Ana':[0,0,1,1]},[dict(identity='Ana',identity_status='model_observed')]))
    original_client = httpx.AsyncClient
    monkeypatch.setattr(tracks.httpx,'AsyncClient',lambda **kwargs:original_client(**{**kwargs,'transport':httpx.MockTransport(lambda request:httpx.Response(200,json={}))}))
    release, observed = asyncio.Event(), asyncio.Event()
    async def verify(*args):
        await release.wait(); return None, {'status':'unavailable'}
    monkeypatch.setattr(identity,'verify_frames',verify)
    updates=[]
    def progress(result):
        updates.append(deepcopy(result))
        if result: observed.set()
    work=asyncio.create_task(tracks.detect_characters('unused',[dict(name='Ana')],'test',clip_id='source',on_progress=progress))
    await asyncio.wait_for(observed.wait(),.5)
    assert updates[-1][0]['boxes'] == {} and not work.done()
    release.set(); assert (await work)[0]['boxes'] == {}


@pytest.mark.asyncio
async def test_actual_file_end_is_precomputed_and_used_instead_of_early_frame(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from backend.frames import ffmpeg
    video = tmp_path/'two-colors.mp4'
    await ffmpeg('-f','lavfi','-i','color=c=red:s=64x64:r=10:d=2',
        '-f','lavfi','-i','color=c=blue:s=64x64:r=10:d=2',
        '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[out]',
        '-map','[out]','-pix_fmt','yuv420p',video)
    engine = Engine(FakeAdapter(), tmp_path/'data', poll_seconds=.001)
    s = AdaptiveSession(engine,GazeFeed(),EegFeed(),engine.directory,'Explorers find a box.',
                        [dict(name='Ana'),dict(name='Bea')],15,'480P',verify_image(png()),opening_video=video)
    s.max_scenes = 2
    monkeypatch.setattr(s,'start_detection',lambda *a:None)
    calls=[]
    original = engine.extractor
    async def extract(path):
        calls.append(path); return await original(path)
    monkeypatch.setattr(engine,'extractor',extract)
    await s.predefined_scene(dict(id='opening'))
    clip = s.clips[0]
    assert clip['status']=='ready' and clip['boundaryFrameStatus']=='ready'
    boundary=s.boundary_frames[clip['id']]
    rgb=Image.open(BytesIO(boundary.data)).convert('RGB').getpixel((32,32))
    assert rgb[2]>200 and rgb[0]<30, 'the actual final frame is blue, unlike the red opening'
    submitted=[]
    async def generate(job, images):
        submitted.append(images); job.update(status='completed')
    async def media(job): return video
    monkeypatch.setattr(engine,'run_job',generate);monkeypatch.setattr(engine,'media_path',media)
    now=time.time()
    s.tick(0,0,True,{},now,clip_id=clip['id'])
    s.tick(0,3.5,True,{},now+.01,clip_id=clip['id'])
    await s.task
    assert submitted[0]['start'][0] is boundary
    assert calls==[video], 'deadline does not decode the frame again'
    assert s.clips[1]['status']=='ready'
    await engine.close()


@pytest.mark.asyncio
async def test_boundary_failure_never_substitutes_opening_frame(tmp_path, monkeypatch, clock):
    s, clip, engine, submitted=setup(tmp_path,monkeypatch,clock)
    s.boundary_frames.clear()
    async def broken(path): raise ValueError('decode failed')
    monkeypatch.setattr(engine,'extractor',broken)
    tick(s,clock,0);tick(s,clock,3.5)
    await s.task
    assert s.status=='failed' and not submitted
    assert clip['boundaryFrameStatus']=='failed'
    assert 'actual final frame' in s.error
    await engine.close()


@pytest.mark.asyncio
async def test_cancelled_detection_does_not_strand_running_playback(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    # Restore real scheduler while mocking its slow detector only.
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered=asyncio.Event()
    async def detect(*args):
        entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(s,'track',detect)
    s.start_detection(clip,'source.mp4',None,None,15)
    await entered.wait()
    tasks=list(s.detection_tasks)
    for task in tasks:task.cancel()
    await asyncio.gather(*tasks,return_exceptions=True)
    monkeypatch.setattr(s,'start_detection',lambda *a:None)
    tick(s,clock,0);tick(s,clock,3.5)
    await s.task
    assert s.status=='running' and s.clips[1]['status']=='ready' and len(submitted)==1
    await engine.close()


@pytest.mark.asyncio
async def test_late_detection_results_after_scene_change_are_ignored(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered,release=asyncio.Event(),asyncio.Event()
    async def detect(path,seeds,focus,seconds,clip_id,publish):
        entered.set();await release.wait()
        result=[dict(t=0,boxes={'Ana':[0,0,1,1]},clip_id=clip_id,session_id=s.id)]
        publish(result);return result
    monkeypatch.setattr(s,'track',detect)
    s.start_detection(clip,'source.mp4',None,None,15);await entered.wait()
    s.clips.append(dict(id='new',index=1,status='ready',duration=15,ticks=[],track=[]))
    s.tick(1,0,True,{},clock[0],clip_id='new')
    release.set();await asyncio.gather(*s.detection_tasks)
    assert clip['track']==[] and s.clips[1]['track']==[] and clip['detectionStatus']=='processing'
    await engine.close()


@pytest.mark.asyncio
async def test_failed_provider_cancellation_cannot_hold_generation_owner(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    entered,release=asyncio.Event(),asyncio.Event()
    async def pending(job,images):
        submitted.append(job)
        job.update(requestId='accepted',model='mock',status='generating')
        entered.set();await release.wait();job.update(status='completed')
    async def cancel(model,request):raise ValueError('cancel unavailable')
    monkeypatch.setattr(engine,'run_job',pending)
    monkeypatch.setattr(engine.adapter,'cancel',cancel)
    tick(s,clock,0);tick(s,clock,3.5)
    old=s.task
    await entered.wait()
    # An external ordered scene change invalidates the pending source job.
    s.clips[1]['status']='ready'
    s.tick(1,0,True,{},clock[0]+.01,clip_id=s.clips[1]['id'])
    await asyncio.gather(old,return_exceptions=True)
    await asyncio.sleep(0)
    job=submitted[0]
    assert job['cancelRequested'] and 'could not be confirmed' in job['connectionWarning']
    assert not s.generation_lock.locked() and s.status=='running' and s.playing==1
    release.set();await asyncio.sleep(0)
    assert not s.clips[1].get('url'), 'late completion cannot replace current playback'
    await engine.close()


@pytest.mark.asyncio
async def test_sensor_read_errors_use_unknown_fallback_without_waiting(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    def broken(*args):raise ValueError('sensor unavailable')
    monkeypatch.setattr(s.gaze,'window',broken)
    monkeypatch.setattr(s.eeg,'window',broken)
    monkeypatch.setattr(s.eeg,'status',broken)
    tick(s,clock,0);tick(s,clock,3.5)
    await s.task
    assert s.status=='running' and len(submitted)==1
    assert clip['analysis']['samples']==0 and not clip['analysis']['eeg_available']
    assert submitted[0][0]['engagementDecision']['focus'] is None
    await engine.close()
