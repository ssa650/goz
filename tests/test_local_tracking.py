"""Real references/decoded video and bounded-lifecycle regressions; no live API."""
import asyncio
from copy import deepcopy
from io import BytesIO
import multiprocessing
from pathlib import Path
import time

import cv2
import httpx
import numpy as np
from PIL import Image
import pytest

from backend.adaptive import local_tracker, tracks
from backend.adaptive.session import AdaptiveSession
from backend.frames import ffmpeg
from test_adaptive_deadline import clock, setup, tick
from test_detection_acceptance import CASES, fixture_image

CAST = ['SpongeBob','Patrick']
ROOT = Path(__file__).resolve().parents[1]


def reference():
    im = np.array(Image.open(ROOT/'presets/secret-box/frames/00-30.jpg').convert('RGB'))
    h,w = im.shape[:2]
    return cv2.resize(im,(640,round(h*640/w)))


@pytest.fixture(autouse=True)
def bounded_cv_threads():
    old = cv2.getNumThreads()
    cv2.setNumThreads(0)
    yield
    cv2.setNumThreads(old)


def test_actual_reference_recognizes_both_with_geometric_evidence():
    boxes,regions = local_tracker.reference_regions(reference(),CAST)
    assert set(boxes)==set(CAST)
    assert boxes['Patrick'][0]<boxes['SpongeBob'][0]
    assert all(r['identity_status']=='verified_reference' and r['verification']['inliers']>=12 for r in regions)
    assert all('confidence' not in r for r in regions)


def test_real_background_and_unreferenced_names_abstain():
    case=next(c for c in CASES if c['id']=='background')
    jpeg,_,_=fixture_image(case)
    pixels=np.array(Image.open(BytesIO(jpeg)).convert('RGB'))
    assert local_tracker.reference_regions(pixels,CAST)[0]=={}
    assert local_tracker.reference_regions(reference(),['Squidward','secret box'])[0]=={}
    assert local_tracker.reference_regions(np.full((360,640,3),127,np.uint8),CAST)[0]=={}


def test_identity_is_recognized_after_staging_swap_not_reassigned_by_position():
    pixels=reference()
    swapped=np.concatenate((pixels[:,320:],pixels[:,:320]),axis=1)
    boxes,_=local_tracker.reference_regions(swapped,CAST)
    assert set(boxes)==set(CAST)
    assert boxes['SpongeBob'][0]<boxes['Patrick'][0]


def test_short_flow_preserves_match_but_cut_and_blank_reset():
    tracker=local_tracker.ReferenceTracker(CAST)
    pixels=reference()
    first=tracker.step(pixels,0)
    flow=tracker.step(pixels,.125)
    assert set(first['boxes'])==set(flow['boxes'])==set(CAST)
    assert all(r['identity_status']=='tracked_reference' for r in flow['regions'])
    blank=tracker.step(np.zeros_like(pixels),.25)
    assert blank['cut'] and blank['boxes']=={} and blank['shot_id']==1
    assert not tracker.active
    assert tracker.step(pixels,.375)['boxes']


def test_failed_rerecognition_and_expired_identity_abstain(monkeypatch):
    pixels=reference()
    tracker=local_tracker.ReferenceTracker(CAST)
    assert tracker.step(pixels,0)['boxes']
    monkeypatch.setattr(local_tracker,'reference_regions',lambda *a:({},[]))
    assert tracker.step(pixels,.5)['boxes']=={}
    tracker.active={'Patrick':dict(box=[.1,.1,.3,.8],recognized_at=0)}
    tracker.next_recognition=10
    assert tracker.step(pixels,1)['boxes']=={}


def test_cloud_merge_never_carries_identity_across_local_cut_or_conflict():
    local=[dict(t=t,boxes={},cut=t==.5,valid_until=t+.125,status='unavailable') for t in [0,.25,.5,.75,1]]
    remote=[dict(t=0,boxes={'Patrick':[.1,.1,.4,.9]},valid_until=.8,source='openai_reference_vision')]
    merged=tracks.merge_tracking(local,remote)
    assert tracks.boxes_at(merged,.25)==remote[0]['boxes']
    assert tracks.boxes_at(merged,.5)=={} and tracks.boxes_at(merged,.75)=={}
    local[0]['boxes']={'SpongeBob':[.1,.1,.4,.9]}
    assert tracks.merge_tracking(local,remote)[0]['boxes']=={}


@pytest.mark.asyncio
async def test_decoded_full_clip_has_t0_tail_tags_thread_limits_and_progress(tmp_path):
    still=ROOT/'presets/secret-box/frames/00-30.jpg'
    video=tmp_path/'reference.mp4'
    await ffmpeg('-loop','1','-i',still,'-t','4.5','-vf','scale=640:400','-r','8','-pix_fmt','yuv420p',video)
    progress=[]
    result=await local_tracker.detect_local(video,CAST,clip_id='clip',session_id='session',generation_id='generation',
        on_progress=lambda records:progress.append(deepcopy(records)))
    assert result[0]['t']==0 and result[-1]['t']==4.375
    assert len(result)==36 and progress
    assert all(f['clip_id']=='clip' and f['session_id']=='session' and f['generation_id']=='generation' for f in result)
    assert all(f['resources']['opencv_threads']==1 and f['resources']['decode_threads']==1 for f in result)
    assert all(f['valid_until']-f['t']<=.125+1e-6 for f in result)
    assert set(tracks.boxes_at(result,4.3))==set(CAST)
    assert tracks.boxes_at(result,4.6)=={}


@pytest.mark.asyncio
async def test_local_worker_cancel_cooperatively_exits_and_releases_gate(tmp_path):
    video=tmp_path/'long.mp4'
    # Enough queued frames to cancel after first progress; no long encode wait.
    await ffmpeg('-framerate','8','-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','4.5',
                 '-vf','scale=640:400','-r','8','-pix_fmt','yuv420p',video)
    baseline={p.pid for p in multiprocessing.active_children()}
    observed=asyncio.Event()
    task=asyncio.create_task(local_tracker.detect_local(video,CAST,clip_id='clip',session_id='session',
        on_progress=lambda result:observed.set()))
    await asyncio.wait_for(observed.wait(),local_tracker.MAX_WALL_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task,2)
    gate=local_tracker._gates[asyncio.get_running_loop()]
    if {p.pid for p in multiprocessing.active_children()} != baseline:
        # Bounded cancellation can return while native cleanup still owns slot.
        assert gate.locked()
        assert not local_tracker._worker_slot.acquire(blocking=False)
        await asyncio.wait_for(gate.acquire(),10)
        gate.release()
    assert {p.pid for p in multiprocessing.active_children()}==baseline
    assert not gate.locked()


@pytest.mark.asyncio
async def test_worker_timeout_is_bounded_and_keeps_partial_progress(tmp_path,monkeypatch):
    video=tmp_path/'source.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','5',
                 '-vf','scale=640:400','-r','8','-pix_fmt','yuv420p',video)
    baseline={p.pid for p in multiprocessing.active_children()}
    monkeypatch.setattr(local_tracker,'MAX_WALL_SECONDS',.01)
    with pytest.raises(TimeoutError):
        await local_tracker.detect_local(video,CAST,clip_id='clip',session_id='session')
    # Cooperative startup cancellation may outlive the caller by a bounded startup
    # interval. The slot stays occupied until the child actually exits.
    deadline=time.monotonic()+5
    while {p.pid for p in multiprocessing.active_children()} != baseline and time.monotonic()<deadline:
        await asyncio.sleep(.05)
    assert {p.pid for p in multiprocessing.active_children()}==baseline


@pytest.mark.parametrize('provider',['fal','opencv','people'])
def test_provider_validated_and_public_per_session(tmp_path,monkeypatch,clock,provider):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    s.tracker=tracks.tracking_provider(provider)
    assert s.public()['tracker']==provider
    with pytest.raises(ValueError,match='Tracking provider'):
        tracks.tracking_provider('cloud-training')


@pytest.mark.asyncio
async def test_local_provider_never_calls_cloud_and_streams_tail(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    s.tracker='opencv'
    def forbidden(*a,**kw):raise AssertionError('Local mode called cloud')
    monkeypatch.setattr(tracks,'detect_characters',forbidden)
    async def local(*args,**kwargs):
        records=[dict(t=0,boxes={}),dict(t=14,boxes={'Ana':[0,0,1,1]})]
        kwargs['on_progress'](records)
        return records
    monkeypatch.setattr(local_tracker,'detect_local',local)
    updates=[]
    result=await s.track('unused',None,None,15,clip['id'],lambda result:updates.append(result))
    assert updates[-1][-1]['t']==14 and result==updates[-1]
    await engine.close()


@pytest.mark.asyncio
async def test_florence_retained_with_full_clip_local_progress_and_late_cloud(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    s.tracker='fal'
    entered,release=asyncio.Event(),asyncio.Event()
    async def cloud(*args,**kwargs):
        assert kwargs['observation_seconds']==3.5
        entered.set();await release.wait()
        return [dict(t=0,boxes={'Bea':[.6,.1,.9,.9]},valid_until=.8)]
    async def local(*args,**kwargs):
        result=[dict(t=0,boxes={'Ana':[.1,.1,.4,.9]},valid_until=.125),dict(t=14,boxes={},valid_until=14.125)]
        kwargs['on_progress'](result)
        return result
    monkeypatch.setattr(tracks,'detect_characters',cloud)
    monkeypatch.setattr(local_tracker,'detect_local',local)
    updates=[]
    task=asyncio.create_task(s.track('unused',None,None,15,clip['id'],updates.append))
    await entered.wait()
    assert not task.done() and updates[-1][-1]['t']==14
    release.set();result=await task
    assert set(result[0]['boxes'])=={'Ana','Bea'}
    await engine.close()


@pytest.mark.asyncio
async def test_pending_full_tracker_does_not_gate_freeze_generation_or_clip_end(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered=asyncio.Event()
    async def pending(*args):
        entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(s,'track',pending)
    s.start_detection(clip,'unused',None,None,15)
    await entered.wait()
    task=s.detection_by_clip[clip['id']]
    monkeypatch.setattr(s,'start_detection',lambda *a:None)
    tick(s,clock,0);tick(s,clock,3.5)
    frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,.5)
    assert len(submitted)==1 and s.clips[1]['status']=='ready' and not task.done()
    start=time.perf_counter();s.ended(0)
    assert time.perf_counter()-start<.1
    await asyncio.gather(task,return_exceptions=True)
    assert task.cancelled() and clip['frozenEvidence']==frozen
    assert clip['detectionLifecycle']=='cancelled'
    await engine.close()


@pytest.mark.asyncio
async def test_wrong_generation_and_post_end_publish_cannot_mutate_clip(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered,release=asyncio.Event(),asyncio.Event()
    async def detect(path,seeds,focus,seconds,clip_id,publish):
        publish([dict(t=0,boxes={'Ana':[0,0,1,1]},generation_id='old')])
        entered.set()
        try:await release.wait()
        except asyncio.CancelledError:pass  # Provider can race cancellation.
        result=[dict(t=0,boxes={'Bea':[0,0,1,1]})]
        publish(result);return result
    monkeypatch.setattr(s,'track',detect)
    s.start_detection(clip,'unused',None,None,15);await entered.wait()
    assert clip['track']==[]
    clip['endedAt']=clock[0]
    s.cancel_detection(clip)
    await asyncio.gather(*s.detection_tasks,return_exceptions=True)
    assert clip['track']==[]
    await engine.close()


def test_florence_diagnostics_preserve_status_and_redact_secrets():
    key='secret-key-123'
    request=httpx.Request('POST','https://fal.run/private?key='+key,headers={'Authorization':'Key '+key})
    response=httpx.Response(429,json={'detail':key},request=request)
    error=httpx.HTTPStatusError('bad '+key,request=request,response=response)
    details=tracks.failure_details(error,key)
    assert details['http_status']==429 and details['error_category']=='rate_limit'
    assert key not in str(details) and 'https://' not in str(details)
    assert tracks.failure_details(httpx.ReadTimeout('network '+key),key)['error_category']=='timeout'
    assert key not in str(tracks.failure_details(ValueError('Rejected '+key),key))


@pytest.mark.asyncio
async def test_cancel_florence_also_cancels_shared_inflight_requests(monkeypatch):
    monkeypatch.setattr(tracks,'sample_scene_frames',lambda *a:[dict(t=0,jpeg=b'a',width=2,height=2,shot=0,cut=False)])
    entered,cancelled=asyncio.Event(),asyncio.Event()
    async def delayed(request):
        entered.set()
        try:await asyncio.Event().wait()
        finally:cancelled.set()
    task=asyncio.create_task(tracks.detect_characters('unused',[dict(name='Ana')],'test',clip_id='clip',
        transport=httpx.MockTransport(delayed)))
    await entered.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,.5)
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider',['fal','opencv','invalid'])
async def test_http_session_provider_is_accepted_or_rejected_without_generation(tmp_path,monkeypatch,provider):
    import json
    from test_adaptive_http_ordering import local_session
    from test_backend import png
    async with local_session(tmp_path,monkeypatch) as (old,engine,adapter,client):
        old.status='stopped'
        async def no_generation(self):pass
        monkeypatch.setattr(AdaptiveSession,'start',no_generation)
        response=await client.post('/api/adaptive/sessions',data=dict(premise='Reference story',
            characters=json.dumps([dict(name=n) for n in CAST]),tracker=provider,use_saved_sequence='0',duration='15'),
            files={'start':('reference.png',png(),'image/png')})
        if provider=='invalid':
            assert response.status_code==400 and 'Tracking provider' in response.json()['error']
        else:
            assert response.status_code==202 and response.json()['tracker']==provider
        assert adapter.submissions==[] and adapter.uploads==[]


@pytest.mark.asyncio
async def test_session_supplies_frozen_quality_window_to_policy_without_waiting(tmp_path,monkeypatch,clock):
    from backend.adaptive import fusion
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    original=fusion.analyze
    calls=[]
    def analyze(*args,**kwargs):
        calls.append(deepcopy(kwargs))
        return original(*args,**kwargs)
    monkeypatch.setattr(fusion,'analyze',analyze)
    tick(s,clock,0);tick(s,clock,3.5)
    frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,.5)
    assert calls[0]['eeg_quality']==frozen['quality']
    assert calls[0]['eeg_window']==(frozen['start'],frozen['end'])
    assert len(submitted)==1 and s.clips[1]['status']=='ready'
    assert clip['frozenEvidence']==frozen
    assert clip['analysis']['eeg_policy']['action']=='keep'
    await engine.close()
