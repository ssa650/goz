"""Offline warm-worker current-frame input, bounds and validated-copy handoff."""
import asyncio
import base64
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
import pytest

from backend.adaptive import local_tracker, tracks
from backend.adaptive.session import AdaptiveSession
from backend.adaptive.sensors import GazeFeed, EegFeed
from backend.engine import Engine
from test_backend import FakeAdapter
from test_prewarm_local_tracker import video
from test_adaptive_http_ordering import local_session

CAST=['SpongeBob','Patrick']
ROOT=Path(__file__).resolve().parents[1]


def jpeg():
    rgb=np.array(Image.open(ROOT/'presets/secret-box/frames/00-30.jpg').convert('RGB'))
    bgr=cv2.cvtColor(cv2.resize(rgb,(640,400)),cv2.COLOR_RGB2BGR)
    ok,data=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,80]);assert ok
    assert len(data)<=local_tracker.MAX_FRAME_BYTES
    return bytes(data)


@pytest.mark.asyncio
async def test_warm_presented_frame_precedes_local_copy_and_cancel_handoff_keeps_full_attempt(tmp_path):
    worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10)
    tags=dict(clip_id='clip',session_id='session',generation_id='generation')
    state=dict(media_t=1,at=time.monotonic(),playing=True,current=True,epoch=7,rate=1)
    pending=[dict(jpeg=jpeg(),t=1,epoch=7,tags=tags)]
    rows=[];seen=asyncio.Event()
    def publish(result):
        rows[:]=result
        if result:seen.set()
    task=asyncio.create_task(local_tracker.detect_local(None,CAST,**tags,provider='color',warm_worker=worker,
        frame_source=lambda:pending.pop() if pending else None,playback_state=lambda:state,on_progress=publish))
    try:
        await asyncio.wait_for(seen.wait(),5)
        assert set(rows[0]['boxes'])==set(CAST)
        assert rows[0]['t']==1 and rows[0]['playback_epoch']==7 and rows[0]['valid_until']==1.25
        assert rows[0]['decode']['decoder']=='browser_presented_jpeg'
        assert not (tmp_path/'full.mp4').exists(),'No local media dependency'
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,2)
        video(tmp_path/'full.mp4',frames=16)
        full=await local_tracker.detect_local(tmp_path/'full.mp4',CAST,**tags,provider='color',warm_worker=worker)
        assert [f['t'] for f in full]==[0,.25,.5,.75,1,1.25,1.5,1.75]
        assert all(f['clip_id']=='clip' for f in full) and worker.active_token is None
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        await worker.close()


@pytest.mark.asyncio
async def test_presented_input_rejects_malformed_oversized_stale_seek_and_ended(tmp_path):
    engine=Engine(FakeAdapter(),tmp_path)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'Scene',[dict(name=n) for n in CAST],15,'480P',None)
    clip=dict(id=str(uuid4()),index=0,status='ready',duration=15,ticks=[],track=[],analysisStarted=True,
        trackingGenerationId='g',trackingRunId='r')
    s.clips.append(clip);s.presented_frames[clip['id']]=local_tracker.LatestFrameSlot()
    wall=time.time();s.tick(0,1,True,{},wall,clip_id=clip['id'],epoch=1)
    body=dict(clip=0,clip_id=clip['id'],epoch=1,video_t=1,wall=wall*1000,tracking_frame=base64.b64encode(jpeg()).decode())
    assert not s.offer_tracking_frame({**body,'tracking_frame':'!invalid'})
    assert not s.offer_tracking_frame({**body,'tracking_frame':'x'*80001})
    assert not s.offer_tracking_frame({**body,'tracking_frame':base64.b64encode(b'not JPEG').decode()})
    assert not s.offer_tracking_frame({**body,'clip_id':'old'})
    assert not s.offer_tracking_frame({**body,'epoch':0})
    assert s.offer_tracking_frame(body)
    assert not s.offer_tracking_frame(body),'Rate bound applies before decoding'
    next_wall=time.time()+.001;s.tick(0,1.125,True,{},next_wall,clip_id=clip['id'],epoch=1)
    assert s.offer_tracking_frame({**body,'video_t':1.125,'wall':next_wall*1000})
    slot=s.presented_frames[clip['id']];assert slot.dropped==1
    assert slot.take(1.125)[0]['t']==1.125,'Only latest pending pixels survive'
    clip['track']=[dict(t=1,boxes={},playback_epoch=1)]
    s.tick(0,.1,True,{},next_wall+.001,clip_id=clip['id'],epoch=2)
    assert clip['track']==[] and not s.offer_tracking_frame(body)
    clip['status']='watched';assert not s.offer_tracking_frame(body)
    await engine.close()


@pytest.mark.asyncio
async def test_late_live_publication_and_cancel_cannot_replace_local_handoff(tmp_path,monkeypatch):
    engine=Engine(FakeAdapter(),tmp_path)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'Scene',[dict(name=n) for n in CAST],15,'480P',None)
    clip=dict(id=str(uuid4()),index=0,status='ready',duration=15,ticks=[],track=[],jobId='g')
    s.clips.append(clip)
    entered=asyncio.Event();local=asyncio.Event();release=asyncio.Event()
    async def track(path,seeds,focus,seconds,clip_id,publish):
        if path is None:
            entered.set()
            try:await asyncio.Event().wait()
            except asyncio.CancelledError:
                publish([dict(t=0,boxes={'late':[0,0,1,1]})])
                raise
        local.set();await release.wait()
        return [dict(t=1,boxes={'SpongeBob':[.1,.1,.5,.8]})]
    monkeypatch.setattr(s,'track',track)
    s.start_detection(clip,None,None,None,15);await entered.wait()
    s.start_detection(clip,Path('validated.mp4'),None,None,15);await local.wait();await asyncio.sleep(0)
    assert clip['detectionLifecycle']=='active' and clip['track']==[]
    release.set();await s.detection_by_clip[clip['id']]
    assert clip['detectionLifecycle']=='complete' and list(clip['track'][0]['boxes'])==['SpongeBob']
    await engine.close()


@pytest.mark.asyncio
async def test_stream_route_uses_validated_copy_for_later_ranges_without_provider(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        path=tmp_path/'full.mp4';video(path,frames=8)
        job=engine.new_job(dict(mode='text',prompt='saved',duration=15,resolution='480P'))
        job.update(status='completed',video=dict(url='https://fal.media/completed.mp4'))
        clip=s.clips[1];clip.update(jobId=job['id'],playbackDelivery='stream',streamMetadata=dict(bytes=path.stat().st_size),
            localMediaStatus='validated',path=str(path))
        async def forbidden(*args,**kwargs):raise AssertionError('No provider connection after validation')
        monkeypatch.setattr('backend.progressive_media.open_video',forbidden)
        result=await client.get(f'/api/adaptive/clips/{s.id}/1/stream',headers={'Range':'bytes=0-31'})
        assert result.status_code==206 and result.content==path.read_bytes()[:32]

@pytest.mark.asyncio
async def test_stream_readiness_starts_live_input_before_slow_validation_and_hands_off(tmp_path,monkeypatch):
    from backend import progressive_media
    from test_adaptive import png
    from backend.frames import verify_image
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'Scene',[dict(name=n) for n in CAST],15,'480P',verify_image(png()),tracker='color',playback_mode='stream')
    s.max_scenes=1;s.tracking_readiness['state']='ready'
    release=asyncio.Event();entered=asyncio.Event();paths=[]
    async def media(job):await release.wait();return Path('validated.mp4')
    async def probe(url):return dict(bytes=1000,metadata='early_moov',rangeSupported=True)
    async def track(path,*args):
        paths.append(path);entered.set()
        if path is None:await asyncio.Event().wait()
        return []
    monkeypatch.setattr(engine,'media_path',media);monkeypatch.setattr(progressive_media,'probe_video',probe);monkeypatch.setattr(s,'track',track)
    await s.make_scene({},s.opening,None,[]);await entered.wait()
    clip=s.clips[0]
    assert clip['localMediaStatus']=='downloading' and clip['detectionInput']=='presented_video_frame'
    assert clip['status']=='ready' and paths==[None]
    release.set();await s.local_media_tasks[clip['id']];await s.detection_by_clip[clip['id']]
    assert paths==[None,Path('validated.mp4')] and clip['detectionLifecycle']=='complete'
    assert clip['localMediaStatus']=='validated' and clip['detectionInput']=='validated_local_clip'
    await engine.close()

@pytest.mark.asyncio
async def test_freeze_copies_bounded_gaze_receipt_diagnostics_even_with_no_samples(tmp_path):
    engine=Engine(FakeAdapter(),tmp_path)
    gaze=GazeFeed();gaze.diagnostics.update(packets=7,rejected=7,eventLoopDelayMsMax=8000)
    s=AdaptiveSession(engine,gaze,EegFeed(),tmp_path,'Scene',[dict(name=n) for n in CAST],15,'480P',None)
    wall=time.time();clip=dict(id=str(uuid4()),index=0,status='playing',duration=15,ticks=[],track=[],playStartedAt=wall-3.5)
    s.clips.append(clip);s.freeze_evidence(clip,wall)
    saved=clip['frozenEvidence']['gazeInputDiagnostics']
    assert clip['frozenEvidence']['gaze']==[] and saved['packets']==7 and saved['eventLoopDelayMsMax']==8000
    gaze.diagnostics['packets']=8
    assert saved['packets']==7,'Freeze owns a copy rather than mutable feed counters'
    await engine.close()


@pytest.mark.asyncio
async def test_real_tick_route_warm_worker_slow_local_handoff_and_ordering(tmp_path,monkeypatch):
    from backend.adaptive import tracking_diagnostics
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        s.tracker='color';s.max_scenes=1
        s.local_worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10)
        s.tracking_readiness['state']='ready'
        clip=s.clips[0];clip.update(jobId='offline',localMediaStatus='downloading')
        s.start_detection(clip,None,None,None,15)
        browser_run=clip['trackingRunId']
        path=tmp_path/'validated.mp4';video(path,frames=16)
        release=asyncio.Event()
        async def media(job):await release.wait();return path
        monkeypatch.setattr(engine,'media_path',media)
        local=engine.spawn(s.prepare_local_media(clip,dict(id='offline'),{},15,time.perf_counter()))
        started=time.monotonic();encoded=base64.b64encode(jpeg()).decode()
        def body(t,epoch=1):
            return dict(session_id=s.id,clip=0,clip_id=clip['id'],epoch=epoch,video_t=t,playing=True,
                wall=time.time()*1000,tracking_run_id=browser_run,tracking_frame=encoded,
                tracking_capture=dict(reason='encoded',counters=dict(encoded=1)))
        for _ in range(30):
            response=await client.post('/api/adaptive/tick',json=body(.5+time.monotonic()-started))
            assert response.status_code==200,response.text
            if clip['track']:break
            await asyncio.sleep(.04)
        assert response.json()['tickAccepted'] and clip['track']
        assert clip['localMediaStatus']=='downloading' and 'path' not in clip
        assert clip['track'][0]['decode']['decoder']=='browser_presented_jpeg'
        assert clip['track'][0]['valid_until']==clip['track'][0]['t']+.25
        assert set(clip['track'][0]['boxes'])==set(CAST)
        assert not adapter.submissions
        # New wall time must not revive an old seek epoch or rewind its media.
        newer=body(.1,epoch=2)
        assert (await client.post('/api/adaptive/tick',json=newer)).json()['trackingFrameAccepted']
        assert not (await client.post('/api/adaptive/tick',json=body(.9))).json()['tickAccepted']
        assert not (await client.post('/api/adaptive/tick',json=body(.05,epoch=2))).json()['tickAccepted']
        release.set();await local;await s.detection_by_clip[clip['id']]
        assert clip['detectionInput']=='validated_local_clip' and clip['trackingRunId']!=browser_run
        assert clip['detectionLifecycle']=='complete' and max(f['t'] for f in clip['track'])==1.75
        late=body(.25,epoch=2)
        assert not (await client.post('/api/adaptive/tick',json=late)).json()['trackingFrameAccepted']
        journal=tracking_diagnostics.for_clip(s,clip);await journal.flush()
        rows=tracking_diagnostics.read(tmp_path,s.id)
        assert any(r['event']=='tracking_cancel_requested' and r['reason']=='local_handoff' for r in rows)
        assert clip['presentedFrameDiagnostics']['counters']['route:accepted']>0
        assert clip['presentedFrameDiagnostics']['counters']['route:local_handoff']==1


@pytest.mark.asyncio
async def test_tick_http_payload_limits_reasons_mapping_and_inactive_ticks(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        clip=s.clips[0];clip.update(trackingGenerationId='g',trackingRunId='r')
        s.presented_frames[clip['id']]=local_tracker.LatestFrameSlot()
        def body(t=.5,**fields):
            return dict(session_id=s.id,clip=0,clip_id=clip['id'],epoch=1,video_t=t,playing=True,
                wall=time.time()*1000,tracking_run_id='r',**fields)
        for i,(frame,reason) in enumerate([('!invalid','invalid_base64'),('x'*80001,'payload_limit'),
                (base64.b64encode(b'not JPEG').decode(),'invalid_jpeg')]):
            result=await client.post('/api/adaptive/tick',json=body(.5+i*.125,tracking_frame=frame))
            assert result.status_code==200,result.text
            assert not result.json()['trackingFrameAccepted']
            assert clip['presentedFrameDiagnostics']['lastReason']==reason
        ticks=len(clip['ticks'])
        result=await client.post('/api/adaptive/tick',json=body(1,tracking_frame='x'*120000))
        assert result.status_code==413 and len(clip['ticks'])==ticks
        encoded=base64.b64encode(jpeg()).decode()
        result=await client.post('/api/adaptive/tick',json=body(1,tracking_frame=encoded,tracking_capture=dict(reason='cors_tainted',
            counters=dict(cors_tainted=1,unknown=9)),mapping=dict(valid=False,reason='test',method='test',
            coordinateSpace='screen-points',viewportWidth=1280,viewportHeight=720,windowX=-10,windowY=20,
            fullscreen=True,gazeOverlayEnabled=False,boxesOverlayEnabled=True,recoveryActive=False)))
        assert result.json()['trackingFrameAccepted']
        assert clip['ticks'][-1]['mapping']['viewportWidth']==1280
        assert clip['ticks'][-1]['mapping']['windowX']==-10
        assert clip['presentedFrameDiagnostics']['capture']['counters']==dict(cors_tainted=1)
        paused=body(1.125,tracking_frame=encoded);paused['playing']=False
        assert not (await client.post('/api/adaptive/tick',json=paused)).json()['trackingFrameAccepted']
        stale=body(1.25,tracking_frame=encoded);stale['wall']-=400
        # Out-of-order first, while a fresh tick with old capture time is also
        # rejected independently by the session's freshness guard.
        assert not (await client.post('/api/adaptive/tick',json=stale)).json()['trackingFrameAccepted']
        clip['ticks']=[];stale['video_t']=1.5
        assert not (await client.post('/api/adaptive/tick',json=stale)).json()['trackingFrameAccepted']
        assert clip['presentedFrameDiagnostics']['lastReason']=='stale_capture'
        wrong=body(1.75,tracking_frame=encoded);wrong['tracking_run_id']='old'
        assert not (await client.post('/api/adaptive/tick',json=wrong)).json()['trackingFrameAccepted']
        assert clip['presentedFrameDiagnostics']['lastReason']=='tracking_run'


@pytest.mark.asyncio
async def test_busy_worker_keeps_only_latest_frame_and_cooperatively_cancels(tmp_path,monkeypatch):
    from backend.adaptive import tracking_diagnostics
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        s.tracker='color';s.tracking_readiness['state']='ready'
        s.local_worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10)
        assert local_tracker._worker_slot.acquire(blocking=False)
        clip=s.clips[0];clip['jobId']='offline';s.start_detection(clip,None,None,None,15)
        try:
            for t in (.5,.625,.75):
                wall=time.time();s.tick(0,t,True,None,wall,clip_id=clip['id'],epoch=1)
                assert s.offer_tracking_frame(dict(clip=0,clip_id=clip['id'],epoch=1,video_t=t,wall=wall*1000,
                    tracking_frame=base64.b64encode(jpeg()).decode()))
                await asyncio.sleep(.01)
            slot=s.presented_frames[clip['id']]
            assert slot.dropped==2 and slot.pending[1]==.75 and clip['track']==[]
            s.cancel_detection(clip,'session_stop')
            await asyncio.gather(s.detection_by_clip[clip['id']],return_exceptions=True)
            assert slot.pending is None and slot.provenance is None
        finally:local_tracker._worker_slot.release()
        assert s.local_worker.active_token is None


def test_worker_rejects_decode_bombs_wrong_identity_and_stale_frames_without_inference():
    import multiprocessing,queue,threading
    from io import BytesIO
    tags=dict(session_id='session',clip_id='clip',generation_id='g')
    inputs=queue.Queue();stop=threading.Event();rows=[]
    malformed=b'\xff\xd8broken\xff\xd9'
    buffer=BytesIO();Image.new('RGB',(641,2)).save(buffer,format='JPEG')
    cases=[dict(jpeg=jpeg(),t=1,epoch=2,tags=dict(tags,clip_id='old')),
        dict(jpeg=jpeg(),t=0,epoch=2,tags=tags),dict(jpeg=jpeg(),t=1,epoch=1,tags=tags),
        dict(jpeg=malformed,t=1,epoch=2,tags=tags),dict(jpeg=buffer.getvalue(),t=1,epoch=2,tags=tags),
        dict(jpeg=b'x'*60001,t=1,epoch=2,tags=tags)]
    for item in cases:inputs.put(item)
    playback=multiprocessing.Array('d',[1,time.monotonic(),1,1,2,1])
    class Output:
        def put(self,record,**kwargs):
            rows.append(record)
            if record.get('reason')=='payload_limit':stop.set()
    local_tracker._frames_worker(CAST,tags,Output(),stop,playback,inputs,'color')
    assert {r.get('reason') for r in rows} >= {'provenance_mismatch','inactive_or_stale','invalid_jpeg','dimensions_or_format','payload_limit'}
    assert not any('boxes' in r for r in rows)
