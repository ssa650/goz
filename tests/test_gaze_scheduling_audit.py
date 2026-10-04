"""Ground truth geometry, real full-clip decoding, and scheduling validity."""
import asyncio
from copy import deepcopy
import time
import pytest
from backend.adaptive import local_tracker, fusion, gaze_audit, decision_trace
from backend.adaptive.session import AdaptiveSession
from backend.frames import ffmpeg
from test_adaptive_deadline import clock, setup, tick
from test_local_tracking import ROOT, CAST


def test_latest_live_pending_slot_replaces_backlog_and_rejects_old_inflight():
    slot=local_tracker.LatestFrameSlot()
    p=('session','clip','generation',0)
    for i in range(32):slot.offer(object(),i/8,p)
    assert slot.dropped==31
    item=slot.take(31/8);assert item[1]==31/8 and slot.publishable(item,4)
    assert not slot.publishable(item,5)
    slot.offer(object(),4,(*p[:3],1))
    assert not slot.publishable(item,4) # seek/epoch changed while inference ran
    assert slot.take(5) is None
    slot.offer(object(),6,p);assert slot.take(5) is None # future is not evidence
    slot.cancel();assert not slot.publishable(item,4)


def test_playback_priority_skips_stale_work_but_offline_and_pause_keep_coverage():
    state=dict(media_t=4,at=10,playing=True,current=True,rate=2,epoch=1)
    assert local_tracker.playback_target(state,10.1)==pytest.approx(4.2)
    assert not local_tracker.inference_due(3,state,10.1)
    assert local_tracker.inference_due(4.1,state,10.1)
    assert local_tracker.inference_due(0,None,10.1)
    for update in [dict(playing=False),dict(current=False),dict(at=9)]:
        assert local_tracker.inference_due(0,dict(state,**update),10.1)


@pytest.mark.parametrize('scale',[1,1.25,.8,2])
def test_scripted_screen_gaze_ground_truth_and_invalid_flags_never_change_gates(scale):
    rect=dict(x=90+100*scale,y=130+275*scale,w=800*scale,h=450*scale)
    boxes={'Ana':[.1,.1,.4,.9],'Bea':[.6,.1,.9,.9]}
    ticks=[dict(wall=100+i/20,video_t=i/20,playing=True,rect=rect,sessionId='s',clipId='a',epoch=0) for i in range(71)]
    track=[dict(t=0,valid_until=4,boxes=boxes,clip_id='a')]
    samples=[dict(t=k['wall'],x=rect['x']+.75*rect['w'],y=rect['y']+.5*rect['h'],valid=True,blink=False,face=True,confidence=1) for k in ticks]
    tl=fusion.label(samples,ticks,track);a=fusion.analyze(tl,[],track,list(boxes))
    assert a['characters']['Bea']['dwell_s']==3.5 and a['comparison_s']==3.5
    for flag in [dict(valid=False),dict(blink=True),dict(stale=True),dict(face=False),dict(yaw=30)]:
        invalid=fusion.label([dict(s,**flag) for s in samples],ticks,track)
        assert fusion.analyze(invalid,[],track,list(boxes))['characters']['Bea']['dwell_s']==0
    paused=fusion.label(samples,[dict(k,playing=False) for k in ticks],track)
    assert fusion.analyze(paused,[],track,list(boxes))['valid_gaze_s']==0
    outside=fusion.label([dict(s,y=rect['y']-1) for s in samples],ticks,track)
    assert all(e['state']=='outside-video' and e['target'] is None for e in outside)


@pytest.mark.asyncio
async def test_real_14_5s_offline_clip_retains_tail_coverage_and_measured_budget(tmp_path):
    path=tmp_path/'full.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','14.5','-vf','scale=640:400','-r','8','-pix_fmt','yuv420p',path)
    # Match session startup: warm once before any clip tracking is submitted.
    worker=await local_tracker.prewarm_local_tracker('opencv')
    try:
        result=await local_tracker.detect_local(path,CAST,clip_id='a',session_id='s',generation_id='g',warm_worker=worker)
    finally:
        await worker.close()
    assert len(result)==116 and result[0]['t']==0 and result[-1]['t']==14.375
    assert set(result[-1]['boxes'])==set(CAST)
    cost=result[-1]['scheduling']
    print("TRACKING_BUDGET",cost)
    assert cost['mode']=='offline_full_clip' and cost['dropped_stale_inputs']==0
    assert cost['queue_capacity']==2 and cost['worker_cpu_s']>0 and cost['worker_wall_s']>0
    assert all(r['clip_id']=='a' and r['generation_id']=='g' for r in result)
    # The standalone cold helper remains reachable. Its contract is bounded
    # progress, including real partial frames if startup/inference uses budget.
    cold_progress=[];started=time.monotonic()
    try:
        cold=await local_tracker.detect_local(path,CAST,clip_id='cold',session_id='s',generation_id='cold-g',
            on_progress=lambda rows:cold_progress.append(list(rows)))
    except TimeoutError:
        cold=cold_progress[-1] if cold_progress else []
    assert time.monotonic()-started<=local_tracker.MAX_WALL_SECONDS+5
    assert cold and cold[0]['t']==0 and cold[-1]['t']<=14.375
    assert all(r['clip_id']=='cold' and r['generation_id']=='cold-g' and r['t']==r['media_timestamp_s'] for r in cold)
    assert all(a['t']<b['t'] for a,b in zip(cold,cold[1:]))
    gate=local_tracker._gates[asyncio.get_running_loop()]
    deadline=time.monotonic()+10
    while gate.locked() and time.monotonic()<deadline:await asyncio.sleep(.05)
    assert not gate.locked()
    print('COLD_BUDGET',dict(records=len(cold),throughMediaS=cold[-1]['t'],wallS=round(time.monotonic()-started,3)))


@pytest.mark.asyncio
async def test_live_playback_skips_old_inference_and_keeps_unknown_provenance(tmp_path):
    path=tmp_path/'live.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','5','-vf','scale=640:400','-r','8','-pix_fmt','yuv420p',path)
    state=lambda:dict(media_t=4,at=time.monotonic(),playing=True,current=True,epoch=2)
    result=await local_tracker.detect_local(path,CAST,clip_id='a',session_id='s',generation_id='g',playback_state=state)
    gaps=[r for r in result if r.get('input_gap')]
    assert gaps and all(not r['cut'] for r in gaps)  # A decode skip is not a visual scene cut.
    assert all(r['t']>=3.75 for r in result if r['boxes'])
    assert result[-1]['t']==4.875 and result[-1]['scheduling']['dropped_stale_inputs']>0


@pytest.mark.asyncio
async def test_invalid_mapping_audit_and_frozen_trace_are_nonblocking(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    clip['trackingTiming']=dict(firstRecordMs=12.5,throughMediaS=14.375)
    for i in range(71):
        clock[0]=s.started+i/20
        s.gaze.add(dict(t=clock[0],x=80,y=50,valid=True,blink=i%2==0))
        s.tick(0,i/20,True,dict(x=0,y=0,w=100,h=100),clock[0],clip_id='source',
               mapping=dict(valid=False,method='pointer-affine-screen-points',scale=None))
    frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,.5)
    assert submitted and submitted[0][0]['engagementDecision']['focus'] is None
    audit=clip['analysis']['gaze_audit']
    assert audit['received_samples']==71 and audit['flags']['blink_gated']==36
    assert audit['invalid_mapping_ticks']==71 and audit['unaligned_samples']==71
    assert clip['frozenEvidence']==frozen
    trace=s.clips[1]['decisionTrace']
    assert trace['evidence']['gazeAudit']==audit
    assert trace['timing']['trackingAtFreeze']['firstRecordMs']==12.5
    assert trace['timing']['freezeWorkMs']>=0
    await engine.close()


@pytest.mark.asyncio
async def test_yoloe_session_contract_is_lazy_and_never_calls_other_provider(tmp_path,monkeypatch,clock):
    from backend.adaptive import tracks, yoloe_detector
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock);s.tracker='yoloe'
    sentinel=object();monkeypatch.setattr(tracks,'yoloe_config',lambda:sentinel)
    def forbidden(*a,**kw):raise AssertionError('Unexpected local/cloud provider')
    monkeypatch.setattr(local_tracker,'detect_local',forbidden)
    monkeypatch.setattr(tracks,'detect_characters',forbidden)
    async def optional(path,names,**kwargs):
        assert kwargs['config'] is sentinel and kwargs['clip_id']=='source' and kwargs['session_id']==s.id
        records=[dict(t=0,boxes={},source='yoloe_visual_reference_experimental'),
                 dict(t=14,boxes={'Ana':[0,0,.4,1]},source='yoloe_visual_reference_experimental')]
        kwargs['on_progress'](records);return records
    monkeypatch.setattr(yoloe_detector,'detect_yoloe',optional)
    progress=[];result=await s.track('unused',None,None,15,clip['id'],progress.append)
    assert progress[-1]==result and result[-1]['t']==14
    await engine.close()


@pytest.mark.asyncio
async def test_http_optional_unavailable_rejects_before_any_generation(tmp_path,monkeypatch):
    import json
    from backend.adaptive import tracks
    from test_adaptive_http_ordering import local_session
    from test_backend import png
    monkeypatch.setattr(tracks,'yoloe_availability',lambda:dict(available=False,reason='test runtime unavailable'))
    async with local_session(tmp_path,monkeypatch) as (old,engine,adapter,client):
        old.status='stopped'
        response=await client.post('/api/adaptive/sessions',data=dict(premise='Reference story',
            characters=json.dumps([dict(name=n) for n in CAST]),tracker='yoloe',use_saved_sequence='0',duration='15'),
            files={'start':('reference.png',png(),'image/png')})
        assert response.status_code==400 and 'YOLOE unavailable' in response.json()['error']
        assert not adapter.submissions and not adapter.uploads


@pytest.mark.asyncio
async def test_http_invalid_mapping_is_recorded_and_untrusted_rect_is_not_used(tmp_path,monkeypatch):
    from test_adaptive_http_ordering import local_session,tick_body
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        body=tick_body(s);body['mapping']=dict(valid=False,method='pointer-affine-screen-points',reason='window-changed')
        response=await client.post('/api/adaptive/tick',json=body)
        assert response.status_code==200
        latest=s.clips[0]['ticks'][-1]
        assert latest['rect'] is None and latest['mapping']['valid'] is False
        body['mapping']['valid']='true'
        assert (await client.post('/api/adaptive/tick',json=body)).status_code==400
        assert not adapter.submissions


@pytest.mark.asyncio
async def test_yoloe_pending_detection_does_not_delay_freeze_or_end(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock);s.tracker='yoloe'
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered=asyncio.Event()
    async def pending(*args):entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(s,'track',pending)
    s.start_detection(clip,'unused',None,None,15);await entered.wait()
    task=s.detection_by_clip[clip['id']]
    monkeypatch.setattr(s,'start_detection',lambda *a:None)
    tick(s,clock,0);tick(s,clock,3.5)
    frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,.5)
    assert len(submitted)==1 and not task.done()
    started=time.perf_counter();s.ended(0);assert time.perf_counter()-started<.1
    await asyncio.gather(task,return_exceptions=True)
    assert task.cancelled() and clip['frozenEvidence']==frozen
    await engine.close()


@pytest.mark.asyncio
async def test_superseded_generation_cannot_publish_or_change_new_lifecycle(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock)
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered,release=asyncio.Event(),asyncio.Event()
    async def delayed(path,seeds,focus,seconds,clip_id,publish):
        entered.set()
        try:await release.wait()
        except asyncio.CancelledError:pass
        publish([dict(t=0,boxes={'Ana':[0,0,1,1]})])
        return [dict(t=0,boxes={'Ana':[0,0,1,1]})]
    monkeypatch.setattr(s,'track',delayed)
    s.start_detection(clip,'unused',None,None,15);await entered.wait()
    old=s.detection_by_clip[clip['id']]
    clip['trackingGenerationId']='superseding-generation'
    clip['detectionLifecycle']='active'
    old.cancel();await asyncio.gather(old,return_exceptions=True)
    assert not clip['track'] and clip['detectionLifecycle']=='active'
    await engine.close()


def test_failed_yoloe_identity_check_stays_disabled_even_with_runtime_config(monkeypatch):
    from backend.adaptive import tracks
    monkeypatch.setenv('GOZ_YOLOE_LICENSE_REVIEWED','1')
    monkeypatch.setenv('GOZ_YOLOE_WEIGHTS','/tmp/experimental.pt')
    status=tracks.yoloe_availability()
    assert status['available'] is False and 'identity check failed' in status['reason']
    assert status['benchmark']['wrong_identity_labels']==4
