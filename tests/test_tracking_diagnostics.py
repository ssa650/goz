"""Real decoder/HTTP provenance and durable diagnostics; no live providers."""
import asyncio
import json
import time

import cv2
import numpy as np
import pytest

from backend.adaptive import local_tracker, tracking_diagnostics
from backend.adaptive.color_detector import ColorCharacterDetector
from backend.frames import ffmpeg
from test_adaptive_http_ordering import local_session
from test_color_detector import background, sponge
from test_local_tracking import ROOT, CAST


def test_small_disjoint_patch_cannot_silence_supported_large_torso():
    image=sponge(background())
    cv2.rectangle(image,(300,50),(309,62),(240,210,40),-1)
    cv2.rectangle(image,(302,53),(306,56),(240,240,240),-1)
    result=ColorCharacterDetector(['SpongeBob']).step(image,0)
    assert set(result['boxes'])=={'SpongeBob'}
    assert result['boxes']['SpongeBob'][0]<.4
    assert any(r['identity_status']=='suppressed_duplicate' for r in result['regions'])


@pytest.mark.asyncio
async def test_fast_decoder_preserves_real_timestamps_and_persists_pipeline_stages(tmp_path):
    video=tmp_path/'source.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','2',
                 '-vf','scale=640:400','-r','24','-pix_fmt','yuv420p',video)
    result=await local_tracker.detect_local(video,CAST,clip_id='clip',session_id='session',
        generation_id='generation',provider='color',diagnostics_directory=tmp_path)
    assert [r['t'] for r in result]==[0,.25,.5,.75,1,1.25,1.5,1.75]
    assert all(r['decode']['actual_media_t']==r['t'] and r['resources']['decode_threads']==1 for r in result)
    events=tracking_diagnostics.read(tmp_path,'session')
    kinds={r['event'] for r in events}
    assert {'job_requested','worker_slot_acquired','worker_spawned','worker_entered',
            'decoder_open','tracker_loaded','frame_received','worker_complete','job_retiring'}<=kinds
    loaded=next(r for r in events if r['event']=='tracker_loaded')
    assert loaded['provider']=='color' and loaded['opencv_version']==cv2.__version__
    assert set(loaded['functionCodeSha256'])=={'worker','trackerStep','candidateFeatures'}
    assert all(len(value)==64 for value in loaded['functionCodeSha256'].values())
    frames=[r for r in events if r['event']=='frame_received']
    assert len(frames)==8 and all(r['transport']['status']=='accepted' for r in frames)
    assert all(r['sessionId']=='session' and r['generationId']=='generation' for r in events)
    assert all('candidates' in r and 'boxes' in r and 'validUntil' in r for r in frames)
    print('DECODER_BUDGET',next(r for r in events if r['event']=='decoder_open'))


@pytest.mark.asyncio
async def test_live_cursor_seeks_over_old_input_without_inventing_scene_cuts(tmp_path):
    video=tmp_path/'source.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','4',
                 '-vf','scale=640:400','-r','24','-pix_fmt','yuv420p',video)
    def cursor():
        return dict(media_t=2.5,at=time.monotonic(),playing=True,current=True,epoch=1,rate=1)
    result=await local_tracker.detect_local(video,CAST,clip_id='c',session_id='s',
        provider='color',playback_state=cursor)
    assert result[0]['t']>=2.5
    assert result[0]['input_gap'] and not result[0]['cut']
    assert result[0]['scheduling']['dropped_stale_inputs']>=10
    assert all(r['abstention_reason']!='stale_work_skipped_for_playback' for r in result)


@pytest.mark.asyncio
async def test_overlay_route_persists_only_current_provenance_and_bounded_coordinates(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (session,engine,_,client):
        clip=session.clips[0]
        body=dict(session_id=session.id,clip_id=clip['id'],wall=time.time()*1000,
            diagnostic=dict(enabled=True,status='expired_frame',mediaTime=12,frameMediaTime=.75,
                validUntil=1,devicePixelRatio=2,contentRect=dict(x=0,y=36,w=1710,h=1069),
                canvasRect=dict(x=0,y=0,w=1710,h=1107),videoRect=dict(x=0,y=0,w=1710,h=1107),
                boxes=[],raw_camera='must not persist',private_url='https://must-not-persist'))
        bad=dict(body,session_id='older')
        assert (await client.post('/api/adaptive/overlay-diagnostic',json=bad)).status_code==400
        assert not list((tmp_path/'adaptive'/session.id).glob('*tracking.jsonl'))
        response=await client.post('/api/adaptive/overlay-diagnostic',json=body)
        assert response.status_code==200,response.text
        response=await client.post('/api/adaptive/overlay-diagnostic',json=body)
        assert response.json()['rateLimited'] is True
        await tracking_diagnostics.for_clip(session,clip).flush()
        response=await client.get('/api/adaptive/tracking-diagnostics',params={'session_id':session.id})
        assert response.status_code==200,response.text
        records=response.json()['records']
        assert len(records)==1 and records[0]['status']=='expired_frame'
        assert records[0]['clipId']==clip['id'] and records[0]['reportedBy']=='browser'
        assert 'raw_camera' not in json.dumps(records) and 'private_url' not in json.dumps(records)


@pytest.mark.asyncio
async def test_tracking_diagnostic_reader_blocks_path_traversal(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (_,_,_,client):
        response=await client.get('/api/adaptive/tracking-diagnostics',params={'session_id':'../../other'})
        assert response.status_code==400


def test_overlay_numbers_and_names_fail_closed():
    with pytest.raises(ValueError):
        tracking_diagnostics.overlay_evidence(dict(mediaTime=float('nan')),CAST)
    with pytest.raises(ValueError):
        tracking_diagnostics.overlay_evidence(dict(boxes=[dict(name='invented',normalized=[0,0,1,1],canvas=[0,0,1,1])]),CAST)


@pytest.mark.asyncio
async def test_worker_cancel_after_actual_frame_releases_both_slots_without_killing(tmp_path):
    import multiprocessing
    from PIL import Image
    video=tmp_path/'local.mp4'
    rgb=np.array(Image.open(ROOT/'presets/secret-box/frames/00-30.jpg').convert('RGB'))
    frame=cv2.cvtColor(cv2.resize(rgb,(640,400)),cv2.COLOR_RGB2BGR)
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),24,(640,400))
    assert writer.isOpened()
    try:
        for _ in range(24*30):
            writer.write(frame)
    finally:
        writer.release()
    baseline={p.pid for p in multiprocessing.active_children()}
    observed=asyncio.Event()
    task=asyncio.create_task(local_tracker.detect_local(video,CAST,clip_id='clip',session_id='session',
        diagnostics_directory=tmp_path,on_progress=lambda frames:observed.set() if frames else None))
    await asyncio.wait_for(observed.wait(),local_tracker.MAX_WALL_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task,2)
    gate=local_tracker._gates[asyncio.get_running_loop()]
    if {p.pid for p in multiprocessing.active_children()} != baseline:
        # A native call may outlive the bounded cancellation return. Its slot
        # must stay reserved until cooperative exit; no second worker or kill.
        assert gate.locked()
        assert not local_tracker._worker_slot.acquire(blocking=False)
    await asyncio.wait_for(gate.acquire(),10)
    gate.release()
    assert {p.pid for p in multiprocessing.active_children()}==baseline
    assert not gate.locked()
    assert local_tracker._worker_slot.acquire(blocking=False)
    local_tracker._worker_slot.release()
    events=tracking_diagnostics.read(tmp_path,'session')
    assert any(e['event']=='job_cancelled' and e['records']>0 for e in events)
    assert events[-1]['event']=='job_retiring'
