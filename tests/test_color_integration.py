"""Shared color provider contract, offline coverage, and deadline regressions."""
import asyncio
import time
from copy import deepcopy
import pytest
from backend.adaptive import tracks,local_tracker
from backend.adaptive.color_detector import ColorCharacterDetector
from backend.adaptive.session import AdaptiveSession
from backend.frames import ffmpeg
from test_adaptive_deadline import clock,setup,tick
from test_local_tracking import ROOT,CAST


def test_color_is_test_default_and_rollback_selections_remain_explicit(monkeypatch):
    monkeypatch.delenv('GOZ_TRACKER',raising=False)
    assert tracks.tracking_provider()=='color'
    for provider in ('color','fal','opencv','people','yoloe'):
        assert tracks.tracking_provider(provider)==provider
    assert not tracks.yoloe_availability()['available']


@pytest.mark.asyncio
async def test_color_worker_real_full_clip_has_tail_provenance_and_no_cloud(tmp_path):
    video=tmp_path/'full-color.mp4'
    await ffmpeg('-loop','1','-i',ROOT/'presets/secret-box/frames/00-30.jpg','-t','14.5',
        '-vf','scale=640:400','-r','4','-pix_fmt','yuv420p',video)
    start=time.perf_counter();first=[]
    def progress(records):
        if records and not first:first.append(time.perf_counter()-start)
    result=await local_tracker.detect_local(video,CAST,clip_id='c',session_id='s',generation_id='g',provider='color',on_progress=progress)
    assert len(result)==58 and result[0]['t']==0 and result[-1]['t']==14.25
    assert result[-1]['valid_until']>=14.5 and set(result[-1]['boxes'])==set(CAST)
    assert all(r['clip_id']=='c' and r['session_id']=='s' and r['generation_id']=='g' for r in result)
    assert all(r['source']=='opencv_color_shape_demo' and r['experimental'] for r in result)
    assert all(r['resources']['flow_fps']==4 and r['resources']['opencv_threads']==1 for r in result)
    print('COLOR_WORKER_BUDGET',dict(first_progress_s=first[0],total_wall_s=time.perf_counter()-start,last=result[-1]['scheduling']))


@pytest.mark.asyncio
async def test_color_session_uses_shared_scheduler_only(tmp_path,monkeypatch,clock):
    s,clip,engine,_=setup(tmp_path,monkeypatch,clock);s.tracker='color'
    async def local(path,names,**kwargs):
        assert kwargs['provider']=='color' and callable(kwargs['playback_state'])
        records=[dict(t=0,boxes={},source='opencv_color_shape_demo'),
                 dict(t=14.25,boxes={'Ana':[0,0,.4,1]},source='opencv_color_shape_demo')]
        kwargs['on_progress'](records);return records
    def forbidden(*a,**kw):raise AssertionError('Color provider called cloud')
    monkeypatch.setattr(local_tracker,'detect_local',local)
    monkeypatch.setattr(tracks,'detect_characters',forbidden)
    result=await s.track('unused',None,None,15,clip['id'])
    assert result[-1]['t']==14.25
    await engine.close()


@pytest.mark.asyncio
async def test_color_pending_worker_does_not_delay_freeze_generation_or_end(tmp_path,monkeypatch,clock):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock);s.tracker='color'
    monkeypatch.setattr(s,'start_detection',AdaptiveSession.start_detection.__get__(s))
    entered=asyncio.Event()
    async def pending(*args):entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(s,'track',pending)
    s.start_detection(clip,'unused',None,None,15);await entered.wait()
    task=s.detection_by_clip[clip['id']]
    monkeypatch.setattr(s,'start_detection',lambda *a:None)
    tick(s,clock,0);tick(s,clock,5.0);frozen=deepcopy(clip['frozenEvidence'])
    await asyncio.wait_for(s.task,.5)
    assert len(submitted)==1 and not task.done()
    started=time.perf_counter();s.ended(0);assert time.perf_counter()-started<.1
    await asyncio.gather(task,return_exceptions=True)
    assert task.cancelled() and clip['frozenEvidence']==frozen
    await engine.close()
