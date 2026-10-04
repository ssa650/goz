"""Candidate-only regression tests; no production app, device or provider calls."""
import asyncio
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import pytest

from backend.adaptive import local_tracker, tracking_diagnostics

ROOT=Path('/Users/shayan/goz')
CAST=['SpongeBob','Patrick']


def video(path, *, blank=False, frames=24):
    rgb=np.array(Image.open(ROOT/'presets/secret-box/frames/00-30.jpg').convert('RGB'))
    frame=cv2.cvtColor(cv2.resize(rgb,(640,400)),cv2.COLOR_RGB2BGR)
    if blank:
        frame[:]=0
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(640,400))
    assert writer.isOpened()
    try:
        for _ in range(frames):
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.asyncio
async def test_setup_readiness_reuses_pid_and_resets_identity_and_provenance(tmp_path):
    baseline={p.pid for p in multiprocessing.active_children()}
    events=[]
    def observer(event):
        events.append(event)
        raise RuntimeError('diagnostic observer failed')
    worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10,on_diagnostic=observer)
    pid=worker.process.pid
    try:
        assert [e['diagnostic_event'] for e in events]==['warming','warm_ready']
        assert worker.ready['opencv_threads']==1
        source=tmp_path/'source.mp4';empty=tmp_path/'empty.mp4'
        video(source,frames=12);video(empty,blank=True,frames=12)
        first=await local_tracker.detect_local(source,CAST,clip_id='first',session_id='session',
            generation_id='one',provider='color',warm_worker=worker,diagnostics_directory=tmp_path)
        assert [f['t'] for f in first]==[0,.25,.5,.75,1,1.25]
        assert all(set(f['boxes'])==set(CAST) for f in first)
        assert worker.process.pid==pid and worker.active_token is None
        worker.output.put(dict(token=-1,record=first[-1]))
        second=await local_tracker.detect_local(empty,CAST,clip_id='second',session_id='session',
            generation_id='two',provider='color',warm_worker=worker,diagnostics_directory=tmp_path)
        assert len(second)==6 and all(f['boxes']=={} for f in second)
        assert second[0]['shot_id']==0 and not second[0]['cut']
        assert all(f['clip_id']=='second' and f['generation_id']=='two' for f in second)
        assert worker.process.pid==pid and worker.active_token is None
        logs=tracking_diagnostics.read(tmp_path,'session')
        reused=[r for r in logs if r['event']=='worker_reused']
        assert len(reused)==2 and {r['workerPid'] for r in reused}=={pid}
        assert any(r['event']=='transport_rejected' and r['reason']=='warm_job_token_mismatch' for r in logs)
    finally:
        await asyncio.wait_for(worker.close(),20)
    assert {p.pid for p in multiprocessing.active_children()}==baseline


@pytest.mark.asyncio
async def test_cancelled_warm_job_retires_before_next_job_without_killing(tmp_path):
    worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10)
    source=tmp_path/'source.mp4';video(source,frames=40)
    observed=asyncio.Event()
    try:
        task=asyncio.create_task(local_tracker.detect_local(source,CAST,clip_id='one',session_id='session',
            provider='color',warm_worker=worker,on_progress=lambda rows:observed.set() if rows else None))
        await asyncio.wait_for(observed.wait(),30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task,2)
        second=await asyncio.wait_for(local_tracker.detect_local(source,CAST,clip_id='two',session_id='session',
            provider='color',warm_worker=worker),20)
        assert second and all(f['clip_id']=='two' for f in second)
        assert worker.active_token is None and worker.process.is_alive()
        assert local_tracker._worker_slot.acquire(blocking=False)
        local_tracker._worker_slot.release()
    finally:
        await asyncio.wait_for(worker.close(),20)


@pytest.mark.asyncio
async def test_provider_mismatch_releases_slots_and_does_not_poison_warm_worker(tmp_path):
    worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=10)
    source=tmp_path/'source.mp4';video(source,frames=8)
    try:
        with pytest.raises(ValueError,match='does not match'):
            await local_tracker.detect_local(source,CAST,clip_id='one',session_id='session',
                provider='opencv',warm_worker=worker)
        assert not local_tracker._gates[asyncio.get_running_loop()].locked()
        assert local_tracker._worker_slot.acquire(blocking=False)
        local_tracker._worker_slot.release()
        result=await local_tracker.detect_local(source,CAST,clip_id='two',session_id='session',
            provider='color',warm_worker=worker)
        assert result
    finally:
        await asyncio.wait_for(worker.close(),20)


@pytest.mark.asyncio
async def test_idle_expiry_fails_closed_instead_of_importing_again_at_playback(tmp_path):
    worker=await local_tracker.prewarm_local_tracker('color',idle_seconds=1)
    try:
        await asyncio.to_thread(worker.process.join,10)
        assert not worker.process.is_alive()
        with pytest.raises(RuntimeError,match='prepare it before playback'):
            await local_tracker.detect_local(tmp_path/'unused.mp4',CAST,clip_id='one',session_id='session',
                provider='color',warm_worker=worker)
        assert not local_tracker._gates[asyncio.get_running_loop()].locked()
    finally:
        await asyncio.wait_for(worker.close(),20)


@pytest.mark.asyncio
async def test_setup_timeout_while_waiting_for_slot_reports_failure_and_does_not_release_other_owner(monkeypatch):
    assert local_tracker._worker_slot.acquire(blocking=False)
    monkeypatch.setattr(local_tracker,'MAX_WALL_SECONDS',.05)
    events=[]
    try:
        with pytest.raises(TimeoutError):
            await local_tracker.prewarm_local_tracker('color',on_diagnostic=events.append)
        assert [e['diagnostic_event'] for e in events]==['warming','warm_failed']
        assert not local_tracker._worker_slot.acquire(blocking=False)
        assert not local_tracker._gates[asyncio.get_running_loop()].locked()
    finally:
        local_tracker._worker_slot.release()
