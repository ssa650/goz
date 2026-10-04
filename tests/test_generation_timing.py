"""Timing provenance, malformed results and local media readiness checks."""
import asyncio
import math

import pytest

from backend.engine import Engine, timing_ms
from backend.frames import ffmpeg
from test_backend import FakeAdapter


@pytest.mark.parametrize('value', [None, '2.3', True, -1, math.inf, math.nan, {}])
def test_unavailable_or_bad_provider_timing_is_not_invented(value):
    assert timing_ms(value) is None


class TimedAdapter(FakeAdapter):
    def __init__(self, metrics=None, timings=None, malformed=False):
        super().__init__()
        self.metrics, self.timings, self.malformed = metrics, timings, malformed
    async def submit(self, model, payload):
        await asyncio.sleep(.002)
        return await super().submit(model, payload)
    async def status(self, model, request_id):
        self.status_calls += 1
        if self.status_calls == 1:
            return dict(status='IN_PROGRESS')
        return dict(status='COMPLETED', metrics=self.metrics)
    async def result(self, model, request_id):
        await asyncio.sleep(.002)
        return ['invalid'] if self.malformed else dict(video=dict(url='https://fal.media/test.mp4'), timings=self.timings)


@pytest.mark.asyncio
@pytest.mark.parametrize('reported', [False, True])
async def test_client_observed_timings_remain_separate_from_provider_timings(tmp_path, reported):
    adapter = TimedAdapter(dict(inference_time=1.2) if reported else None, dict(inference=.7) if reported else None)
    engine = Engine(adapter, tmp_path, poll_seconds=.001)
    job = engine.new_job(dict(mode='text', prompt='test', duration=5, resolution='480P'))
    await engine.run_job(job, {})
    assert job['status'] == 'completed'
    for key in ['submissionMs', 'payloadConstructionMs', 'queueObservedMs', 'providerObservedElapsedMs', 'resultRetrievalMs']:
        assert job[key] >= 0
    assert job['providerRunnerMs'] == (1200 if reported else None)
    assert job['providerInferenceMs'] == (700 if reported else None)
    assert 'mediaReadyAt' not in job
    assert 'image_url' not in job['generationInput']
    await engine.close()


@pytest.mark.asyncio
async def test_malformed_result_never_becomes_completed_playable_clip(tmp_path):
    engine = Engine(TimedAdapter(malformed=True), tmp_path, poll_seconds=.001)
    job = engine.new_job(dict(mode='text', prompt='test', duration=5, resolution='480P'))
    await engine.run_job(job, {})
    assert job['status'] == 'failed' and 'malformed result' in job['error']
    assert 'video' not in job
    await engine.close()


@pytest.mark.asyncio
async def test_media_ready_is_measured_after_validated_download_once(tmp_path, monkeypatch):
    import backend.engine as engine_module
    source = tmp_path/'source.mp4'
    await ffmpeg('-f', 'lavfi', '-i', 'color=c=green:size=64x64:rate=10', '-t', '0.5', '-pix_fmt', 'yuv420p', source)
    downloads = []
    async def download(url, path):
        downloads.append(url)
        path.write_bytes(source.read_bytes())
    monkeypatch.setattr(engine_module, 'download_video', download)
    engine = Engine(FakeAdapter(), tmp_path/'data', poll_seconds=.001)
    job = engine.new_job(dict(mode='text', prompt='test', duration=5, resolution='480P'))
    await engine.run_job(job, {})
    paths = await asyncio.gather(engine.media_path(job), engine.media_path(job))
    assert paths[0] == paths[1] and len(downloads) == 1
    assert job['mediaDownloadMs'] >= 0 and job['mediaValidationMs'] >= 0
    assert job['mediaReadyElapsedMs'] >= job['totalElapsedMs']
    assert job['actualDuration'] == .5 and job['hasAudio'] is False
    await engine.close()


@pytest.mark.asyncio
async def test_html_returned_as_video_is_rejected_before_ready(tmp_path, monkeypatch):
    import backend.engine as engine_module
    async def download(url, path):
        path.write_bytes(b'<html>provider error</html>')
    monkeypatch.setattr(engine_module, 'download_video', download)
    engine = Engine(FakeAdapter(), tmp_path, poll_seconds=.001)
    job = engine.new_job(dict(mode='text', prompt='test', duration=5, resolution='480P'))
    await engine.run_job(job, {})
    with pytest.raises(Exception):
        await engine.media_path(job)
    assert 'mediaReadyAt' not in job and not (engine.media/f"{job['id']}.mp4").exists()
    assert not list(engine.media.glob('*.part'))
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('frame', [None, b'partial'])
async def test_container_header_without_a_complete_decoded_frame_never_becomes_ready(tmp_path, monkeypatch, frame):
    import backend.engine as engine_module
    import backend.frames as frames
    def reader(*args, **kwargs):
        assert '-frames:v' in kwargs['output_params']
        assert '-threads' in kwargs['input_params']
        yield dict(size=(64,64),duration=15)
        if frame is not None:
            yield frame
    async def download(url, path):
        path.write_bytes(b'container-with-truncated-frame')
    monkeypatch.setattr(frames.imageio_ffmpeg, 'read_frames', reader)
    monkeypatch.setattr(engine_module, 'download_video', download)
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    job=engine.new_job(dict(mode='text',prompt='test',duration=15,resolution='480P'))
    await engine.run_job(job,{})
    with pytest.raises(ValueError,match='video frame'):
        await engine.media_path(job)
    assert job['mediaStatus']=='failed'
    assert job['mediaFailedAt']>=job['mediaDownloadStartedAt']
    assert 'mediaReadyAt' not in job and 'actualDuration' not in job
    assert not (engine.media/f"{job['id']}.mp4").exists()
    assert not list(engine.media.glob('*.part'))
    await engine.close()


@pytest.mark.asyncio
async def test_cancelled_download_records_phase_and_removes_partial_without_resubmission(tmp_path, monkeypatch):
    import backend.engine as engine_module
    entered=asyncio.Event()
    async def download(url,path):
        path.write_bytes(b'partial'); entered.set(); await asyncio.Event().wait()
    monkeypatch.setattr(engine_module,'download_video',download)
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    job=engine.new_job(dict(mode='text',prompt='test',duration=15,resolution='480P'))
    await engine.run_job(job,{})
    task=asyncio.create_task(engine.media_path(job)); await entered.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert job['mediaStatus']=='cancelled' and 'mediaReadyAt' not in job
    assert not list(engine.media.glob('*.part'))
    assert len(engine.adapter.submissions)==1
    await engine.close()
