import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from backend.adaptive import director, fusion, profile, tracks
from backend.adaptive.sensors import EegFeed, GazeFeed
from backend.adaptive.session import AdaptiveSession
from backend.demo import DemoAdapter
from backend.engine import Engine
from backend.sensor_setup import SensorSetup
from test_adaptive import watch, png
from backend.frames import verify_image, ffmpeg


def test_blinks_low_confidence_and_head_turns_do_not_create_character_dwell():
    ticks, track, gaze, eeg = watch()
    for sample in gaze:
        sample['blink'] = True
    analysis = fusion.analyze(fusion.label(gaze, ticks, track), eeg, track, ['Ana', 'Bea'])
    assert all(c['dwell_s'] == 0 for c in analysis['characters'].values())
    assert analysis['gaze_confidence'] == 0
    for sample in gaze:
        sample.update(blink=False, confidence=.1)
    assert not any(e['target'] for e in fusion.label(gaze, ticks, track))
    for sample in gaze:
        sample.update(confidence=1, yaw=40)
    assert not any(e['target'] for e in fusion.label(gaze, ticks, track))


def test_tracking_gap_never_becomes_a_long_fixation_or_dwell():
    ticks, track, gaze, _ = watch()
    timeline = fusion.label([gaze[0], gaze[-1]], ticks, track)
    assert fusion.fixations(timeline) == []
    analysis = fusion.analyze(timeline, [], track, ['Ana', 'Bea'])
    assert analysis['watched_s'] <= .2
    assert max(c['dwell_s'] for c in analysis['characters'].values()) <= .2
    original = profile.new_profile(['Ana', 'Bea'])
    updated, _ = profile.update(original, analysis)
    assert updated['characters'] == original['characters']


def test_noisy_eeg_has_zero_weight_but_gaze_still_learns():
    ticks, track, gaze, eeg = watch()
    timeline = fusion.label(gaze, ticks, track)
    analysis = fusion.analyze(timeline, eeg, track, ['Ana', 'Bea'], eeg_confidence=0)
    assert analysis['eeg_confidence'] == 0 and analysis['eeg_mean_z'] == 0
    assert all(c['response'] == c['attention'] and not c['strong'] for c in analysis['characters'].values())
    learned, _ = profile.update(profile.new_profile(['Ana', 'Bea']), analysis)
    assert profile.decide(learned, analysis)['focus'] == 'Bea'


def test_default_sensor_policy_continues_without_hardware_but_strict_gate_remains(tmp_path, monkeypatch):
    monkeypatch.delenv('GOZ_REQUIRE_SENSORS', raising=False)
    sensors = SimpleNamespace(gaze=GazeFeed(), eeg=EegFeed(), gaze_mode='off', eeg_mode='off')
    setup = SensorSetup(sensors, tmp_path)
    assert setup.snapshot()['generationReady']
    setup.require_ready()
    monkeypatch.setenv('GOZ_REQUIRE_SENSORS', '1')
    strict = SensorSetup(sensors, tmp_path)
    with pytest.raises(Exception, match='Sensor setup is not ready'):
        strict.require_ready()


@pytest.mark.asyncio
async def test_florence_partial_failure_keeps_successful_detection_and_cache(tmp_path, monkeypatch):
    from test_detection_acceptance import CASES, fixture_image
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'cache-key')
    jpeg, w, h = fixture_image(CASES[1])
    # Distinct frames give distinct cache keys. The failed frame stays unavailable.
    other, ow, oh = fixture_image(CASES[0])
    monkeypatch.setattr(tracks, 'sample_scene_frames', lambda *a: [
        dict(t=0, jpeg=jpeg, width=w, height=h, shot=0, cut=False),
        dict(t=2, jpeg=other, width=ow, height=oh, shot=0, cut=False)])
    calls = []
    def handler(request):
        import json
        body = json.loads(request.content)
        assert 'text_input' not in body
        calls.append(body)
        if len(calls) > 1:
            return httpx.Response(503)
        return httpx.Response(200, json=CASES[1]['response'])
    result = await tracks.detect_characters(video, [{'name': 'SpongeBob'}, {'name': 'Patrick'}], 'test',
                                            transport=httpx.MockTransport(handler), directory=tmp_path)
    assert set(result[0]['boxes']) == {'SpongeBob', 'Patrick'}
    assert result[1]['status'] == 'unavailable' and result[1]['boxes'] == {}
    assert len(list((tmp_path / 'detections').glob('*.json'))) == 1
    cached = await tracks.detect_characters(video, [{'name': 'SpongeBob'}, {'name': 'Patrick'}], 'test',
                                           transport=httpx.MockTransport(handler), directory=tmp_path)
    assert cached[0]['cache_hit'] and cached[0]['boxes'] == result[0]['boxes']
    assert len(calls) == 3  # only the failed frame is retried on a new invocation


@pytest.mark.asyncio
async def test_florence_auth_failure_stops_pending_paid_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(tracks, 'sample_scene_frames', lambda *a: [dict(t=i, jpeg=b'jpeg', width=100,
        height=100, shot=0, cut=False) for i in range(8)])
    calls = []
    def reject(request):
        calls.append(request)
        return httpx.Response(403)
    video = tmp_path / 'unused'
    video.write_bytes(b'fixture')
    result = await tracks.detect_characters(video, [{'name': 'A'}, {'name': 'B'}], 'test', transport=httpx.MockTransport(reject))
    assert len(calls) <= 2
    assert all(f['unknown'] == ['A', 'B'] and not f['boxes'] for f in result)


def new_session(engine, **kwargs):
    return AdaptiveSession(engine, GazeFeed(), EegFeed(), engine.directory, 'A mystery in Bikini Bottom.',
                           [{'name': 'SpongeBob'}, {'name': 'Patrick'}], 5, '480P', verify_image(png()), **kwargs)


@pytest.mark.asyncio
async def test_controller_and_florence_failures_still_produce_playable_clip(tmp_path, monkeypatch):
    adapter = DemoAdapter(tmp_path / 'demo')
    adapter.demo = False  # local media transport with production failure handling
    engine = Engine(adapter, tmp_path, poll_seconds=.01)
    async def broken(*args, **kwargs):
        raise ValueError('provider unavailable')
    monkeypatch.setattr(director, 'write_scene', broken)
    monkeypatch.setattr(tracks, 'detect_characters', broken)
    path = tmp_path / 'test.mp4'
    await ffmpeg('-f', 'lavfi', '-i', 'color=c=yellow:s=160x90:r=10', '-t', '1', '-pix_fmt', 'yuv420p', path)
    async def complete(job, images):
        job.update(status='completed')
    async def local_media(job):
        return path
    monkeypatch.setattr(engine, 'run_job', complete)
    monkeypatch.setattr(engine, 'media_path', local_media)
    s = new_session(engine, tracker='fal')  # Explicit rollback provider under test.
    await s.make_scene({}, s.opening, None, [])
    assert s.status == 'running' and s.clips[0]['status'] == 'ready'
    assert s.clips[0]['track'] == [] and s.clips[0]['writer'] == 'template-fallback'
    await asyncio.gather(*s.detection_tasks)
    assert {w['component'] for w in s.warnings} == {'Controller', 'Florence'}
    await engine.close()


@pytest.mark.asyncio
async def test_failed_generation_holds_story_without_replay_or_resubmission(tmp_path, monkeypatch):
    engine = Engine(DemoAdapter(tmp_path / 'demo'), tmp_path, poll_seconds=.01)
    s = new_session(engine)
    await s.make_scene({}, s.opening, None, [])
    s.clips[0]['status'] = 'watched'
    calls = []
    async def broken(job, images):
        calls.append(job)
        job.update(status='failed', error='out of credits')
    monkeypatch.setattr(engine, 'run_job', broken)
    await s.make_scene({}, s.opening, None, [])
    assert s.clips[1]['fallback'] and not s.clips[1].get('url')
    assert 'not generated' in s.clips[1]['fallbackReason'] and s.status == 'failed'
    await s.make_scene({}, s.opening, None, [])
    assert len(s.clips) == 2 and len(calls) == 1
    assert len(s.story['scenes']) == 1, 'never advance story state after failure'
    await engine.close()


@pytest.mark.asyncio
async def test_stop_during_planning_submits_no_job(tmp_path, monkeypatch):
    adapter = DemoAdapter(tmp_path / 'demo')
    adapter.demo = False
    engine = Engine(adapter, tmp_path)
    s = new_session(engine)
    started, release = asyncio.Event(), asyncio.Event()
    async def delayed(story, profile, decision, duration):
        started.set()
        await release.wait()
        return director.template(story, decision, duration, s.names), 'test'
    monkeypatch.setattr(director, 'write_scene', delayed)
    task = asyncio.create_task(s.make_scene({}, s.opening, None, []))
    await started.wait()
    s.stop()
    release.set()
    await task
    assert s.status == 'stopped' and not engine.jobs
    await engine.close()


def test_custom_story_advances_events_and_preserves_cast():
    story = {'premise': 'Patrick has a secret box.', 'characters': [{'name': 'Patrick', 'description': 'pink starfish'}], 'scenes': []}
    opening = director.continuation(story)
    story['scenes'].append({'summary': 'Patrick shows the box.'})
    next_scene = director.continuation(story)
    assert opening != next_scene
    assert 'pink starfish' in opening and 'pink starfish' in next_scene
    assert 'Patrick shows the box.' in next_scene


@pytest.mark.asyncio
async def test_scene_limit_finishes_after_playback_not_at_analysis_trigger(tmp_path, monkeypatch):
    engine = Engine(DemoAdapter(tmp_path / 'demo'), tmp_path, poll_seconds=.01)
    s = new_session(engine)
    s.max_scenes = 1
    await s.make_scene({}, s.opening, None, [])
    s.clips[0]['analysisStarted'] = True
    await s.adapt(0)
    assert s.status == 'running'
    s.ended(0)
    assert s.status == 'finished'
    await engine.close()


@pytest.mark.asyncio
async def test_blank_frame_never_calls_florence(tmp_path):
    video = tmp_path / 'black.mp4'
    await ffmpeg('-f', 'lavfi', '-i', 'color=black:s=160x90:r=10', '-t', '2', '-pix_fmt', 'yuv420p', video)
    def forbidden(request):
        raise AssertionError('Blank frames should not incur a paid query')
    result = await tracks.detect_characters(video, [{'name':'SpongeBob'}], 'test', transport=httpx.MockTransport(forbidden))
    assert result and all(not f['boxes'] and not f['unknown'] for f in result)


@pytest.mark.asyncio
async def test_story_objects_are_measured_separately_from_character_preferences(tmp_path):
    engine = Engine(DemoAdapter(tmp_path / 'demo'), tmp_path, poll_seconds=.01)
    s = new_session(engine, objects=[{'name':'secret box','description':'small brown box'}])
    s.max_scenes = 1
    await s.make_scene({}, s.opening, None, [])
    assert 'secret box' in s.clips[0]['track'][0]['boxes']
    s.clips[0]['analysisStarted'] = True
    await s.adapt(0)
    analysis = s.clips[0]['analysis']
    assert 'secret box' in analysis['objects']
    assert 'secret box' not in analysis['characters'] and 'secret box' not in s.profile['characters']
    await engine.close()


@pytest.mark.asyncio
async def test_active_adaptive_session_blocks_other_generation_paths(tmp_path):
    from uuid import uuid4
    from backend.fal_adapter import FalError
    engine = Engine(DemoAdapter(tmp_path / 'demo'), tmp_path)
    clip = engine.clips.add({'prompt':'A scene'})
    engine.adaptive_active = lambda: True
    assert engine.busy()
    with pytest.raises(FalError, match='Stop the adaptive story'):
        await engine.clips.generate(clip['id'], str(uuid4()))
    await engine.close()
