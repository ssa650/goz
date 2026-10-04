"""Real local HTTP route regressions; no camera, Bluetooth or provider requests."""
import copy
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest

from backend.adaptive.session import AdaptiveSession
from backend.app import create_app
from backend.engine import Engine
from test_backend import FakeAdapter


@asynccontextmanager
async def local_session(tmp_path, monkeypatch):
    monkeypatch.setenv('GOZ_GAZE', 'off')
    monkeypatch.setenv('GOZ_EEG', 'off')
    monkeypatch.setenv('GOZ_REQUIRE_SENSORS', '0')
    adapter = FakeAdapter()
    engine = Engine(adapter, tmp_path, poll_seconds=.001)
    app = create_app(engine)
    async with app.router.lifespan_context(app):
        sensors = app.state.sensors
        session = AdaptiveSession(engine, sensors.gaze, sensors.eeg, tmp_path, 'Shared scene',
                                  [dict(name='SpongeBob'), dict(name='Patrick')], 15, '480P', None)
        sensors.session = session
        window_end = time.time() - 2
        for index in range(2):
            decision_id = str(uuid4())
            session.clips.append(dict(index=index, id=str(uuid4()), sessionId=session.id,
                status='playing' if index == 0 else 'ready', duration=15, ticks=[], track=[],
                decisionId=decision_id, decision=dict(id=decision_id, focus=None, reasons=[]),
                observationWindow=dict(end=window_end) if index else None,
                plan=dict(scene_title=f'Scene {index+1}', summary='Shared scene'),
                analysisStarted=True, detectionStatus='ready'))
        session.playing = 0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            yield session, engine, adapter, client
        assert adapter.submissions == []
        assert adapter.uploads == []


def event_body(session, index, kind, wall):
    return dict(session_id=session.id, clip=index, clip_id=session.clips[index]['id'], kind=kind, wall=wall*1000)


def tick_body(session, index=0, wall=None):
    return dict(session_id=session.id, clip=index, clip_id=session.clips[index]['id'],
                video_t=.5, playing=True, wall=(wall or time.time())*1000,
                rect=dict(x=0,y=0,w=640,h=360), epoch=0, playback_rate=1)


@pytest.mark.asyncio
async def test_late_predecessor_ended_repairs_transition_and_duplicate_events_preserve_first_times(tmp_path, monkeypatch):
    async with local_session(tmp_path, monkeypatch) as (session, engine, adapter, client):
        start = time.time() - .1
        ended = start - .075
        response = await client.post('/api/adaptive/playback-event', json=event_body(session, 1, 'playing', start))
        assert response.status_code == 200, response.text
        successor = session.clips[1]
        assert successor['transitionMs'] is None  # Predecessor end has not arrived yet.
        expected_feedback = start-successor['observationWindow']['end']
        assert successor['feedbackDelayS'] == pytest.approx(expected_feedback, abs=1e-5)

        response = await client.post('/api/adaptive/playback-event', json=event_body(session, 0, 'ended', ended))
        assert response.status_code == 200, response.text
        assert session.clips[0]['status'] == 'watched'
        assert successor['transitionMs'] == pytest.approx(75, abs=.01)
        assert session.clips[0]['endedAt'] == pytest.approx(ended, abs=1e-5)
        task_count = len(engine.tasks)

        # Browser resume notifications / delayed poll completion are duplicates,
        # not a second transition or an additional analysis/generation request.
        assert (await client.post('/api/adaptive/playback-event', json=event_body(session, 1, 'playing', start+.03))).status_code == 200
        assert (await client.post('/api/adaptive/playback-event', json=event_body(session, 0, 'ended', ended+.04))).status_code == 200
        assert (await client.post('/api/adaptive/ended', json=dict(session_id=session.id, clip=0))).status_code == 200
        assert successor['firstPresentedAt'] == pytest.approx(start, abs=1e-5)
        assert successor['transitionMs'] == pytest.approx(75, abs=.01)
        assert len(engine.tasks) == task_count
        assert adapter.submissions == []


@pytest.mark.asyncio
async def test_in_order_transition_has_same_measurement(tmp_path, monkeypatch):
    async with local_session(tmp_path, monkeypatch) as (session, _, _, client):
        ended = time.time() - .1
        assert (await client.post('/api/adaptive/playback-event', json=event_body(session, 0, 'ended', ended))).status_code == 200
        assert (await client.post('/api/adaptive/playback-event', json=event_body(session, 1, 'playing', ended+.075))).status_code == 200
        assert session.clips[1]['transitionMs'] == pytest.approx(75, abs=.01)


@pytest.mark.asyncio
@pytest.mark.parametrize('endpoint', ['tick', 'ended', 'stop', 'playback-event'])
@pytest.mark.parametrize('identity', ['old-session', None])
async def test_stale_or_missing_session_identity_cannot_mutate_active_session(tmp_path, monkeypatch, endpoint, identity):
    async with local_session(tmp_path, monkeypatch) as (session, engine, _, client):
        body = tick_body(session) if endpoint == 'tick' else event_body(session, 0, 'ended', time.time())
        if identity is None:
            body.pop('session_id')
        else:
            body['session_id'] = identity
        before = copy.deepcopy(session.clips)
        response = await client.post('/api/adaptive/'+endpoint, json=body)
        assert response.status_code == 400, response.text
        assert session.status == 'running'
        assert session.playing == 0 and session.clips == before
        assert engine.tasks == set()


@pytest.mark.asyncio
async def test_late_old_clip_tick_cannot_rewind_current_clip_over_http(tmp_path, monkeypatch):
    async with local_session(tmp_path, monkeypatch) as (session, _, _, client):
        now = time.time()
        response = await client.post('/api/adaptive/tick', json=tick_body(session, 1, now))
        assert response.status_code == 200, response.text
        assert session.playing == 1
        assert session.clips[1]['status'] == 'playing'
        old = copy.deepcopy(session.clips[0])
        response = await client.post('/api/adaptive/tick', json=tick_body(session, 0, now+.01))
        assert response.status_code == 200, response.text
        assert session.playing == 1
        assert session.clips[0] == old


@pytest.mark.asyncio
async def test_wrong_clip_identity_is_rejected_before_recording_tick_or_event(tmp_path, monkeypatch):
    async with local_session(tmp_path, monkeypatch) as (session, engine, _, client):
        before = copy.deepcopy(session.clips)
        for endpoint, body in [('tick', tick_body(session)), ('playback-event', event_body(session, 0, 'playing', time.time()))]:
            body['clip_id'] = str(uuid4())
            response = await client.post('/api/adaptive/'+endpoint, json=body)
            assert response.status_code == 400, response.text
        assert session.clips == before
        assert engine.tasks == set()


def diagnostic_body(session, destination=1, source=0, phase='state', epoch=1):
    anchor=min(destination,len(session.clips)-1)
    body=event_body(session,anchor,'diagnostic',time.time())
    body['diagnostic']=dict(phase=phase,index=destination,previousIndex=source,epoch=epoch,
        elapsedMs=50,holdMs=100,readyState=1,networkState=2,videoTime=0,mediaErrorCode=None,
        reason='buffering' if phase=='state' else None,
        sourceClipId=session.clips[source]['id'] if source is not None else None,
        targetClipId=session.clips[destination]['id'] if destination<len(session.clips) else None)
    return body


@pytest.mark.asyncio
async def test_transition_diagnostics_preserve_source_destination_and_do_not_count_as_viewing(tmp_path, monkeypatch):
    import json
    async with local_session(tmp_path,monkeypatch) as (session,engine,adapter,client):
        before=copy.deepcopy(session.clips)
        body=diagnostic_body(session)
        body['diagnostic']['url']='https://unexpected.example/private'
        response=await client.post('/api/adaptive/playback-event',json=body)
        assert response.status_code==200,response.text
        assert response.json()['recorded']
        row=json.loads((session.dir/'events.jsonl').read_text().splitlines()[-1])
        assert row['kind']=='browser_transition' and row['sourceClipId']==session.clips[0]['id']
        assert row['destinationClipId']==session.clips[1]['id'] and row['destinationIndex']==1
        assert 'url' not in row and 'https://' not in json.dumps(row)
        assert session.clips==before and engine.tasks==set()
        # No destination clip has been created yet: preserve the predecessor identity.
        absent=diagnostic_body(session,destination=2,source=1)
        assert (await client.post('/api/adaptive/playback-event',json=absent)).status_code==200
        assert session.events[-1]['destinationClipId'] is None
        assert session.events[-1]['sourceClipId']==session.clips[1]['id']
        assert session.events[-1]['clientDestinationKnown'] is False
        assert session.clips==before and adapter.submissions==[]


@pytest.mark.asyncio
async def test_diagnostic_ingestion_is_bounded_across_retries_and_client_epochs(tmp_path, monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (session,_,_,client):
        body=diagnostic_body(session)
        for _ in range(3): assert (await client.post('/api/adaptive/playback-event',json=body)).status_code==200
        assert len(session.events)==1
        for epoch in range(2,50):
            body['diagnostic']['epoch']=epoch
            assert (await client.post('/api/adaptive/playback-event',json=body)).status_code==200
        assert len(session.events)==24


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [dict(sourceClipId='wrong'),dict(targetClipId='wrong'),dict(phase='fake'),
    dict(index=100),dict(index=.5),dict(holdMs=-1),dict(networkState=5),dict(epoch=True),dict(errorName='https://private')])
async def test_malformed_transition_diagnostic_never_mutates_session(tmp_path,monkeypatch,change):
    async with local_session(tmp_path,monkeypatch) as (session,engine,_,client):
        body=diagnostic_body(session); body['diagnostic'].update(change)
        before=copy.deepcopy(session.clips)
        response=await client.post('/api/adaptive/playback-event',json=body)
        assert response.status_code==400,response.text
        assert session.clips==before and not session.events and engine.tasks==set()
