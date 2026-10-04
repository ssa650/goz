"""Provider diagnostics use mocked HTTP only; no generation or device activation."""
import json

import httpx
import pytest

from backend.app import create_app
from backend.adaptive import director
from backend.adaptive.sensors import EegFeed, GazeFeed
from backend.adaptive.session import AdaptiveSession
from backend.engine import Engine
from backend.fal_adapter import FalAdapter, FalError
from backend.provider_errors import ProviderFailureJournal, diagnostic, safe_diagnostic, safe_id, safe_text

REQUEST_ID = '01a106b5-ba11-7ed3-bbb6-dfcddd25d537'


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['submit', 'status', 'result', 'cancel'])
async def test_top_up_preserves_safe_context_without_retries(stage):
    calls = []
    def respond(request):
        calls.append(request.method)
        return httpx.Response(403, json={
            'detail': {'message': 'User is locked. Reason: TOP_UP.',
                       'error': 'Authorization: Bearer other-secret\nhttps://private.example/path?token=hidden'},
            'request_id': REQUEST_ID, 'input': {'api_key': 'must-not-copy'},
            'headers': {'x-secret': 'never-copy'}},
            headers={'x-request-id': 'different-id', 'x-private': 'never-copy'})
    adapter = FalAdapter('test-secret', transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(FalError) as caught:
            if stage == 'submit':
                await adapter.submit('owner/model', {'prompt': 'mock'})
            else:
                await getattr(adapter, stage)('owner/model', REQUEST_ID)
        info = caught.value.provider_error
        assert info['httpStatus'] == 403 and caught.value.status == 403
        assert info['requestId'] == REQUEST_ID and info['stage'] == stage
        assert info['category'] == 'account_locked_top_up'
        assert info['retryability'] == 'manual_review' and info['automaticRetry'] is False
        assert 'balance cause is unverified' in str(caught.value)
        assert 'No new generation was automatically submitted' in str(caught.value)
        serialized = json.dumps(info) + str(caught.value)
        for secret in ('other-secret', 'private.example', 'hidden', 'never-copy', 'must-not-copy', 'Authorization:'):
            assert secret not in serialized
        assert calls == [{'submit': 'POST', 'cancel': 'PUT'}.get(stage, 'GET')]
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,category,retry', [(401, 'authentication', 'manual_review'),
    (403, 'permission', 'manual_review'), (422, 'invalid_request', 'none'),
    (429, 'rate_limit', 'same_request_read'), (503, 'provider_failure', 'same_request_read')])
async def test_status_category_and_read_retry_scope(status, category, retry):
    adapter = FalAdapter('test-secret', transport=httpx.MockTransport(lambda request:
        httpx.Response(status, json={'detail': [{'msg': 'Invalid field', 'input': 'never-copy'}]})))
    try:
        with pytest.raises(FalError) as caught:
            await adapter.status('owner/model', REQUEST_ID)
        assert caught.value.provider_error['category'] == category
        assert caught.value.provider_error['retryability'] == retry
        assert 'never-copy' not in str(caught.value)
        assert diagnostic('unavailable', status=status, stage='submit')['retryability'] != 'same_request_read'
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [b'<html>Secret page https://private.example</html>', b'x' * 70000])
async def test_unstructured_or_large_error_body_is_omitted(body):
    adapter = FalAdapter(transport=httpx.MockTransport(lambda request: httpx.Response(502, content=body)))
    try:
        with pytest.raises(FalError) as caught:
            await adapter.result('owner/model', REQUEST_ID)
        assert caught.value.provider_error['httpStatus'] == 502
        assert caught.value.provider_error['detail'] == 'Provider did not return a safe diagnostic detail.'
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_transport_error_omits_url_and_paid_post_remains_one_shot():
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ReadTimeout('https://private.example?token=hidden', request=request)
    adapter = FalAdapter(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(FalError) as caught:
            await adapter.submit('owner/model', {})
        assert len(calls) == 1
        assert caught.value.provider_error['httpStatus'] is None
        assert caught.value.provider_error['category'] == 'transport_error'
        assert caught.value.provider_error['retryability'] == 'manual_review'
        assert 'hidden' not in str(caught.value) and 'http' not in str(caught.value)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_success_http_status_with_provider_error_preserves_real_status():
    adapter = FalAdapter(transport=httpx.MockTransport(lambda request: httpx.Response(200,
        json={'status': 'COMPLETED', 'error': {'message': 'User is locked. Reason: TOP_UP.'}})))
    try:
        with pytest.raises(FalError) as caught:
            await adapter.status('owner/model', REQUEST_ID)
        assert caught.value.provider_error['httpStatus'] == 200
        assert caught.value.status is None  # application handler must not send a successful error response
        assert caught.value.provider_error['category'] == 'account_locked_top_up'
        assert caught.value.provider_error['requestId'] == REQUEST_ID
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_ambiguous_submission_stays_blocked_across_restart_with_diagnostics(tmp_path):
    calls = []
    def respond(request):
        calls.append(request.method)
        raise httpx.ReadTimeout('lost confirmation', request=request)
    adapter = FalAdapter('test-secret', transport=httpx.MockTransport(respond))
    engine = Engine(adapter, tmp_path)
    try:
        job = engine.new_job(dict(mode='text', prompt='Mock scene.', duration=5, resolution='480P'))
        await engine.run_job(job, {})
        assert job['status'] == 'failed' and job['requestUncertain'] and engine.busy()
        assert job['providerError']['stage'] == 'submit' and job['providerError']['httpStatus'] is None
        assert job['providerError']['requestId'] is None
        assert job['providerError']['retryability'] == 'manual_review'
        assert calls == ['POST']
    finally:
        await engine.close()
    restarted = Engine(FalAdapter(), tmp_path)
    try:
        assert restarted.jobs[job['id']]['providerError'] == job['providerError']
        assert restarted.busy() and restarted.jobs[job['id']]['requestUncertain']
    finally:
        await restarted.close()


def test_redaction_is_bounded_precedes_truncation_and_handles_unknown_credentials(monkeypatch):
    monkeypatch.setenv('PROVIDER_TEST_TOKEN', 'environment-secret')
    detail = ('TOP_UP token="quoted secret" key=unknown-secret sk-testabc123 '
              'eyJhbGci.aBc.def www.private.example/path private.example/thing '
              'Bearer bearer-secret environment-secret supplied-secret\x00')
    result = safe_text(detail, ('supplied-secret',))
    for value in ('quoted secret', 'unknown-secret', 'sk-testabc123', 'eyJhbGci', 'private.example',
                  'bearer-secret', 'environment-secret', 'supplied-secret', '\x00'):
        assert value not in result
    secret = 'known-short-secret'
    assert safe_text('note ' * 79 + secret, (secret,)).endswith('[reda')
    assert len(safe_text('✓' * 5000)) == 400
    assert safe_id('https://private.example/key') is None
    assert safe_id('environment-secret') is None
    assert safe_id('token=hidden') is None
    assert safe_id(REQUEST_ID) == REQUEST_ID
    assert safe_diagnostic({'category': [], 'stage': {}, 'detail': ['raw data'], 'httpStatus': True})['httpStatus'] is None


def test_provider_journal_rotation_permissions_and_allowlist(tmp_path):
    journal = ProviderFailureJournal(tmp_path)
    info = diagnostic('User is locked. Reason: TOP_UP.', status=403, stage='result', request_id=REQUEST_ID)
    info['rawHeaders'] = {'authorization': 'never-copy'}
    info['url'] = 'https://private.example/never-copy'
    for _ in range(250):
        journal.record(info, job_id='job-1', session_id='session-1')
    assert journal.write_error is None
    files = list(journal.path.parent.iterdir())
    assert len(files) == 2
    for path in files:
        assert path.stat().st_size <= journal.MAX_BYTES
        assert path.stat().st_mode & 0o777 == 0o600
        assert 'never-copy' not in path.read_text()
    row = json.loads(journal.path.read_text().splitlines()[-1])
    assert row['event'] == 'provider_failed' and row['providerError']['requestId'] == REQUEST_ID


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_stage', ['status', 'result'])
async def test_confirmed_submission_failure_persists_and_reaches_adaptive_public_error(tmp_path, monkeypatch, failure_stage):
    calls = []
    def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == 'POST':
            return httpx.Response(200, json={'request_id': REQUEST_ID})
        if request.url.path.endswith('/status') and failure_stage == 'result':
            return httpx.Response(200, json={'status': 'COMPLETED'})
        return httpx.Response(403, json={'detail': 'User is locked. Reason: TOP_UP. key=leaked-secret https://private.example'})
    adapter = FalAdapter('test-secret', transport=httpx.MockTransport(respond))
    engine = Engine(adapter, tmp_path, poll_seconds=.001)
    session = AdaptiveSession(engine, GazeFeed(), EegFeed(), tmp_path, 'A test scene.',
        [{'name': 'A'}, {'name': 'B'}], 5, '480P', None)
    async def plan(story, profile, decision, duration):
        return director.template(story, decision, duration, session.names), 'mock'
    monkeypatch.setattr(director, 'write_scene', plan)
    try:
        await session.make_scene({}, None, None, [])
        assert session.status == 'failed'
        job = next(iter(engine.jobs.values()))
        assert job['requestId'] == REQUEST_ID and job['providerStatus'] == 403
        assert job['providerError']['stage'] == failure_stage
        assert job['providerError']['httpStatus'] == 403
        public = session.public()
        assert public['providerError'] == job['providerError']
        assert REQUEST_ID in public['error'] and 'HTTP 403' in public['error']
        assert any(row['kind'] == 'provider_failed' for row in public['events'])
        await session.make_scene({}, None, None, [])
        assert len([row for row in calls if row[0] == 'POST']) == 1
        assert len(calls) == (3 if failure_stage == 'result' else 2)
        saved = json.loads((tmp_path / 'history.json').read_text())[0]
        assert saved['providerError'] == public['providerError']
        text = (tmp_path / 'history.json').read_text() + (session.dir / 'events.jsonl').read_text()
        text += engine.provider_error_journal.path.read_text()
        for value in ('leaked-secret', 'test-secret', 'private.example'):
            assert value not in text
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_api_error_response_uses_only_safe_metadata(tmp_path):
    adapter = FalAdapter('test-secret', transport=httpx.MockTransport(lambda request: httpx.Response(403,
        json={'detail': 'User is locked. Reason: TOP_UP. token=hidden', 'request_id': REQUEST_ID})))
    engine = Engine(adapter, tmp_path)
    app = create_app(engine)
    app.state.engine = engine  # handler-only fixture; no backend/sensor startup
    @app.get('/test-provider-failure')
    async def fail():
        await adapter.result('owner/model', REQUEST_ID)
    app.router.routes.insert(0, app.router.routes.pop())  # before the static catch-all mount
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            response = await client.get('/test-provider-failure')
        assert response.status_code == 403
        assert response.json()['providerError']['requestId'] == REQUEST_ID
        assert 'hidden' not in response.text and 'test-secret' not in response.text
    finally:
        await engine.close()
