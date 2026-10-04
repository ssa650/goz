"""Trace provenance without live sensors, model requests or paid generation."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from backend.adaptive import decision_trace as trace, director
from backend.adaptive.session import AdaptiveSession
from backend.adaptive.sensors import GazeFeed, EegFeed
from backend.engine import Engine
from test_backend import FakeAdapter


def record(base="Patrick opens a box.", hint=None, engine=None):
    story = dict(premise=base, next_prompt=base, characters=[dict(name="Patrick"), dict(name="SpongeBob")], scenes=[])
    plan = director.template(story, hint or {}, 5, ["Patrick", "SpongeBob"])
    clip = dict(id="clip-one", decisionId="decision-one", index=0, createdAt=123,
                decision=plan["decision"], plan=plan, writer="local-rules", basePrompt=base)
    session = SimpleNamespace(id="session-one", clips=[], engine=engine)
    return trace.create(session, clip), plan


@pytest.mark.asyncio
@pytest.mark.parametrize("focus", [None, "Patrick"])
async def test_exact_adapter_prompt_and_reload(tmp_path, focus):
    saved, plan = record(hint=dict(focus=focus, reasons=["Observed comparative dwell" if focus else "Insufficient evidence"]))
    adapter = FakeAdapter()
    engine = Engine(adapter, tmp_path, poll_seconds=.001)
    job = engine.new_job(dict(mode="text", prompt=plan["video_prompt"], duration=5, resolution="480P"),
                         basePrompt=plan["base_prompt"], decisionTrace=saved)
    await engine.run_job(job, {})
    await engine.trace_journal.flush()
    row = engine.trace_journal.read()[0]
    assert len(adapter.submissions) == 1
    assert row["submittedPrompt"] == adapter.submissions[0][1]["prompt"]
    assert row["providerModelId"] == adapter.submissions[0][0]
    assert row["changed"] is bool(focus)
    assert row["submission"] == "confirmed" and row["promptExact"] is True
    assert bool(row["diff"]) is bool(focus)
    assert row["writer"]["returnedOutput"]["video_prompt"] == plan["video_prompt"]
    assert row["writer"]["modelId"] is None
    assert row["evidence"]["gaze"]["gaze_confidence"] is None
    assert row["returnedOutput"]["videoReference"] == f"/api/jobs/{job['id']}/video"
    reloaded = Engine(FakeAdapter(), tmp_path)
    assert reloaded.jobs[job["id"]]["decisionTrace"]["submittedPrompt"] == row["submittedPrompt"]
    assert reloaded.trace_journal.read("session-one") == [row]
    assert reloaded.trace_journal.read("different-session") == []
    await reloaded.close(); await engine.close()


@pytest.mark.asyncio
async def test_cancelled_before_submission_and_failed_attempt_are_distinct(tmp_path):
    adapter = FakeAdapter(); engine = Engine(adapter, tmp_path, poll_seconds=.001)
    saved, plan = record()
    job = engine.new_job(dict(mode="text", prompt=plan["video_prompt"], duration=5, resolution="480P"), decisionTrace=saved)
    job["cancelRequested"] = True
    await engine.run_job(job, {})
    assert not adapter.submissions
    assert saved["submittedPrompt"] is None and saved["submission"] == "not_submitted"
    saved2, _ = record(); saved2["id"] = "decision-two"; saved2["clipId"] = "clip-two"
    adapter.fail_submit = True
    job2 = engine.new_job(dict(mode="text", prompt=plan["video_prompt"], duration=5, resolution="480P"), decisionTrace=saved2)
    await engine.run_job(job2, {})
    assert len(adapter.submissions) == 1
    assert saved2["submittedPrompt"] == adapter.submissions[0][1]["prompt"]
    assert saved2["submission"] == "failed_unconfirmed" and saved2["timing"]["submittedAt"] is None
    assert "error" not in json.dumps(saved2).lower()
    await engine.close()


@pytest.mark.asyncio
async def test_disabled_generation_keeps_inspectable_plan_without_fake_submission(tmp_path):
    engine = Engine(FakeAdapter(), tmp_path)
    def disabled():
        raise ValueError("Generation disabled")
    engine.generation_guard = disabled
    session = AdaptiveSession(engine, GazeFeed(), EegFeed(), tmp_path, 'Patrick opens a box.',
                              [dict(name='Patrick')], 5, '480P', None)
    await session.make_scene({}, None, None, [])
    await engine.trace_journal.flush()
    row = engine.trace_journal.read()[0]
    assert row['submission'] == 'not_submitted' and row['submittedPrompt'] is None
    assert row['generationStatus'] == 'failed' and row['writer']['returnedOutput'] is not None
    assert not engine.adapter.submissions and not engine.jobs
    await engine.close()


def test_redaction_and_public_diff_do_not_leak_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv('FAL_KEY', 'private-key-value')
    base = '<img src=x onerror=alert(1)> private-key-value https://fal.media/frame?token=secret'
    row, _ = record(base)
    job = dict(decisionTrace=row, basePrompt=base, model='model', firstFrame='https://secret?token=x')
    trace.attempted(job, dict(prompt=base+' extra'), 2000)
    trace.refresh(job)
    result = trace.prepared(row)
    serialized = json.dumps(result)
    assert 'private-key-value' not in serialized and 'https://' not in serialized
    assert result['promptExact'] is False and result['basePromptExact'] is False
    assert result['references']['firstFrame'] is None
    assert result['changed'] is True
    assert '<img' in result['diff']  # preserved as plain text; DOM test verifies rendering


def test_journal_concurrency_partial_tail_dedupe_and_all_runs(tmp_path):
    journal = trace.Journal(tmp_path)
    row, _ = record()
    journal.append([row])
    with journal.path.open('ab') as stream:
        stream.write(b'{"truncated":')
    rows = []
    for i in range(12):
        r, _ = record(); r.update(sessionId=f'run-{i}', id=f'decision-{i}')
        rows.append(r)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r: trace.Journal(tmp_path).append([r]), rows))
    row['generationStatus'] = 'completed'
    journal.append([row])
    results = journal.read()
    assert len(results) == 13
    assert journal.read('session-one')[0]['generationStatus'] == 'completed'
    with journal.path.open('ab') as stream:
        stream.write(b'{"unfinished":')
    assert journal.read() == results


def test_frozen_session_evidence_and_timing_reference_measured_inputs():
    saved, plan = record(hint=dict(focus='Patrick', reasons=['Observed shared visibility']))
    previous = dict(id='source-clip', frozenEvidence=dict(frozenAt=100, gaze=[{'secretRawCoordinate':99}],
        track=[{'boxes':{'Patrick':[0,0,1,1]}}], quality=dict(source='muse', calibrated=True, confidence=.8)),
        analysis=dict(valid_gaze_s=3.2, gaze_confidence=.9, comparison_s=2.5,
             characters={'Patrick':dict(dwell_s=1.8, visible_s=3, comparison_dwell_s=1.8, comparison_visible_s=2.5, attention=.6)}))
    session = SimpleNamespace(id='actual-session', clips=[previous])
    clip = dict(id='next-clip', decisionId='decision', index=1, createdAt=101, writer='local-rules',
                decision=plan['decision'], plan=plan, basePrompt=plan['base_prompt'])
    row = trace.create(session, clip, source=(0, 'source-clip'))
    assert row['evidence']['gaze']['targets']['Patrick']['dwell_s'] == 1.8
    assert row['evidence']['eeg']['calibrated'] is True
    assert row['evidence']['eeg']['eligible'] is None
    assert row['references']['sourceSessionId'] == 'actual-session'
    assert row['references']['analysis'] == 'scene1_signals.json#analysis'
    assert row['evidence']['blink']['used'] == 'gaze validity gate'
    assert 'secretRawCoordinate' not in json.dumps(row) and 'boxes' not in json.dumps(row)
    job = dict(decisionTrace=row, basePrompt=plan['base_prompt'], model='model', continuationReadyAt=104)
    trace.attempted(job, {'prompt':plan['video_prompt']}, 102000)
    trace.refresh(job)
    assert row['timing']['freezeToSubmitMs'] == 2000
    assert row['timing']['freezeToReadyMs'] == 4000


@pytest.mark.asyncio
async def test_runtime_provider_key_redacted_in_actual_payload_trace(tmp_path):
    adapter = FakeAdapter(); engine = Engine(adapter, tmp_path, poll_seconds=.001)
    row, plan = record(base='Patrick opens a box. '+adapter.key, engine=engine)
    job = engine.new_job(dict(mode='text', prompt=plan['video_prompt'], duration=5, resolution='480P'),
        basePrompt=plan['base_prompt'], decisionTrace=row)
    await engine.run_job(job, {})
    assert row['promptExact'] is False
    assert adapter.key not in json.dumps(row)
    await engine.trace_journal.flush()
    assert adapter.key not in engine.trace_journal.path.read_text()
    await engine.close()
