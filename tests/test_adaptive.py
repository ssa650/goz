import asyncio
import json
import math
import time
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image

from backend.adaptive import director, fusion, profile as profiles, tracks
from backend.adaptive.sensors import EEG_RATE, EegFeed
from backend.app import create_app
from backend.demo import DemoAdapter
from backend.engine import Engine

A_BOX, B_BOX = [0.05, 0.2, 0.4, 0.95], [0.6, 0.2, 0.95, 0.95]


def png():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def watch(seconds=10, favourite="Bea", rate=30, t0=1000.0):
    """Synthetic viewing: ticks, gaze mostly on the favourite, EEG rising
    ~0.5 s after each look at the favourite."""
    rect = dict(x=100, y=100, w=800, h=450)
    track = [dict(t=i / 4, boxes={"Ana": A_BOX, "Bea": B_BOX}) for i in range(seconds * 4 + 1)]
    ticks = [dict(wall=t0 + i / 10, video_t=i / 10, playing=True, rect=rect) for i in range(seconds * 10)]
    gaze, eeg = [], []
    for i in range(seconds * rate):
        t = i / rate
        on_fav = (t % 3) < 2.2
        box = B_BOX if (on_fav == (favourite == "Bea")) else A_BOX
        nx, ny = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        gaze.append(dict(t=t0 + t, x=rect["x"] + nx * rect["w"], y=rect["y"] + ny * rect["h"], valid=True,
                         blink=False, yaw=0))
    for i in range(int(seconds / 0.25)):
        t = i * 0.25
        on_fav = ((t - 0.5) % 3) < 2.2 and t > 0.5
        eeg.append((t0 + t, 1.0, 1.5 if on_fav else -0.8, False))
    return ticks, track, gaze, eeg


def test_identity_tracking_uses_staging_then_seeds():
    detections = [(0.0, [B_BOX, A_BOX]), (0.25, [[0.62, 0.2, 0.97, 0.95], [0.07, 0.2, 0.42, 0.95]])]
    track = tracks.assign(detections, ["Ana", "Bea"])
    assert track[0]["boxes"]["Ana"] == A_BOX and track[1]["boxes"]["Bea"][0] == 0.62
    swapped = tracks.assign([(0.0, [A_BOX, B_BOX])], ["Ana", "Bea"], seeds={"Ana": B_BOX, "Bea": A_BOX})
    assert swapped[0]["boxes"]["Ana"] == B_BOX
    assert tracks.target_at({"Ana": A_BOX, "Bea": B_BOX}, 0.8, 0.5) == "Bea"
    assert tracks.target_at({"Ana": A_BOX, "Bea": B_BOX}, 0.5, 0.05) is None


def test_mvp_comparative_gaze_makes_next_scene_focus_b():
    ticks, track, gaze, eeg = watch(favourite="Bea")
    timeline = fusion.label(gaze, ticks, track)
    analysis = fusion.analyze(timeline, eeg, track, ["Ana", "Bea"])
    bea, ana = analysis["characters"]["Bea"], analysis["characters"]["Ana"]
    assert not bea["strong"] and not ana["strong"]
    assert bea["response"] == bea["attention"]  # EEG is not preference evidence.
    assert bea["attention"] > 0.6 and bea["eeg_response"] > 0.5 > ana["eeg_response"]
    profile, changes = profiles.update(profiles.new_profile(["Ana", "Bea"]), analysis)
    assert profile["characters"]["Bea"] > profile["characters"]["Ana"]
    assert any(c["key"] == "character:Bea" and not c["strong"] for c in changes)
    decision = profiles.decide(profile, analysis)
    assert decision["focus"] == "Bea" and "Observed attention" in decision["reasons"][0]
    plan = director.template(dict(premise="Ana and Bea open the box.", characters=[dict(name="Ana"), dict(name="Bea")], scenes=[]),
                             decision, 10, ["Ana", "Bea"])
    assert plan["video_prompt"].startswith("Ana and Bea") and "PRIMARY SHOT: Bea" in plan["video_prompt"]


def test_timeline_parsing_marks_speakers_and_tags():
    beats = director.parse_timeline("0-3 SpongeBob: I'm ready! #humor\n3-7 Squidward: Not today.\n"
                                    "7-12 Patrick crashes through the door #action\nnot a line", 
                                    ["SpongeBob", "Squidward", "Patrick"], 10)
    assert [b["speaker"] for b in beats] == ["SpongeBob", "Squidward", None]
    assert beats[0]["tags"] == ["humor"] and beats[2]["characters"] == ["Patrick"] and beats[2]["t1"] == 10


def test_eeg_rise_while_a_character_speaks_remains_a_separate_observation():
    ticks, track, gaze, _ = watch(favourite="Bea")
    t0 = ticks[0]["wall"]
    eeg = [(t0 + i * 0.25, 1.0, 1.4 if 4.3 <= i * 0.25 <= 8.3 else -0.6, False) for i in range(40)]
    plan = dict(beats=[dict(t0=0, t1=4, speaker="Ana", dialogue=True, tags=[]),
                       dict(t0=4, t1=8, speaker="Bea", dialogue=True, tags=[]),
                       dict(t0=8, t1=10, speaker=None, dialogue=False, tags=[])])
    analysis = fusion.analyze(fusion.label(gaze, ticks, track), eeg, track, ["Ana", "Bea"], plan)
    assert analysis["characters"]["Bea"]["speaking_response"] > 0.5 > analysis["characters"]["Ana"]["speaking_response"]
    assert not analysis["characters"]["Bea"]["strong"]
    profile, changes = profiles.update(profiles.new_profile(["Ana", "Bea"]), analysis)
    assert "shared visibility" in next(c["why"] for c in changes if c["key"] == "character:Bea")
    without_eeg, _ = profiles.update(profiles.new_profile(["Ana", "Bea"]), fusion.analyze(fusion.label(gaze, ticks, track), [], track, ["Ana", "Bea"], plan))
    assert profile["characters"] == without_eeg["characters"]


@pytest.mark.asyncio
async def test_fal_character_detection_does_not_force_cast_from_descriptions(tmp_path):
    from backend.frames import ffmpeg
    video = tmp_path/"clip.mp4"
    await ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=10", "-t", "1.5", "-pix_fmt", "yuv420p", video)
    asked = []

    def handler(request):
        body = json.loads(request.content)
        asked.append(body)
        assert set(body) == {"image_url"}
        return httpx.Response(200, json={"results": {"bboxes": [{"x": 0, "y": 0, "w": 70, "h": 90, "label": "colored test pattern"}]}})

    chars = [dict(name="Squidward", description="green octopus"), dict(name="SpongeBob", description="yellow sponge")]
    track = await tracks.detect_characters(video, chars, "k", transport=httpx.MockTransport(handler), fps=2)
    assert len(track) == 3 and len(asked) == 3
    assert all(not frame["boxes"] and frame["regions"][0]["identity_status"] == "unknown" for frame in track)


def test_gaze_outside_playback_or_while_blinking_is_ignored():
    ticks, track, gaze, _ = watch()
    late = dict(gaze[0], t=gaze[0]["t"] + 500)
    blink = dict(gaze[0], valid=False, blink=True)
    timeline = fusion.label([late, blink], ticks, track)
    assert len(timeline) == 1 and timeline[0]["target"] is None


def test_eeg_engagement_z_rises_with_beta():
    feed = EegFeed()
    t = 0.0
    for second in range(90):
        beta = 3.0 if second >= 80 else 1.0
        for _ in range(4):
            n = EEG_RATE // 4
            ts = t + np.arange(n) / EEG_RATE
            x = beta * np.sin(2 * math.pi * 20 * ts) + 2 * np.sin(2 * math.pi * 10 * ts) + np.random.normal(0, .3, n)
            feed.push((np.stack([x] * 4, axis=1) * 10).tolist(), list(ts))
            t += n / EEG_RATE
    calm = [z for (tt, _, z, _) in feed.series if 60 < tt < 79]
    engaged = [z for (tt, _, z, _) in feed.series if tt > 83]
    assert np.mean(engaged) > 2 > abs(np.mean(calm))


@pytest.mark.asyncio
async def test_director_local_rule_does_not_call_remote_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    def handler(request):
        raise AssertionError("A bounded gaze decision must not add a remote model call")
    story = dict(premise="P.", characters=[dict(name="Ana"), dict(name="Bea")], scenes=[dict(summary="Opening")],
                 next_prompt="Ana and Bea open the box. Bea says: 'Look!'")
    plan, writer = await director.write_scene(story, profiles.new_profile(["Ana", "Bea"]), dict(focus="Bea"), 10,
                                              transport=httpx.MockTransport(handler))
    assert writer == "local-rules"
    assert plan["base_prompt"] == story["next_prompt"]
    assert plan["video_prompt"].startswith(story["next_prompt"] + "\n\n")
    assert plan["decision"]["focus"] == "Bea"
    assert plan["scene_spec"]["primary_character"] == "Bea"
    assert plan["decisionElapsedMs"] >= 0
    assert "decisionRequestId" not in plan


@pytest.mark.parametrize("decision", [
    dict(action="invent_new_scene", focus=None, reason="Unsupported"),
    dict(action="focus_character", focus="Unknown", reason="Unsupported"),
    dict(action="keep", focus="Bea", reason="Invalid pairing"),
    dict(action="keep", focus=None, reason="ok", video_prompt="Replace the story"),
])
def test_invalid_bounded_decisions_are_rejected(decision):
    with pytest.raises(ValueError):
        director.validate_decision(decision, ["Ana", "Bea"])


def test_adjustments_preserve_the_full_script_and_limit_size():
    base = "Bea opens the box. Ana says: 'Keep it closed.' " * 45
    assert len(base) > 1500
    for action in director.ACTIONS:
        decision = director.validate_decision(dict(action=action, focus="Bea" if action == "focus_character" else None,
                                                   reason="Measured response"), ["Ana", "Bea"])
        prompt = director.adjust_prompt(base, decision)
        assert prompt.startswith(base) and len(prompt) <= 8000
        if action == "keep":
            assert director.adjust_prompt("x" * 8000, decision) == "x" * 8000
        else:
            with pytest.raises(ValueError, match="No adaptation was silently dropped"):
                director.adjust_prompt("x" * 8000, decision)
    assert director.adjust_prompt(base, director.local_decision({}, ["Ana", "Bea"])) == base


@pytest.mark.asyncio
async def test_end_to_end_demo_loop_adapts_to_simulated_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("GOZ_GAZE", "sim")
    monkeypatch.setenv("GOZ_EEG", "sim")
    monkeypatch.setenv("GOZ_SIM_FAVORITE", "1")
    monkeypatch.setenv("GOZ_SIM_BIAS", "0.95")
    monkeypatch.setenv("GOZ_SIM_SEED", "acceptance")
    engine = Engine(DemoAdapter(tmp_path/"demo"), tmp_path, poll_seconds=.01)
    app = create_app(engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
            r = await c.post("/api/adaptive/sessions", files=[("start", ("open.png", png(), "image/png"))],
                             data=dict(premise="Two explorers in a cave.", duration="5", resolution="480P",
                                       characters=json.dumps([dict(name="Ana"), dict(name="Bea")])))
            assert r.status_code == 202, r.text
            state = None
            for _ in range(300):
                state = (await c.get("/api/adaptive/state")).json()
                if state["session"]["clips"] and state["session"]["clips"][0]["status"] == "ready":
                    break
                await asyncio.sleep(.05)
            assert state["session"]["clips"][0]["track"][0]["boxes"].keys() == {"Ana", "Bea"}
            rect = dict(x=0, y=0, w=640, h=360)
            start = time.time()
            for i in range(60):
                video_t = time.time() - start
                await c.post("/api/adaptive/tick", json=dict(session_id=r.json()["id"], clip=0, video_t=video_t, playing=True, rect=rect,
                                                             wall=time.time() * 1000))
                await asyncio.sleep(.085)
                if len((await c.get("/api/adaptive/state")).json()["session"]["clips"]) > 1:
                    break
            await c.post("/api/adaptive/ended", json=dict(session_id=r.json()["id"], clip=0))
            for _ in range(300):
                state = (await c.get("/api/adaptive/state")).json()
                clips = state["session"]["clips"]
                if len(clips) > 1 and clips[1]["status"] == "ready":
                    break
                await asyncio.sleep(.05)
            s = state["session"]
            assert s["clips"][0]["analysis"]["characters"]["Bea"]["attention"] > s["clips"][0]["analysis"]["characters"]["Ana"]["attention"]
            assert s["profile"]["characters"]["Bea"] > s["profile"]["characters"]["Ana"]
            assert s["clips"][1]["decision"]["focus"] == "Bea", s["clips"][1]["decision"]
            assert s["clips"][1]["writer"] == "template"
            assert state["eeg"]["source"] == "sim" and state["gaze"]["source"] == "sim"
            logged = list((tmp_path/"adaptive"/s["id"]).iterdir())
            assert any(p.name == "scene1_signals.json" for p in logged)


@pytest.mark.asyncio
async def test_predefined_opening_clip_is_scene_one(tmp_path, monkeypatch):
    from backend.frames import ffmpeg
    monkeypatch.setenv("GOZ_GAZE", "sim")
    monkeypatch.setenv("GOZ_EEG", "sim")
    monkeypatch.setenv("GOZ_SIM_BIAS", "0.95")
    monkeypatch.setenv("GOZ_SIM_SEED", "acceptance")
    clip = tmp_path/"episode.mp4"
    await ffmpeg("-f", "lavfi", "-i", "color=c=yellow:s=320x180:r=24", "-t", "4", "-pix_fmt", "yuv420p", clip)
    engine = Engine(DemoAdapter(tmp_path/"demo"), tmp_path, poll_seconds=.01)
    app = create_app(engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as c:
            r = await c.post("/api/adaptive/sessions", files=[("opening", ("episode.mp4", clip.read_bytes(), "video/mp4"))],
                             data=dict(premise="A day at the Krusty Krab.", duration="5", resolution="480P",
                                       timeline="0-2 SpongeBob: Order up! #humor\n2-4 Squidward: Ugh.",
                                       characters=json.dumps([dict(name="SpongeBob", description="yellow sponge"),
                                                              dict(name="Squidward", description="green octopus")])))
            assert r.status_code == 202, r.text
            for _ in range(200):
                s = (await c.get("/api/adaptive/state")).json()["session"]
                if s["clips"] and s["clips"][0]["status"] == "ready":
                    break
                await asyncio.sleep(.05)
            first = s["clips"][0]
            assert first["writer"] == "predefined" and abs(first["duration"] - 4) < 0.2
            assert [b["speaker"] for b in first["plan"]["beats"]] == ["SpongeBob", "Squidward"]
            assert (await c.get(first["url"])).status_code == 200
            assert not engine.jobs, "the opening clip must not be generated"
            rect, start = dict(x=0, y=0, w=640, h=360), time.time()
            while time.time() - start < 3.8:
                await c.post("/api/adaptive/tick", json=dict(session_id=r.json()["id"], clip=0, video_t=time.time() - start, playing=True,
                                                             rect=rect, wall=time.time() * 1000))
                await asyncio.sleep(.08)
            for _ in range(300):
                s = (await c.get("/api/adaptive/state")).json()["session"]
                if len(s["clips"]) > 1 and s["clips"][1]["status"] == "ready":
                    break
                await asyncio.sleep(.05)
            assert s["clips"][1]["decision"]["focus"] == "Squidward", s["clips"][1]["decision"]
            assert len(engine.jobs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['text', 'initial', 'both', 'mixed'])
async def test_adaptive_decision_preserves_ordered_bundle_payloads(tmp_path, monkeypatch, mode):
    from test_backend import harness
    from test_bundle_sequence import WireAdapter, make_clips
    from backend.adaptive.session import AdaptiveSession
    from backend.adaptive.sensors import GazeFeed
    from pathlib import Path
    from copy import deepcopy

    async with harness(tmp_path, WireAdapter()) as (engine, adapter, client):
        bundles = await make_clips(client, mode)
        originals = deepcopy(bundles)
        gaze, eeg = GazeFeed(), EegFeed()
        session = AdaptiveSession(engine, gaze, eeg, tmp_path, 'Existing story.',
                                  [dict(name='Ana'), dict(name='Bea')], 10, '768P', None, clip_bundles=bundles)
        # The session must freeze complete bundles, not refer to editable objects.
        bundles[1]['prompt'] = 'An unrelated later edit'
        bundles[1]['seed'] = 99999
        async def media_path(job): return Path('fake.mp4')
        async def track(*args): return []
        engine.media_path = media_path
        session.track = track
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
        for i in range(4):
            if session.clips:
                # This payload-order unit test consumes each clip before asking
                # for another; a separate queue test covers admission rejection.
                session.clips[-1]['status'] = 'watched'
            await session.make_scene({}, None, None, [])
            assert session.status == 'running', session.error
            clip = session.clips[i]
            job = engine.jobs[clip['jobId']]
            model, payload = adapter.submissions[i]
            original = originals[i]
            assert clip['bundle'] == original and job['clipId'] == original['id'] and job['sceneIndex'] == i
            assert payload['prompt'].startswith(original['prompt']) and job['basePrompt'] == original['prompt']
            assert payload['seed'] == original['seed'] and payload['duration'] == original['duration']
            assert payload['resolution'] == original['resolution']
            assert payload['prompt_expansion_mode'] == original['promptExpansionMode']
            assert payload.get('image_url') == (engine.reference(original['firstFrame'])['providerUrl'] if original['firstFrame'] else None)
            assert payload.get('end_image_url') == (engine.reference(original['endFrame'])['providerUrl'] if original['endFrame'] else None)
            if i == 0:
                assert payload['prompt'] == original['prompt'] and job['decisionModel'] == 'local-rules'
            else:
                assert job['decisionModel'] == 'local-rules' and job.get('decisionRequestId') is None
                assert job['engagementDecision']['action'] == 'keep'
        assert [c['bundle']['id'] for c in session.clips] == [c['id'] for c in originals]
        assert [c['prompt'] for c in originals] == [f'Scene {i+1}' for i in range(4)]
