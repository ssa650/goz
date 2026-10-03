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


def test_mvp_strong_response_to_b_makes_next_scene_focus_b():
    ticks, track, gaze, eeg = watch(favourite="Bea")
    timeline = fusion.label(gaze, ticks, track)
    analysis = fusion.analyze(timeline, eeg, track, ["Ana", "Bea"])
    bea, ana = analysis["characters"]["Bea"], analysis["characters"]["Ana"]
    assert bea["strong"] and not ana["strong"]
    assert bea["attention"] > 0.6 and bea["eeg_response"] > 0.5 > ana["eeg_response"]
    profile, changes = profiles.update(profiles.new_profile(["Ana", "Bea"]), analysis)
    assert profile["characters"]["Bea"] > profile["characters"]["Ana"]
    assert any(c["key"] == "character:Bea" and c["strong"] for c in changes)
    decision = profiles.decide(profile, analysis)
    assert decision["focus"] == "Bea" and "strong" in decision["reasons"][0]
    plan = director.template(dict(premise="P.", characters=[dict(name="Ana"), dict(name="Bea")], scenes=[]),
                             decision, 10, ["Ana", "Bea"])
    assert "pushes in on Bea" in plan["video_prompt"]


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
async def test_director_openai_request_and_validation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        content = json.dumps(dict(scene_title="Bea's move", summary="Bea acts.", video_prompt="Bea steps forward.",
                                  change_note="Focus on Bea.", beats=[dict(t0=0, t1=12, characters=["Bea", "Zed"],
                                                                          tags=["action", "nope"], dialogue=True)]))
        return httpx.Response(200, json=dict(choices=[dict(message=dict(content=content))]))

    story = dict(premise="P.", characters=[dict(name="Ana"), dict(name="Bea")], scenes=[])
    plan, writer = await director.write_scene(story, profiles.new_profile(["Ana", "Bea"]), dict(focus="Bea"), 10,
                                              transport=httpx.MockTransport(handler))
    assert writer.startswith("openai:") and seen["response_format"] == {"type": "json_object"}
    assert plan["beats"][0] == dict(t0=0.0, t1=10.0, description="", characters=["Bea"], dialogue=True, tags=["action"])


@pytest.mark.asyncio
async def test_end_to_end_demo_loop_adapts_to_simulated_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("GOZ_GAZE", "sim")
    monkeypatch.setenv("GOZ_EEG", "sim")
    monkeypatch.setenv("GOZ_SIM_FAVORITE", "1")
    monkeypatch.setenv("GOZ_SIM_BIAS", "0.95")
    import backend.adaptive.session as session_module
    monkeypatch.setattr(session_module, "ANALYZE_AT", 0.9)
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
                await c.post("/api/adaptive/tick", json=dict(clip=0, video_t=video_t, playing=True, rect=rect,
                                                             wall=time.time() * 1000))
                await asyncio.sleep(.085)
                if len((await c.get("/api/adaptive/state")).json()["session"]["clips"]) > 1:
                    break
            await c.post("/api/adaptive/ended", json=dict(clip=0))
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
