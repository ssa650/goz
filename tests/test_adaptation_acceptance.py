"""Deterministic acceptance replay, not webcam or generated-video evidence."""
import copy

import httpx
import pytest

from backend.adaptive import director, fusion, profile
from backend.clip_settings import build_h3_request

NAMES = ["SpongeBob", "Patrick"]
BOXES = {"SpongeBob": [.05, .15, .40, .9], "Patrick": [.60, .15, .95, .9]}
BASE = "SpongeBob and Patrick examine the same small seashell on a rock. SpongeBob points at it and Patrick smiles. Keep a wide two-shot. They remain beside the rock at the end."


def replay(focus="Patrick", seconds=6, confidence=1, visible=None):
    rect = dict(x=10, y=20, w=800, h=450)
    visible = NAMES if visible is None else visible
    track = [dict(t=i/10, boxes={n: BOXES[n] for n in visible}) for i in range(int(seconds*10)+1)]
    ticks, gaze = [], []
    for i in range(int(seconds*30)+1):
        t = i/30
        name = focus if i % 30 < 27 else next(n for n in NAMES if n != focus)
        box = BOXES[name]
        ticks.append(dict(wall=1000+t, video_t=t, playing=True, rect=rect))
        gaze.append(dict(t=1000+t, x=rect['x']+(box[0]+box[2])/2*rect['w'],
                         y=rect['y']+(box[1]+box[3])/2*rect['h'], valid=True, face=True,
                         blink=False, yaw=0, confidence=confidence))
    return fusion.analyze(fusion.label(gaze, ticks, track), [], track, NAMES)


def paired_request(focus):
    analysis = replay(focus)
    learned, changes = profile.update(profile.new_profile(NAMES), analysis)
    decision = profile.decide(learned, analysis)
    story = dict(premise=BASE, next_prompt=BASE, scenes=[dict(summary="The two friends find a seashell")],
                 characters=[dict(name=n) for n in NAMES], frame_constraints=dict(first_frame=True, end_frame=True))
    plan = director.template(story, decision, 15, NAMES)
    model, payload = build_h3_request(dict(mode="frames", prompt=plan['video_prompt'], duration=15,
                                            resolution="480P", seed=42, promptExpansionMode="disabled"),
                                      dict(start="https://fal.media/same-first.png", end="https://fal.media/same-last.png"))
    return dict(analysis=analysis, decision=decision, plan=plan, model=model, payload=payload)


def test_patrick_and_spongebob_replay_change_exact_provider_payload_only_in_shot():
    patrick, spongebob = paired_request("Patrick"), paired_request("SpongeBob")
    for run, name in [(patrick, "Patrick"), (spongebob, "SpongeBob")]:
        assert run['decision']['focus'] == name
        assert run['decision']['evidence']['comparison_s'] >= 5.5
        assert run['plan']['scene_spec']['primary_character'] == name
        assert f"PRIMARY SHOT: {name} receives the main medium close-up" in run['payload']['prompt']
        assert run['payload']['prompt'].startswith(BASE)
        assert "overrides competing camera/framing directions" in run['payload']['prompt']
        assert "boundary frames" in run['payload']['prompt']
    assert patrick['payload']['prompt'] != spongebob['payload']['prompt']
    assert {k:v for k,v in patrick['payload'].items() if k != 'prompt'} == {k:v for k,v in spongebob['payload'].items() if k != 'prompt'}


@pytest.mark.parametrize('kwargs', [dict(seconds=.5), dict(confidence=.4), dict(visible=['Patrick']), dict(visible=[])])
def test_insufficient_low_quality_or_single_visible_character_does_not_learn(kwargs):
    original = profile.new_profile(NAMES)
    analysis = replay(**kwargs)
    learned, _ = profile.update(original, analysis)
    assert learned['characters'] == original['characters']
    assert profile.decide(learned, analysis)['focus'] is None


def test_absent_target_and_missing_evidence_never_reuse_old_preference():
    learned, _ = profile.update(profile.new_profile(NAMES), replay())
    missing = dict(characters={}, gaze_confidence=0, valid_gaze_s=0, comparison_s=0)
    assert profile.decide(learned, missing)['focus'] is None
    single = replay(visible=['SpongeBob'])
    assert profile.decide(learned, single)['focus'] is None
    decision = profile.decide(learned, replay())
    plan = director.template(dict(premise='SpongeBob walks home alone.'), decision, 15, NAMES)
    assert plan['decision']['action'] == 'keep'
    assert plan['video_prompt'] == 'SpongeBob walks home alone.'


def test_hysteresis_requires_more_evidence_to_switch_than_retain():
    e = dict(sufficient=True, characters={
        'Patrick': dict(attention=.65, dwell_s=3.9),
        'SpongeBob': dict(attention=.35, dwell_s=2.1)})
    assert profile.focus_from(e, None) == 'Patrick'
    assert profile.focus_from(e, 'Patrick') == 'Patrick'
    assert profile.focus_from(e, 'SpongeBob') is None
    e['characters']['Patrick']['attention'] = .8
    e['characters']['SpongeBob']['attention'] = .2
    assert profile.focus_from(e, 'SpongeBob') == 'Patrick'


def test_eeg_extremes_do_not_create_character_pacing_or_dialogue_preferences():
    analysis = replay(visible=[])
    analysis.update(eeg_mean_z=-100, eeg_confidence=1, blink_rate_per_min=60, look_away_frac=None)
    original = profile.new_profile(NAMES)
    learned, changes = profile.update(original, analysis)
    assert learned['characters'] == original['characters']
    assert learned['dialogue'] == learned['pacing'] == 0
    assert profile.decide(learned, analysis)['focus'] is None


def test_dialogue_and_pacing_require_fresh_valid_gaze_comparisons():
    analysis = dict(valid_gaze_s=6, gaze_confidence=1, comparison_s=0, characters={}, look_away_frac=.5,
                    beats=[dict(dialogue=True, valid_s=3, on_video_s=.5), dict(dialogue=False, valid_s=3, on_video_s=3)])
    learned, _ = profile.update(profile.new_profile(NAMES), analysis)
    decision = profile.decide(learned, analysis)
    assert decision['dialogue'] == 'less' and decision['pacing'] == 'slower'
    plan = director.template(dict(premise=BASE), decision, 15, NAMES)
    assert plan['decision']['action'] == 'less_dialogue'
    assert 'shortest existing line' in plan['video_prompt']
    stale = dict(analysis, valid_gaze_s=0, gaze_confidence=0)
    assert profile.decide(learned, stale)['dialogue'] == 'same'
    assert profile.decide(learned, stale)['pacing'] == 'same'


@pytest.mark.asyncio
async def test_remote_model_is_not_in_bounded_adaptation_critical_path(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-unit-test-key')
    def forbidden(request):
        pytest.fail('local policy made a remote model call')
    run = paired_request('Patrick')
    story = dict(premise=BASE, next_prompt=BASE, characters=[dict(name=n) for n in NAMES], scenes=[dict(summary='start')])
    plan, writer = await director.write_scene(story, {}, run['decision'], 15, transport=httpx.MockTransport(forbidden))
    assert writer == 'local-rules'
    assert plan['decisionElapsedMs'] >= 0 and plan['promptConstructionMs'] >= 0
