"""Offline policy/request acceptance; no claimed EEG or generated-video accuracy."""
from copy import deepcopy
import asyncio
import time

import pytest

from backend.adaptive import director, eeg_policy, fusion, profile
from backend.clip_settings import build_h3_request
from test_adaptation_acceptance import BASE, NAMES, replay
from test_adaptive_deadline import clock, setup, tick

START, END = 1000., 1003.5


def quality(**overrides):
    return dict(source="muse", live=True, calibrated=True, confidence=.9,
                qualityError="", connectionState="streaming", sampleAgeSeconds=.02,
                **overrides)


def samples(z=1.5):
    return [(START+i*.25, .6, z, False) for i in range(15)]


def request(rows=None, q=None, window=(START, END), *, gaze=True, previous=None):
    analysis = replay(seconds=3.5) if gaze else dict(characters={}, valid_gaze_s=0, comparison_s=0)
    # Exercise the public fusion contract; policy has no dependence on gaze availability.
    analysis["eeg_policy"] = fusion.analyze([], samples() if rows is None else rows, [], NAMES,
        eeg_quality=quality() if q is None else q, eeg_window=window)["eeg_policy"]
    p, _ = profile.update(previous or profile.new_profile(NAMES), analysis)
    hint = profile.decide(p, analysis)
    story = dict(premise=BASE, next_prompt=BASE, scenes=[dict(summary="Found a shell")],
                 characters=[dict(name=n) for n in NAMES],
                 frame_constraints=dict(first_frame=True, end_frame=True))
    plan = director.template(story, hint, 15, NAMES)
    model, payload = build_h3_request(dict(mode="frames", prompt=plan["video_prompt"], duration=15,
        resolution="480P", seed=42, promptExpansionMode="disabled"),
        dict(start="https://fal.media/same-first.png", end="https://fal.media/same-last.png"))
    return p, hint, plan, model, payload


def test_opposite_validated_states_change_exact_payload_with_identical_gaze_and_boundaries():
    high, low, neutral = request(), request(samples(-1.5)), request(samples(0))
    for run, state, cue in [(high, "above_baseline", "quicken"), (low, "below_baseline", "slowly")]:
        p, hint, plan, model, payload = run
        assert hint["focus"] == "Patrick"
        assert hint["eeg_policy"]["state"] == state and hint["eeg_policy"]["applied"]
        assert hint["eeg_policy"]["valid_span_s"] == 1.5
        assert plan["decision"]["eeg_applied"]
        assert plan["decision"]["applied_actions"] == ["focus_character", hint["eeg_policy"]["action"]]
        assert payload["prompt"].startswith(BASE)
        assert "PRIMARY SHOT: Patrick" in payload["prompt"]
        assert "EEG DELIVERY TRIAL" in payload["prompt"] and cue in payload["prompt"]
        assert "required first and last frame compositions" in payload["prompt"]
        assert p["characters"] == neutral[0]["characters"]
        assert p["pacing"] == p["dialogue"] == 0
        assert model == neutral[3]
        assert {k:v for k,v in payload.items() if k != "prompt"} == {
            k:v for k,v in neutral[4].items() if k != "prompt"}
    assert high[4]["prompt"] != low[4]["prompt"] != neutral[4]["prompt"]
    assert "EEG DELIVERY TRIAL" not in neutral[4]["prompt"]


@pytest.mark.parametrize("failure", ["missing", "quality_missing", "window_missing", "uncalibrated",
    "stale", "raw_stale", "poor_quality", "artifact", "clipped", "nonfinite", "future_only",
    "prior_clip", "short", "latest_artifact", "latest_clipped", "transient", "mixed", "gap", "duplicates"])
def test_unusable_or_neutral_eeg_leaves_exact_gaze_request_unchanged(failure):
    rows, q, window = samples(), quality(), (START, END)
    if failure == "missing": rows=[]
    if failure == "quality_missing": q={}
    if failure == "window_missing": window=None
    if failure == "uncalibrated": q["calibrated"]=False
    if failure == "stale": q.update(live=False, connectionState="stale")
    if failure == "raw_stale": q["sampleAgeSeconds"]=3.1
    if failure == "poor_quality": q["qualityError"]="clipped sensor channel"
    if failure == "artifact": rows=[(*row[:3], True) for row in rows]
    if failure == "clipped": rows=samples(5)
    if failure == "nonfinite": rows=samples(float("nan"))
    if failure == "future_only": rows=[(t+4,v,z,a) for t,v,z,a in rows]
    if failure == "prior_clip": rows=[(t-10,v,z,a) for t,v,z,a in rows]
    if failure == "short": rows=rows[-4:]
    if failure == "latest_artifact": rows[-1]=(*rows[-1][:3], True)
    if failure == "latest_clipped": rows[-1]=(END, .6, -5, False)
    if failure == "transient": rows=samples(0); rows[-1]=(END,.6,2,False)
    if failure == "mixed": rows=[(t,v,1.5 if i%2 else -1.5,a) for i,(t,v,z,a) in enumerate(rows)]
    if failure == "gap": rows=[row for row in rows if row[0]<1002.5 or row[0]>1003]
    if failure == "duplicates": rows=[rows[-1]]*100
    actual = request(rows, q, window)
    baseline = request([])
    assert not actual[1]["eeg_policy"]["applied"]
    assert actual[4] == baseline[4], failure


def test_eeg_can_change_delivery_without_a_named_character_or_gaze_evidence():
    p, hint, plan, _, payload = request(gaze=False)
    assert hint["focus"] is None
    assert plan["decision"]["action"] == "faster_pacing"
    assert plan["decision"]["eeg_primary"] and plan["decision"]["eeg_applied"]
    assert plan["decision"]["applied_actions"] == ["faster_pacing"]
    assert payload["prompt"].count("EEG DELIVERY TRIAL") == 1
    assert "PRIMARY SHOT" not in payload["prompt"]
    assert p["characters"] == profile.new_profile(NAMES)["characters"]


def test_current_gaze_readability_pacing_suppresses_conflicting_eeg_even_with_focus():
    _, hint, _, _, _ = request()
    hint["pacing"]="slower"
    plan=director.template(dict(premise=BASE),hint,15,NAMES)
    assert not plan["decision"]["eeg_applied"]
    assert plan["decision"]["eeg_policy"]["suppressed_by"]=="gaze_readability_pacing"
    assert "EEG DELIVERY TRIAL" not in plan["video_prompt"]


def test_retention_requires_current_evidence_and_unknown_resets_it():
    high=request()[0]
    retained=request(samples(.7),previous=high)
    assert retained[1]["eeg_policy"]["state"]=="above_baseline"
    assert request(samples(.7))[1]["eeg_policy"]["state"]=="near_baseline"
    unknown=request([],previous=high)
    assert unknown[1]["eeg_policy"]["state"]=="unknown"
    assert request(samples(.7),previous=unknown[0])[1]["eeg_policy"]["state"]=="near_baseline"
    assert request(samples(-1.5),previous=high)[1]["eeg_policy"]["state"]=="below_baseline"


def test_mindmonitor_requires_raw_artifact_coverage_and_fresh_paired_bands():
    q=quality(); q.update(source="mindmonitor", artifactCoverage="raw clipping/movement and contacts",
                         contactAgeSeconds=.1,alphaAgeSeconds=.1,betaAgeSeconds=.1)
    assert request(q=q)[1]["eeg_policy"]["applied"]
    for key,value in [("artifactCoverage","contacts only; enable raw EEG OSC for artifact checks"),
                      ("betaAgeSeconds",3.5),("confidence",.25)]:
        bad=deepcopy(q);bad[key]=value
        assert request(q=bad)[4]==request([])[4]


def test_muse_feature_windows_before_playback_never_count_toward_duration():
    rows=[(START+i*.25,.6,1.5,False) for i in range(10)]
    assert not eeg_policy.observe(rows, quality(), (START, START+2.25))["eligible"]


def test_prompt_limit_never_silently_discards_eeg_cue():
    hint=request(gaze=False)[1]
    with pytest.raises(ValueError,match="8,000"):
        director.template(dict(premise="x"*7999),hint,15,NAMES)


@pytest.mark.asyncio
async def test_frozen_eeg_reaches_exact_mocked_request_at_playback_deadline(tmp_path, monkeypatch, clock):
    """Full existing acquisition->freeze->fusion->policy->provider request path."""
    payloads = []
    for z in (1.5, -1.5):
        s, clip, engine, _ = setup(tmp_path/str(z), monkeypatch, clock,eeg_run_mode="baseline")
        s.names=NAMES; s.target_names=NAMES; s.profile=profile.new_profile(NAMES)
        s.story.update(premise=BASE, characters=[dict(name=n) for n in NAMES])
        boxes={"SpongeBob":[.05,.15,.4,.9], "Patrick":[.6,.15,.95,.9]}
        clip["track"]=[dict(t=i/8,valid_until=(i+1)/8,boxes=boxes,
            session_id=s.id,clip_id="source") for i in range(41)]
        s.eeg.source="muse"
        s.eeg.series.extend((s.started+i*.25,.6,z,False) for i in range(21))
        monkeypatch.setattr(s.eeg, "status", lambda: quality())
        async def generate(job, images):
            job["seed"]=42
            _, payload=build_h3_request(job,dict(start="https://fal.media/same-first.png",
                                               end="https://fal.media/same-last.png"))
            payloads.append(payload)
            job.update(status="completed",apiStartedAt=time.time()*1000)
        monkeypatch.setattr(engine,"run_job",generate)
        for i in range(101):
            clock[0]=s.started+i/20
            s.gaze.add(dict(t=clock[0],x=75,y=50,valid=True,face=True,confidence=.9,yaw=0))
            tick(s,clock,i/20)
        await asyncio.wait_for(s.task,1)
        job=engine.jobs[s.clips[1]["jobId"]]
        decision=job["engagementDecision"]
        assert decision["focus"]=="Patrick" and decision["eeg_applied"]
        assert decision["eeg_policy"]["state"]==("above_baseline" if z>0 else "below_baseline")
        assert decision["observationWindow"]["end"]-decision["observationWindow"]["start"]==5.0
        assert job["observationToSubmitMs"]==5000
        assert clip["detectionStatus"]=="processing" and clip["status"]=="playing"
        assert "endedAt" not in clip
        assert "PRIMARY SHOT: Patrick" in payloads[-1]["prompt"]
        assert "EEG DELIVERY TRIAL" in payloads[-1]["prompt"]
        await engine.close()
    assert payloads[0]["prompt"]!=payloads[1]["prompt"]
    assert {k:v for k,v in payloads[0].items() if k!="prompt"}=={
        k:v for k,v in payloads[1].items() if k!="prompt"}
