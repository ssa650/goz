"""Opening-window coverage and policy compatibility; recorded evidence/free mocks."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time

import httpx
import pytest

from backend.adaptive import fusion, identity, profile, tracks
from backend.clip_settings import build_h3_request
from backend.frames import ffmpeg
from test_adaptive_deadline import clock, setup, tick
from test_adaptation_acceptance import BASE, BOXES, NAMES
from test_detection_acceptance import CASES, fixture_image


def test_adaptive_sampler_prioritizes_eight_real_opening_frames_and_stops_decode(monkeypatch):
    reads = []
    pixels = bytes([0,0,0,255,255,255]*2)
    def reader(path, **kwargs):
        yield dict(size=(2,2),fps=20,duration=30)
        for i in range(600):
            reads.append(i)
            yield pixels
    monkeypatch.setattr(tracks.imageio_ffmpeg,'read_frames',reader)
    frames=tracks.sample_scene_frames('unused',width=2,observation_seconds=3.5)
    sampled=[f for f in frames if f['jpeg']]
    assert [f['t'] for f in sampled]==[i/2 for i in range(8)]
    assert len(sampled)==8 and len(reads)<=72, 'do not decode the full30s clip for opening evidence'
    assert sampled[-1]['t']<=3.5


def test_general_sampler_retains_existing_whole_clip_cadence(monkeypatch):
    pixels=bytes([0,0,0,255,255,255]*2)
    def reader(path,**kwargs):
        yield dict(size=(2,2),fps=20,duration=30)
        for i in range(600):yield pixels
    monkeypatch.setattr(tracks.imageio_ffmpeg,'read_frames',reader)
    frames=tracks.sample_scene_frames('unused',width=2)
    assert [f['t'] for f in frames]==[i*3.75 for i in range(8)]


async def fixture_track(tmp_path,monkeypatch):
    """Actual decoded frames + real stored region/identity responses, no live calls."""
    case=CASES[1]  # two verified visible characters in the real00-30 reference
    jpeg,w,h=fixture_image(case)
    still=tmp_path/'reference.jpg';still.write_bytes(jpeg)
    video=tmp_path/'reference-15s.mp4'
    await ffmpeg('-loop','1','-i',still,'-t','15','-vf','scale=768:432',
                 '-r','10','-pix_fmt','yuv420p',video)
    # Stored Florence boxes are in the source image's pixel coordinates.
    response=deepcopy(case['response'])
    for b in response['results']['bboxes']:
        for key in ('x','w'):b[key]*=768/w
        for key in ('y','h'):b[key]*=432/h
    recorded=json.loads((Path(__file__).parent/'fixtures/identity_verified.json').read_text())['result']['frames'][1]['detections']
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json=response)
    original_client=httpx.AsyncClient
    monkeypatch.setattr(tracks.httpx,'AsyncClient',lambda **kwargs:original_client(**{**kwargs,'transport':httpx.MockTransport(handler)}))
    async def verify(frames,*args):
        assert len([f for f in frames if f['jpeg']])<=8
        assert max(f['t'] for f in frames)<=3.5
        return {f['t']:deepcopy(recorded) for f in frames if f['jpeg']}, {'status':'recorded_fixture'}
    monkeypatch.setattr(identity,'verify_frames',verify)
    track=await tracks.detect_characters(video,[dict(name=n) for n in NAMES],'test-only',directory=tmp_path,
        clip_id='source',session_id='fixture',observation_seconds=3.5)
    assert [f['t'] for f in track]==[i/2 for i in range(8)]
    assert all(set(f['boxes'])==set(NAMES) for f in track)
    assert all(f['source']=='openai_reference_vision' for f in track)
    assert len(calls)<=8
    return track


@pytest.mark.asyncio
async def test_realistic_3_5s_decoded_identity_and_gaze_change_exact_request(tmp_path,monkeypatch,clock):
    track=await fixture_track(tmp_path,monkeypatch)
    requests=[]
    for focused in NAMES:
        s,clip,engine,_=setup(tmp_path/focused,monkeypatch,clock)
        s.names=NAMES;s.target_names=NAMES;s.profile=profile.new_profile(NAMES)
        s.story.update(premise=BASE,characters=[dict(name=n) for n in NAMES])
        clip['track']=deepcopy(track)
        for frame in clip['track']:frame['session_id']=s.id
        clip['detectionStatus']='processing'  # Other work must never gate submission.
        async def generate(job,images):
            job['seed']=42  # Paired acceptance requests use the same generation randomness.
            _,payload=build_h3_request(job,dict(start='https://fal.media/same-boundary.png'))
            requests.append(payload)
            job.update(status='completed',generationInput=payload,apiStartedAt=time.time()*1000)
        monkeypatch.setattr(engine,'run_job',generate)
        box=clip['track'][0]['boxes'][focused]
        x,y=(box[0]+box[2])*50,(box[1]+box[3])*50
        for i in range(106):
            clock[0]=s.started+i/30
            s.gaze.add(dict(t=clock[0],x=x,y=y,valid=True,face=True,confidence=.9,blink=False,yaw=0))
            tick(s,clock,i/30)
        await s.task
        decision=engine.jobs[s.clips[1]['jobId']]['engagementDecision']
        assert decision['focus']==focused and clip['analysis']['comparison_s']>=3.4
        assert decision['observationWindow']['end']-decision['observationWindow']['start']==3.5
        assert f'PRIMARY SHOT: {focused} receives the main medium close-up' in requests[-1]['prompt']
        assert clip['status']=='playing' and 'endedAt' not in clip
        assert all(f['valid_until']-f['t']<=.8+1e-6 for f in clip['track'])
        await engine.close()
    assert requests[0]['prompt']!=requests[1]['prompt']
    assert {k:v for k,v in requests[0].items() if k!='prompt'}=={k:v for k,v in requests[1].items() if k!='prompt'}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['invalid','stale-gaze','poor-quality','missing-gaze','missing-identity','expired-boxes','crossclip','ambiguous'])
async def test_opening_policy_never_forces_focus_on_unusable_evidence(tmp_path,monkeypatch,clock,failure):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    s.names=NAMES;s.target_names=NAMES;s.profile=profile.new_profile(NAMES)
    s.story.update(premise=BASE,characters=[dict(name=n) for n in NAMES])
    boxes=deepcopy(BOXES)
    if failure=='ambiguous':boxes={n:[.1,.1,.9,.9] for n in NAMES}
    clip['track']=[dict(t=i/2,boxes=deepcopy(boxes),clip_id='source',session_id=s.id,
        valid_until=i/2+.8,status='observed') for i in range(8)]
    if failure=='missing-identity':
        for f in clip['track']:f.update(boxes={},status='unavailable')
    elif failure=='expired-boxes':clip['track']=clip['track'][:1]
    elif failure=='crossclip':
        for f in clip['track']:f['clip_id']='old'
    for i in range(106):
        clock[0]=s.started+i/30
        if failure!='missing-gaze':
            s.gaze.add(dict(t=clock[0],x=75 if failure!='ambiguous' else 50,y=50,
                valid=failure!='invalid',face=True,confidence=.4 if failure=='poor-quality' else .9,
                stale=failure=='stale-gaze',blink=False,yaw=0))
        tick(s,clock,i/30)
    await s.task
    decision=submitted[0][0]['engagementDecision']
    assert decision['focus'] is None and decision['action']=='keep'
    assert s.profile['characters']==profile.new_profile(NAMES)['characters']
    assert 'PRIMARY SHOT' not in submitted[0][0]['prompt']
    assert len(submitted)==1 and s.status=='running'
    await engine.close()


def test_readability_shorter_window_retains_absolute_outside_and_dialogue_requirements():
    original=profile.new_profile(NAMES)
    def decide(valid=3.5,away=.4,confidence=.9,dialogue=1.5,silent=1.5):
        a=dict(valid_gaze_s=valid,gaze_confidence=confidence,comparison_s=0,characters={},look_away_frac=away,
               beats=[dict(dialogue=True,valid_s=dialogue,on_video_s=0),dict(dialogue=False,valid_s=silent,on_video_s=silent)])
        p,_=profile.update(original,a)
        return profile.decide(p,a)
    assert decide()['pacing']=='slower' and decide()['dialogue']=='less'
    assert decide(valid=2.99)['pacing']=='same' and decide(valid=2.99)['dialogue']=='same'
    assert decide(away=.39)['pacing']=='same', '3.5*.39 is less than1.4s actual outside gaze'
    assert decide(confidence=.59)['pacing']=='same' and decide(confidence=.59)['dialogue']=='same'
    assert decide(dialogue=1.49)['dialogue']=='same'
    assert decide(silent=1.49)['dialogue']=='same'


def test_opening_native_cut_invalidates_previous_box_before_next_sample(monkeypatch):
    pixels=bytes([0,0,0,255,255,255]*2)
    changed=bytes(255-v for v in pixels)
    def reader(path,**kwargs):
        yield dict(size=(2,2),fps=20,duration=15)
        for i in range(300):yield pixels if i<14 else changed
    monkeypatch.setattr(tracks.imageio_ffmpeg,'read_frames',reader)
    frames=tracks.sample_scene_frames('unused',width=2,observation_seconds=3.5)
    assert [f['t'] for f in frames if f['cut']]==[.7]
    track=[]
    for i,f in enumerate(frames):
        end=min(f['t']+.8,frames[i+1]['t']) if i+1<len(frames) else f['t']+.8
        track.append(dict(t=f['t'],valid_until=end,boxes=BOXES if f['jpeg'] else {},
                          status='observed' if f['jpeg'] else 'shot_boundary'))
    assert tracks.boxes_at(track,.65)==BOXES
    assert tracks.boxes_at(track,.7)=={} and tracks.boxes_at(track,.9)=={}
    assert tracks.boxes_at(track,1.1)==BOXES


@pytest.mark.asyncio
@pytest.mark.parametrize('completed_frames,eligible',[(3,False),(4,True)])
async def test_partial_opening_coverage_uses_only_completed_fresh_frames(tmp_path,monkeypatch,clock,completed_frames,eligible):
    s,clip,engine,submitted=setup(tmp_path,monkeypatch,clock)
    s.names=NAMES;s.target_names=NAMES;s.profile=profile.new_profile(NAMES)
    s.story.update(premise=BASE,characters=[dict(name=n) for n in NAMES])
    # Each completed .5s sample ends at the next known sample/cut boundary.
    # Pending later frames supply no boxes, visibility or dwell.
    clip['track']=[dict(t=i/2,valid_until=(i+1)/2,boxes=BOXES,clip_id='source',session_id=s.id)
                   for i in range(completed_frames)]
    clip['detectionStatus']='processing'
    for i in range(106):
        clock[0]=s.started+i/30
        s.gaze.add(dict(t=clock[0],x=75,y=50,valid=True,face=True,confidence=.9,yaw=0))
        tick(s,clock,i/30)
    await s.task
    assert clip['analysis']['comparison_s']==pytest.approx(completed_frames/2)
    decision=submitted[0][0]['engagementDecision']
    assert decision['focus']==('Patrick' if eligible else None)
    assert len(submitted)==1 and clip['detectionStatus']=='processing'
    assert decision['observationWindow']['end']-decision['observationWindow']['start']==3.5
    await engine.close()
