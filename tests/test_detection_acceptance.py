"""Replay real cached provider responses on real episode image evidence.

No calls are made by tests. These fixtures exercise the complete parsing and
reference checks, including absent cast and a documented occlusion false negative.
"""
import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image
from backend.adaptive import tracks

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((Path(__file__).parent / 'fixtures/florence_regions.json').read_text())
CAST = [{'name': 'SpongeBob'}, {'name': 'Patrick'}]


def fixture_image(case):
    im = Image.open(ROOT / case['source']).convert('RGB')
    if 'crop_fraction' in case:
        w, h = im.size
        im = im.crop(tuple(int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(case['crop_fraction'])))
    if 'crop_pixels' in case:
        im = im.crop(case['crop_pixels'])
    if 'thumbnail' in case:
        im.thumbnail(tuple(case['thumbnail']))
    buffer = BytesIO()
    im.save(buffer, format='JPEG')
    return buffer.getvalue(), *im.size


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_actual_florence_regions_are_not_forced_to_expected_cast(case):
    jpeg, w, h = fixture_image(case)
    boxes, regions = tracks.parse_regions(case['response'], jpeg, w, h, CAST)
    assert set(boxes) == set(case['expected'])
    assert all('confidence' not in r for r in regions)
    assert all(r['identity_status'] in ('unknown', 'model_observed', 'verified_reference', 'ambiguous_duplicate',
                                       'ambiguous_conflicting_evidence') for r in regions)
    if case['id'] == 'background':
        assert regions and not any(r['identity'] for r in regions)
    if case['id'] == 'patrick-only':
        assert case['response']['results']['bboxes'] == []
        assert regions[0]['source'] == 'local_reference_geometry'


def test_unprompted_identity_rejects_descriptions_nearest_names_and_conflicting_names():
    assert tracks.visual_identity('pink starfish-shaped flower', ['Patrick']) is None
    assert tracks.visual_identity('SpongeBob SquarePants treasure chest', ['SpongeBob']) is None
    assert tracks.visual_identity('yellow sponge on a kitchen table', ['SpongeBob']) is None
    assert tracks.visual_identity('Squidward Spongebob Squarepants cartoon character', ['SpongeBob']) is None
    assert tracks.visual_identity('SpongeBob SquarePants Patrick Star holding box', ['Patrick', 'SpongeBob']) == 'Patrick'


def test_overlap_outside_and_past_only_expiry():
    boxes = {'SpongeBob': [.1,.1,.6,.9], 'Patrick': [.4,.1,.9,.9]}
    assert tracks.target_at(boxes, .5, .5) is None
    assert tracks.target_at(boxes, .2, .5) == 'SpongeBob'
    assert tracks.target_at(boxes, -.01, .5) is None
    track = [dict(t=1, boxes=boxes, clip_id='a', valid_until=1.8),
             dict(t=1.4, boxes={}, clip_id='a', status='shot_boundary', valid_until=2)]
    assert tracks.boxes_at(track, .9) == {}
    assert tracks.boxes_at(track, 1.2, 'b') == {}
    assert tracks.boxes_at(track, 1.2, 'a') == boxes
    assert tracks.boxes_at(track, 1.5, 'a') == {}
    assert tracks.boxes_at([track[0]], 1.9) == {}
    assert tracks.final_boxes(track) == {}


@pytest.mark.asyncio
async def test_local_person_detection_does_not_manufacture_character_identity(monkeypatch):
    monkeypatch.setattr(tracks, 'detect_people', lambda *a: [(0, [[.1,.1,.4,.8]])])
    result = await tracks.track_clip('unused', 'unused', ['Patrick'], seeds={'Patrick': [.1,.1,.4,.8]})
    assert result[0]['boxes'] == {}
    assert result[0]['regions'][0]['identity_status'] == 'unknown'


@pytest.mark.asyncio
async def test_real_frame_transport_has_no_cast_prompt_and_keeps_provenance(tmp_path, monkeypatch):
    case = CASES[1]
    jpeg, w, h = fixture_image(case)
    monkeypatch.setattr(tracks, 'sample_scene_frames', lambda *a: [dict(t=1.25, jpeg=jpeg, width=w, height=h, shot=2, cut=False)])
    calls = []
    def handler(request):
        body = json.loads(request.content)
        assert set(body) == {'image_url'}
        calls.append(body)
        return httpx.Response(200, json=case['response'])
    video = tmp_path / 'video'
    video.write_bytes(b'cached fixture input')
    result = await tracks.detect_characters(video, CAST, 'test', directory=tmp_path,
        transport=httpx.MockTransport(handler), clip_id='clip-a', session_id='session-a')
    second = await tracks.detect_characters(video, CAST, 'test', directory=tmp_path,
        transport=httpx.MockTransport(handler), clip_id='clip-b', session_id='session-b')
    assert len(calls) == 1
    assert set(result[0]['boxes']) == {'SpongeBob', 'Patrick'}
    assert result[0]['coordinate_space'] == 'video-normalized' and result[0]['media_timestamp_s'] == 1.25
    assert result[0]['clip_id'] == 'clip-a' and second[0]['clip_id'] == 'clip-b'
    assert second[0]['cache_hit'] and not result[0]['provider_confidence_available']


def test_visual_names_require_independent_foreground_geometry_on_real_background():
    # These false vision boxes were actually returned by the low-reasoning
    # seven-image trial; they describe the preceding frame, not this background.
    case=next(c for c in CASES if c['id']=='background')
    jpeg,w,h=fixture_image(case)
    _,regions=tracks.parse_regions(case['response'],jpeg,w,h,CAST)
    record={'regions':regions}
    assert not tracks.corroborates_body(record,'SpongeBob',[.171,.483,.495,1])
    assert not tracks.corroborates_body(record,'Patrick',[.553,.162,.812,1])


def test_foreground_guard_uses_generic_region_geometry_not_its_unreliable_identity():
    record={'regions':[dict(box=[.1,.1,.9,.9],label='SpongeBob SquarePants treasure chest',identity_status='unknown')]}
    assert tracks.corroborates_body(record,'Patrick',[.1,.1,.9,.9])  # generic figure region; never uses its unreliable name
    record['regions'][0]['label']='human face'
    assert tracks.corroborates_body(record,'Patrick',[.1,.1,.9,.9])
    assert not tracks.corroborates_body(record,'Patrick',[.91,.1,1,.9])


@pytest.mark.asyncio
async def test_identical_decoded_images_share_one_provider_request(tmp_path,monkeypatch):
    case=CASES[1]
    jpeg,w,h=fixture_image(case)
    monkeypatch.setattr(tracks,'sample_scene_frames',lambda *a:[dict(t=i,jpeg=jpeg,width=w,height=h,shot=0,cut=False) for i in range(8)])
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(200,json=case['response'])
    video=tmp_path/'source';video.write_bytes(b'fixture')
    result=await tracks.detect_characters(video,CAST,'test',transport=httpx.MockTransport(handler))
    assert len(calls)==1 and [r['t'] for r in result]==list(range(8))
    assert sum(r['provider_request_shared'] for r in result)==7
    assert all(set(r['boxes'])=={'Patrick','SpongeBob'} for r in result)
