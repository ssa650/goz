import json
import httpx
import pytest
from backend.adaptive import identity


def result(names=('Patrick',)):
    return {'frames':[{'frame_index':0,'detections':[{'identity':n,'box':[.1,.1,.4,.9],
        'evidence':'recognizable face and body match the reference'} for n in names]}]}


@pytest.mark.parametrize('bad', [
    {'frames':[]},
    {'frames':[{'frame_index':0,'detections':[{'identity':'invented','box':[.1,.1,.3,.5],'evidence':'x'}]}]},
    {'frames':[{'frame_index':0,'detections':[{'identity':'Patrick','box':[-1,.1,.3,.5],'evidence':'x'}]}]},
    {'frames':[{'frame_index':0,'detections':[{'identity':'Patrick','box':[.5,.1,.3,.5],'evidence':'x'}]}]},
    {'frames':[{'frame_index':0,'detections':[]},{'frame_index':0,'detections':[]}]},
])
def test_identity_rejects_malformed_incompatible_or_missing_frame_output(bad):
    with pytest.raises(ValueError):identity.validate(bad,1,['Patrick','SpongeBob'])


def test_empty_detected_cast_is_valid():
    assert identity.validate({'frames':[{'frame_index':0,'detections':[]}]},1,['Patrick']) == {0:[]}


@pytest.mark.asyncio
async def test_single_batched_request_caches_exact_frame_evidence(tmp_path,monkeypatch):
    from test_detection_acceptance import CASES,fixture_image
    jpeg,_,_=fixture_image(CASES[0])
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    monkeypatch.setenv('GOZ_IDENTITY_MODEL','configured-model')
    calls=[]
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        assert body['model']=='configured-model' and not body['store']
        assert body['text']['format']['type']=='json_schema'
        content=body['input'][0]['content']
        assert len([c for c in content if c['type']=='input_image'])==3 #2references+1scene
        assert 'absent' in content[0]['text'] and 'unknown' in content[0]['text']
        return httpx.Response(200,json={'id':'response-test','output':[{'content':[{'type':'output_text','text':json.dumps(result())}]}]})
    frames=[dict(t=0,jpeg=jpeg)]
    a,provenance=await identity.verify_frames(frames,['Patrick','SpongeBob'],tmp_path,httpx.MockTransport(handler))
    b,cached=await identity.verify_frames(frames,['Patrick','SpongeBob'],tmp_path,httpx.MockTransport(handler))
    assert a==b and len(calls)==1 and cached['cache_hit']
    assert not provenance['confidence_available']
    assert 'test-only' not in next((tmp_path/'detections').glob('*.json')).read_text()


@pytest.mark.asyncio
async def test_provider_rejection_is_bounded_unavailable_and_not_fake_empty(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-only');monkeypatch.setenv('GOZ_IDENTITY_MODEL','configured-model')
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(403)
    from test_detection_acceptance import CASES,fixture_image
    jpeg,_,_=fixture_image(CASES[0])
    observations,status=await identity.verify_frames([dict(t=0,jpeg=jpeg)],['UnknownName'],tmp_path,httpx.MockTransport(handler))
    assert observations is None and status['status']=='unavailable' and len(calls)==1
    assert not list(tmp_path.rglob('*.json'))


def test_clear_full_frame_character_is_valid_and_not_rejected_as_background():
    item=result()
    item['frames'][0]['detections'][0]['box']=[0,0,1,1]
    assert identity.validate(item,1,['Patrick'])[0][0]['box']==[0,0,1,1]


def test_live_indexed_medium_responses_abstain_on_background_and_keep_clear_positives():
    from pathlib import Path
    from test_detection_acceptance import CASES,fixture_image
    from backend.adaptive import tracks
    stored=json.loads((Path(__file__).parent/'fixtures/identity_verified.json').read_text())
    observations=identity.validate(stored['result'],len(CASES),['SpongeBob','Patrick'])
    for i,case in enumerate(CASES):
        expected={'SpongeBob','Patrick'} if case['id']=='heldout-02-00' else set(case['expected'])
        assert {d['identity'] for d in observations[i]}==expected
        jpeg,w,h=fixture_image(case)
        _,regions=tracks.parse_regions(case['response'],jpeg,w,h,[{'name':'SpongeBob'},{'name':'Patrick'}])
        accepted={d['identity'] for d in observations[i] if tracks.corroborates_body({'regions':regions},d['identity'],d['box'])}
        assert accepted <= expected  # no false identity attribution on real image evidence
        if case['id']!='heldout-02-00':
            assert accepted==expected
        else:
            assert accepted=={'SpongeBob'} # documented conservative occlusion miss
