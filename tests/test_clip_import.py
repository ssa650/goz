import asyncio
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4
from PIL import Image
import pytest
from test_backend import harness, until, FakeAdapter
from test_bundle_sequence import WireAdapter
from backend.clip_import import parse_clip_import
from backend.engine import Engine


def project():
    seeds=[741203,184927,593814,318650,862145,426731,975284,257419,684352,539781,103648,821576]
    names=[f'{i*15//60:02}-{i*15%60:02}.jpg' for i in range(13)]
    return {'clips':[dict(id=f'secret-box-{i+1:03}',order=i,duration=15,seed=seeds[i],
                         initialFrameFile=names[i],endFrameFile=names[i+1],promptExpansionMode='disabled',
                         prompt=f'Scene {i+1}\nMultiline dialogue and action.') for i in range(12)]}


def frame_files(source=None):
    source=source or project()
    names=list(dict.fromkeys(name for clip in source['clips'] for name in (clip['initialFrameFile'],clip['endFrameFile'])))
    files=[]
    for index,name in enumerate(names):
        output=BytesIO();Image.new('RGB',(64,64),(index*17,20,200-index*13)).save(output,format='JPEG')
        files.append(('frames',(name,output.getvalue(),'image/jpeg')))
    return files


def request_for(clips):
    return {'id':str(uuid4()),'clips':[dict(id=c['id'],order=i,prompt=c['prompt'],seed=c['seed'],duration=c['duration'],resolution=c['resolution'],
        promptExpansionMode=c['promptExpansionMode'],firstFrame=c['firstFrame']['id'] if c['firstFrame'] else None,endFrame=c['endFrame']['id'] if c['endFrame'] else None) for i,c in enumerate(clips)]}


@pytest.mark.asyncio
async def test_twelve_structured_clips_shared_frames_and_exact_wire_payloads(tmp_path,monkeypatch):
    source=project()
    async with harness(tmp_path,WireAdapter()) as (e,a,c):
        await c.post('/api/clips',json={'prompt':'An existing card','seed':42})
        response=await c.post('/api/clips/import',json={'text':json.dumps(source)})
        assert response.status_code==200,response.text
        imported=response.json()
        assert len(imported)==12
        for actual,original in zip(imported,source['clips']):
            assert actual['sourceId']==original['id']
            for field in ('prompt','seed','duration','order','initialFrameFile','endFrameFile','promptExpansionMode'):
                assert actual[field]==original[field]
        # Filename references never silently fall back to text-to-video.
        blocked=await c.post('/api/sequences',json=request_for(imported))
        assert blocked.status_code==400 and len(blocked.json()['clipErrors'])==12
        assert not a.submissions and not e.jobs
        one=await c.post(f"/api/clips/{imported[0]['id']}/generate",json={'token':str(uuid4())})
        assert one.status_code==400 and '00-00.jpg' in one.json()['error']
        # Deliberately reverse selection order. Names, not positions, bind assets.
        response=await c.post('/api/clips/frames',files=list(reversed(frame_files(source))))
        assert response.status_code==200,response.text
        clips=response.json()
        for actual,original in zip(clips,source['clips']):
            assert actual['firstFrame']['name']==original['initialFrameFile']
            assert actual['endFrame']['name']==original['endFrameFile']
            assert actual['seed']==original['seed']
        for first,second in zip(clips,clips[1:]):
            assert first['endFrame']['id']==second['firstFrame']['id']
        # Restore the whole clip object order before batch generation.
        reordered=[clips[0],clips[2],clips[1],*clips[3:]]
        await c.post('/api/clips/order',json={'clipIds':[x['id'] for x in reordered]})
        stitched=[]
        async def media_path(job):return Path(f"{job['clipId']}.mp4")
        async def stitch(inputs,path):stitched.extend(inputs);path.write_bytes(b'test MP4')
        e.media_path=media_path
        monkeypatch.setattr('backend.bundle_sequence.stitch_videos',stitch)
        body=request_for(reordered)
        response=await c.post('/api/sequences',json=body)
        assert response.status_code==202,response.text
        run=e.sequences[body['id']]
        await until(lambda:run['status']=='completed')
        assert len(a.submissions)==12 and len(a.uploads)==13
        assert [x['clipId'] for x in stitched]==[x['id'] for x in reordered]
        for clip in clips:
            model,payload=next((m,p) for m,p in a.submissions if p['seed']==clip['seed'])
            assert model=='minimax/h3-max-turbo/image-to-video'
            assert payload['prompt']==clip['prompt'] and payload['duration']==15
            assert payload['seed']==clip['seed'] and payload['prompt_expansion_mode']=='disabled'
            assert payload['image_url']==e.reference(clip['firstFrame']['id'])['providerUrl']
            assert payload['end_image_url']==e.reference(clip['endFrame']['id'])['providerUrl']
    restored=Engine(FakeAdapter(),tmp_path)
    assert [c['id'] for c in restored.clips.list()]==[x['id'] for x in reordered]
    assert [c['initialFrameFile'] for c in restored.clips.list()]==[x['initialFrameFile'] for x in reordered]
    await restored.close()


@pytest.mark.asyncio
async def test_missing_duplicate_unknown_invalid_files_do_not_shift_any_associations(tmp_path):
    source=project()
    async with harness(tmp_path) as (e,a,c):
        clips=(await c.post('/api/clips/import',json={'text':json.dumps(source)})).json()
        files=frame_files(source)
        missing=await c.post('/api/clips/frames',files=files[:-1])
        assert missing.status_code==400 and clips[-1]['id'] in missing.json()['clipErrors']
        assert '03-00.jpg' in missing.json()['clipErrors'][clips[-1]['id']]
        duplicate=await c.post('/api/clips/frames',files=files+[files[0]])
        assert duplicate.status_code==400 and 'same filename' in duplicate.json()['error']
        unknown=await c.post('/api/clips/frames',files=files+[('frames',('extra.jpg',files[0][1][1],'image/jpeg'))])
        assert unknown.status_code==400 and 'extra.jpg' in unknown.json()['error']
        invalid=deepcopy(files);invalid[-1]=('frames',('03-00.jpg',b'invalid-image','image/jpeg'))
        assert (await c.post('/api/clips/frames',files=invalid)).status_code==400
        assert all(not record.get('firstFrame') and not record.get('endFrame') for record in e.clips.library()['clipDefinitions'])
        assert not e.image_refs and not a.submissions


@pytest.mark.asyncio
async def test_twenty_four_distinct_frames_and_duplicate_pending_file_requirements(tmp_path):
    source=project()
    for i,clip in enumerate(source['clips']):
        clip['initialFrameFile']=f'start-{i}.jpg';clip['endFrameFile']=f'end-{i}.jpg'
    async with harness(tmp_path) as (e,a,c):
        clips=(await c.post('/api/clips/import',json={'text':json.dumps(source)})).json()
        response=await c.post('/api/clips/frames',files=frame_files(source))
        assert response.status_code==200,response.text
        assert len(e.image_refs)==24
        assert all(x['firstFrame']['name']==f'start-{i}.jpg' and x['endFrame']['name']==f'end-{i}.jpg' for i,x in enumerate(response.json()))
        # Clear capacity and copy the complete imported binding metadata as well.
        await c.delete(f"/api/clips/{clips[-1]['id']}")
        duplicate=(await c.post(f"/api/clips/{clips[0]['id']}/duplicate",json={})).json()
        assert duplicate['id']!=clips[0]['id'] and duplicate['seed']==clips[0]['seed']
        assert duplicate['initialFrameFile']=='start-0.jpg' and duplicate['endFrameFile']=='end-0.jpg'


@pytest.mark.asyncio
async def test_import_replace_undo_append_and_atomic_invalid_json(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        before=(await c.post('/api/clips',json={'prompt':'Keep this original','seed':55})).json()
        source=project()
        imported=await c.post('/api/clips/import',json={'text':json.dumps(source),'mode':'replace'})
        assert imported.status_code==200 and len(imported.json())==12
        assert (await c.get('/api/config')).json()['canUndoImport']
        bad=deepcopy(source);bad['clips'][-1]['seed']=1.5
        assert (await c.post('/api/clips/import',json={'text':json.dumps(bad),'mode':'replace'})).status_code==400
        assert len(e.clips.list())==12
        response=await c.post('/api/clips/import/undo',json={})
        assert response.status_code==200
        assert response.json()[0]['id']==before['id'] and response.json()[0]['seed']==55
        assert not (await c.get('/api/config')).json()['canUndoImport']
        added=await c.post('/api/clips/import',json={'text':json.dumps(['One','Two']),'mode':'append'})
        assert [x['prompt'] for x in added.json()]==['Keep this original','One','Two']
        assert (await c.post('/api/clips/import',json={'text':json.dumps(source),'mode':'append'})).status_code==400
        assert len(e.clips.list())==3


@pytest.mark.parametrize('change',[
    lambda p:p['clips'][0].update(order=1),
    lambda p:p['clips'][0].pop('order'),
    lambda p:p['clips'][0].update(initialFrameFile='../frame.jpg'),
    lambda p:p['clips'][0].update(initialFrameFile=None),
    lambda p:p['clips'][0].update(prompt=''),
    lambda p:p['clips'][0].update(seed=True),
    lambda p:p['clips'][0].update(promptExpansionMode='unsupported'),
    lambda p:p['clips'][0].update(id='secret-box-002'),
])
def test_structured_clip_import_rejects_ambiguous_settings_and_filenames(change):
    source=project();change(source)
    with pytest.raises(ValueError,match=r'Clip \d+'):
        parse_clip_import(json.dumps(source))


def test_explicit_order_and_non_json_text_formats_are_preserved():
    source=project();source['clips'].reverse()
    records,structured=parse_clip_import(json.dumps(source))
    assert structured and records[0]['sourceId']=='secret-box-001' and records[-1]['sourceId']=='secret-box-012'
    records,structured=parse_clip_import('One\nTwo')
    assert not structured and [x['settings']['prompt'] for x in records]==['One','Two']
    with pytest.raises(ValueError,match='Invalid JSON'):
        parse_clip_import('{"clips":')
