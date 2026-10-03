import hashlib
import json

import httpx
import pytest
from backend.app import create_app
from backend.engine import Engine
from backend.preset import PRESET_DIRECTORY, ensure_default_sequence, load_default_sequence
from test_backend import FakeAdapter, harness


@pytest.mark.asyncio
async def test_fresh_app_loads_all_bundled_inputs_without_provider_calls(tmp_path,monkeypatch):
    monkeypatch.setenv('GOZ_DEMO','1')
    monkeypatch.setenv('GOZ_DATA_DIR',str(tmp_path))
    app=create_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
            clips=(await client.get('/api/clips')).json()
            manifest=json.loads((PRESET_DIRECTORY/'prompts.json').read_text())['clips']
            assert len(clips)==12
            for clip,original in zip(clips,manifest):
                assert clip['sourceId']==original['id']
                for field in ('prompt','seed','duration','order','promptExpansionMode'):
                    assert clip[field]==original[field]
                for field,name in [('firstFrame','initialFrameFile'),('endFrame','endFrameFile')]:
                    assert clip[field]['name']==original[name]
                    assert clip[field]['providerUrl'] is None
                    response=await client.get(clip[field]['previewUrl'])
                    assert response.status_code==200
                    assert hashlib.sha256(response.content).digest()==hashlib.sha256((PRESET_DIRECTORY/'frames'/original[name]).read_bytes()).digest()
            for previous,following in zip(clips,clips[1:]):
                assert previous['endFrame']['id']==following['firstFrame']['id']
            assert len(app.state.engine.image_refs)==13
            assert not app.state.engine.jobs and not app.state.engine.adapter.requests
            assert (await client.get('/api/config')).json()['presetAvailable']
    restored=Engine(FakeAdapter(),tmp_path)
    await ensure_default_sequence(restored)
    assert [c['id'] for c in restored.clips.list()]==[c['id'] for c in clips]
    # Even deliberate removal of all cards is preserved after initialization.
    for clip in clips:
        restored.clips.remove(clip['id'])
    await ensure_default_sequence(restored)
    assert restored.clips.list()==[]
    await restored.close()


@pytest.mark.asyncio
async def test_saved_project_is_preserved_and_explicit_preset_reset_can_be_undone(tmp_path):
    async with harness(tmp_path) as (engine,adapter,client):
        original=(await client.post('/api/clips',json={'prompt':'My saved edit','seed':123})).json()
        await ensure_default_sequence(engine)
        assert engine.clips.list()==[original]
        clips=(await client.post('/api/clips/preset',json={})).json()
        assert len(clips)==12 and all(c['firstFrame'] and c['endFrame'] for c in clips)
        assert not engine.jobs and not adapter.submissions and not adapter.uploads
        restored=(await client.post('/api/clips/import/undo',json={})).json()
        assert restored==[original]
        await ensure_default_sequence(engine)
        assert engine.clips.list()==[original]


@pytest.mark.asyncio
async def test_bad_bundled_frame_does_not_replace_saved_cards_or_store_partial_assets(tmp_path):
    broken=tmp_path/'broken-preset'
    (broken/'frames').mkdir(parents=True)
    (broken/'prompts.json').write_text(json.dumps({'clips':[dict(
        id='broken',order=0,prompt='Bad frame',seed=1,initialFrameFile='broken.jpg')]}))
    (broken/'frames'/'broken.jpg').write_bytes(b'not an image')
    engine=Engine(FakeAdapter(),tmp_path/'data')
    original=engine.clips.add({'prompt':'Keep this clip','seed':42})
    with pytest.raises(ValueError,match='Built-in frame broken.jpg could not load'):
        await load_default_sequence(engine,broken)
    assert engine.clips.list()==[original]
    assert not engine.image_refs and not engine.jobs
    (broken/'frames'/'broken.jpg').unlink()
    with pytest.raises(ValueError,match='Built-in frame broken.jpg could not load'):
        await load_default_sequence(engine,broken)
    assert engine.clips.list()==[original]
    await engine.close()
