import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image
import pytest

from backend.app import create_app
from backend.config import MODELS, build_input
from backend.engine import Engine
from backend.fal_adapter import FalAdapter, FalError
from backend.frames import verify_image, video_url, ffmpeg, extract_last_frame
from backend.prompts import plan


def png(color="red"):
    buffer = BytesIO()
    Image.new("RGB", (64,64), color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeAdapter:
    demo = False
    def __init__(self):
        self.submissions = []
        self.uploads = []
        self.cancelled = []
        self.key = "fake-test-key"
        self.status_calls = 0
        self.upload_gate = None
        self.submit_gate = None
        self.queued = False
        self.fail_status_once = False
        self.fail_submit = False
    def configured(self):
        return bool(self.key)
    def set_key(self, key):
        self.key = key
    def redact(self, text):
        return text.replace(self.key, "[redacted]") if self.key else text
    async def upload(self, image):
        self.uploads.append(image)
        if self.upload_gate:
            await self.upload_gate.wait()
        return f"https://fal.media/frame-{len(self.uploads)}.png"
    async def submit(self, model, payload):
        self.submissions.append((model, payload))
        if self.submit_gate:
            await self.submit_gate.wait()
        if self.fail_submit:
            raise httpx.ReadTimeout("Lost confirmation")
        return {"request_id": f"request-{len(self.submissions)}"}
    async def status(self, model, request_id):
        self.status_calls += 1
        if self.fail_status_once and self.status_calls == 1:
            raise httpx.ReadTimeout("Offline")
        if request_id in self.cancelled:
            return {"status": "CANCELLED"}
        return {"status": "IN_QUEUE" if self.queued else "COMPLETED"}
    async def result(self, model, request_id):
        return {"video": {"url": f"https://fal.media/{request_id}.mp4"}, "timings": {"inference": .2}}
    async def cancel(self, model, request_id):
        self.cancelled.append(request_id)
    async def close(self):
        pass


@asynccontextmanager
async def harness(tmp_path, adapter=None, **kwargs):
    adapter = adapter or FakeAdapter()
    engine = Engine(adapter, tmp_path, poll_seconds=.001, **kwargs)
    app = create_app(engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            yield engine, adapter, client


async def until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(.005)
    raise AssertionError("Timed out waiting for mock pipeline")


def sequence_body(prompts=("First scene", "Second scene"), mode="keyframes", run_id=None, duration=5):
    data = dict(id=run_id or str(uuid4()), prompts=json.dumps(list(prompts)), mode=mode, duration=str(duration), resolution="480P")
    colors = ["red", "green", "blue"] if mode == "keyframes" else ["red"]
    files = [("frames" if mode == "keyframes" else "start", (f"{i}.png", png(color), "image/png")) for i,color in enumerate(colors)]
    return data, files


def test_python_planning_and_timed_dialogue():
    scene = "DURATION: 10 seconds.\n0:00–0:05: Start\nHello there.\n0:05–0:10: End\nGoodbye.\nEND FRAME:\nBlue sky."
    p = plan(json.dumps([scene]), "keyframes", 5)
    assert len(p["clips"]) == 2 and p["frameCount"] == 3
    assert "Hello there." in p["clips"][0] and "Goodbye." not in p["clips"][0]
    assert "Goodbye." in p["clips"][1] and "Hello there." not in p["clips"][1]
    assert "Blue sky." not in p["clips"][0] and "Blue sky." in p["clips"][1]
    assert plan(json.dumps([scene]), "chain", 5)["clips"] == [scene]
    with pytest.raises(ValueError):
        plan('["DURATION: 99999999 seconds."]', "keyframes", 5)


def test_input_modes_and_images():
    payload = build_input(dict(mode="combined", prompt="Ada waves.", duration=8, names=["Ada"]), dict(start="first",end="last",characters=["ada"]))
    assert payload["reference_image_urls"] == ["first", "last", "ada"]
    assert "Image 3 is the character reference for Ada" in payload["prompt"]
    assert "over 8 seconds" in payload["prompt"]
    assert "image_url" not in payload
    assert verify_image(png()).content_type == "image/png"
    with pytest.raises(ValueError):
        verify_image(b"\x89PNG\r\n\x1a\ninvalid")
    for url in ("http://fal.media/x", "https://evil.example/x", "https://fal.media.evil.example/x", "https://user@fal.media/x"):
        with pytest.raises(ValueError):
            video_url(url)


@pytest.mark.asyncio
async def test_ordered_keyframes_timing_and_idempotency(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        data, files = sequence_body()
        response = await c.post("/api/sequences", data=data, files=files)
        assert response.status_code == 202
        run = e.sequences[data["id"]]
        await until(lambda: run["status"] == "completed")
        assert [p[1]["prompt"] for p in a.submissions] == ["First scene", "Second scene"]
        assert all(p[0] == MODELS["frames"] and p[1]["duration"] == 5 for p in a.submissions)
        assert a.submissions[0][1]["end_image_url"] == a.submissions[1][1]["image_url"]
        assert len(a.uploads) == 3
        assert [clip["index"] for clip in run["clips"]] == [0,1]
        assert all(clip["url"].startswith("/api/jobs/") and clip["inferenceMs"] == 200 for clip in run["clips"])
        duplicate = await c.post("/api/sequences", data=data, files=files)
        assert duplicate.json()["id"] == run["id"] and len(a.submissions) == 2
        assert (await c.get('/.env')).status_code == 404
        assert (await c.get('/backend/app.py')).status_code == 404
        assert "fake-test-key" not in (tmp_path/"history.json").read_text()


@pytest.mark.asyncio
async def test_chain_uses_extracted_actual_frame(tmp_path):
    actual = verify_image(png("yellow"))
    extracted = []
    async def extractor(path):
        extracted.append(path)
        return actual
    async with harness(tmp_path, extractor=extractor) as (e,a,c):
        async def media_path(job):
            return Path("actual-output.mp4")
        e.media_path = media_path
        data,files = sequence_body(mode="chain")
        await c.post("/api/sequences",data=data,files=files)
        await until(lambda: e.sequences[data['id']]['status'] == 'completed')
        assert extracted == [Path("actual-output.mp4")]
        assert a.uploads[1].data == actual.data
        assert "end_image_url" not in a.submissions[0][1]
        assert a.submissions[1][1]['image_url'] != a.submissions[0][1]['image_url']


@pytest.mark.asyncio
async def test_invalid_inputs_and_missing_key_never_submit(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        for field,value in (("duration","5.5"),("duration","16"),("mode","bad"),("resolution","2K"),("prompts","[1]")):
            data, files = sequence_body()
            data[field] = value
            response = await c.post('/api/sequences',data=data,files=files)
            assert response.status_code == 400, response.text
        data,files = sequence_body()
        response = await c.post('/api/sequences',data=data,files=files[:1])
        assert response.status_code == 400
        files[0] = ('frames',('bad.png',b'not an image','image/png'))
        assert (await c.post('/api/sequences',data=data,files=files)).status_code == 400
        a.key = ''
        data,files = sequence_body()
        assert (await c.post('/api/sequences',data=data,files=files)).status_code == 503
        assert not a.submissions


@pytest.mark.asyncio
async def test_concurrent_runs_only_submit_once(tmp_path):
    a = FakeAdapter(); a.queued = True
    async with harness(tmp_path,a) as (e,a,c):
        body1,files1=sequence_body();body2,files2=sequence_body()
        responses = await asyncio.gather(c.post('/api/sequences',data=body1,files=files1), c.post('/api/sequences',data=body2,files=files2))
        assert sorted(r.status_code for r in responses) == [202,409]
        await until(lambda: len(a.submissions) == 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('stage',['upload','submit','queued'])
async def test_cancellation_stops_remaining_clips(tmp_path,stage):
    a=FakeAdapter()
    if stage=='upload': a.upload_gate=asyncio.Event()
    if stage=='submit': a.submit_gate=asyncio.Event()
    a.queued=True
    async with harness(tmp_path,a) as (e,a,c):
        data,files=sequence_body()
        await c.post('/api/sequences',data=data,files=files)
        await until(lambda: bool(a.uploads) if stage=='upload' else bool(a.submissions))
        response=await c.post(f"/api/sequences/{data['id']}/cancel")
        assert response.json()['status']=='cancelled'
        if a.upload_gate: a.upload_gate.set()
        if a.submit_gate: a.submit_gate.set()
        await until(lambda: not e.busy())
        assert len(a.submissions)==(0 if stage=='upload' else 1)
        if stage!='upload': assert a.cancelled==['request-1']


@pytest.mark.asyncio
async def test_lost_submission_is_blocked_across_restart(tmp_path):
    a=FakeAdapter();a.fail_submit=True
    async with harness(tmp_path,a) as (e,a,c):
        data,files=sequence_body()
        await c.post('/api/sequences',data=data,files=files)
        await until(lambda:e.sequences[data['id']]['status']=='failed')
        assert e.busy() and list(e.jobs.values())[0]['requestUncertain']
    async with harness(tmp_path) as (e,a,c):
        data,files=sequence_body()
        assert (await c.post('/api/sequences',data=data,files=files)).status_code==409
        assert not a.submissions


@pytest.mark.asyncio
async def test_transient_status_reconnect_and_legacy_modes(tmp_path):
    a=FakeAdapter();a.fail_status_once=True
    async with harness(tmp_path,a) as (e,a,c):
        for mode in ('frames','characters','combined'):
            files=[]
            if mode!='characters': files += [(name,(name+'.png',png(),'image/png')) for name in ('start','end')]
            if mode!='frames': files += [('characters',('ada.png',png('green'),'image/png'))]
            r=await c.post('/api/jobs',data=dict(mode=mode,prompt='Ada waves.',duration='8',names='["Ada"]'),files=files)
            assert r.status_code==202,r.text
            job=e.jobs[r.json()['id']]
            await until(lambda:job['status']=='completed')
            assert a.submissions[-1][0]==MODELS[mode]
            assert a.submissions[-1][1]['duration']==8
            sse=await c.get(f"/api/jobs/{job['id']}/events")
            assert 'completed' in sse.text
        assert len(a.submissions)==3 and a.status_calls==4


@pytest.mark.asyncio
async def test_restart_monitors_existing_request_without_continuing_sequence(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path/'history.json').write_text(json.dumps([dict(id='saved-job',model=MODELS['frames'],status='queued',requestId='saved-request',startedAt=0,apiStartedAt=0)]))
    (tmp_path/'sequences.json').write_text(json.dumps([dict(id='saved-sequence',status='generating',clips=[],prompts=['a','b'],index=0)]))
    async with harness(tmp_path,max_job_seconds=10**12) as (e,a,c):
        await until(lambda:e.jobs['saved-job']['status']=='completed')
        assert e.sequences['saved-sequence']['status']=='interrupted'
        assert not a.submissions


@pytest.mark.asyncio
async def test_key_restore_redaction_and_origin(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        assert (await c.post('/api/key',json={'key':'new-secret'},headers={'Origin':'https://evil.example'})).status_code==403
        assert (await c.post('/api/key',json={'key':'new-secret'})).json()=={'configured':True}
        assert e.error(ValueError('failed new-secret'))=='failed [redacted]'
        assert 'new-secret' not in (await c.get('/api/config')).text
        assert (await c.get('/api/config',headers={'Host':'evil.example'})).status_code==403


@pytest.mark.asyncio
async def test_real_video_last_frame_decoding(tmp_path):
    path=tmp_path/'two-colors.mp4'
    await ffmpeg('-f','lavfi','-i','color=red:s=64x64:r=24:d=0.5','-f','lavfi','-i','color=blue:s=64x64:r=24:d=0.5',
                 '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[out]','-map','[out]','-c:v','libx264','-threads','2',path)
    frame=await extract_last_frame(path)
    image=Image.open(BytesIO(frame.data))
    r,g,b=image.getpixel((32,32))
    assert b>200 and r<30 and g<30


@pytest.mark.asyncio
async def test_paid_post_has_no_retry_and_queue_urls():
    calls=[]
    def respond(request):
        calls.append(request)
        return httpx.Response(503,json={'detail':'temporary failure'})
    adapter=FalAdapter('test-secret',transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(FalError):
            await adapter.submit(MODELS['frames'],{'prompt':'test'})
        assert len(calls)==1
        assert str(calls[0].url)=='https://queue.fal.run/minimax/h3-max-turbo/image-to-video'
        assert adapter.request_url(MODELS['frames'],'request-1')=='https://queue.fal.run/minimax/h3-max-turbo/requests/request-1'
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_cancel_during_extraction_stops_next_submission(tmp_path):
    entered=asyncio.Event(); cancelled=asyncio.Event()
    async def extractor(path):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
    async with harness(tmp_path,extractor=extractor) as (e,a,c):
        async def media_path(job): return Path('synthetic.mp4')
        e.media_path=media_path
        data,files=sequence_body(mode='chain')
        await c.post('/api/sequences',data=data,files=files)
        await entered.wait()
        await c.post(f"/api/sequences/{data['id']}/cancel")
        await cancelled.wait()
        assert len(a.submissions)==1 and not e.busy()


@pytest.mark.asyncio
async def test_prompt_file_splitting_and_video_byte_ranges(tmp_path):
    from backend.demo import DemoAdapter
    adapter=DemoAdapter(tmp_path/'demo')
    async with harness(tmp_path,adapter) as (e,a,c):
        data,files=sequence_body()
        del data['prompts']
        files.append(('prompt_file',('scene.json',json.dumps(['DURATION: 10 seconds. A simple scene.']).encode(),'application/json')))
        response=await c.post('/api/sequences',data=data,files=files)
        assert response.status_code==202,response.text
        run=e.sequences[data['id']]
        await until(lambda:run['status']=='completed')
        assert len(run['clipDefinitions'])==2
        assert 'part 1 of 2' in run['clipDefinitions'][0]['prompt']
        video=await c.get(run['clips'][0]['url'],headers={'Range':'bytes=0-31'})
        assert video.status_code==206 and len(video.content)==32
        assert video.headers['content-type']=='video/mp4'
        download=await c.get(f"/api/jobs/{run['clips'][0]['jobId']}/download")
        assert download.status_code==200 and 'attachment' in download.headers['content-disposition']


async def upload_reference(client, color='red'):
    r=await client.post('/api/images',files={'image':('frame.png',png(color),'image/png')})
    assert r.status_code==201,r.text
    return r.json()


@pytest.mark.asyncio
async def test_individual_clips_seed_stability_duplicate_and_frame_isolation(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        first=await upload_reference(c)
        end=await upload_reference(c,'green')
        r=await c.post('/api/clips',json=dict(prompt='Clip one',seed=483729,firstFrame=first['id'],endFrame=end['id'],promptExpansionMode='balanced',duration=8,resolution='768P'))
        assert r.status_code==201,r.text
        clip1=r.json()
        clip2=(await c.post('/api/clips',json={'prompt':'Clip two'})).json()
        original_seed=clip2['seed']
        for _ in range(3):
            cards=(await c.get('/api/clips')).json()
            assert cards[1]['seed']==original_seed
        duplicate=(await c.post(f"/api/clips/{clip1['id']}/duplicate")).json()
        assert duplicate['id']!=clip1['id']
        for field in ('prompt','seed','firstFrame','endFrame','promptExpansionMode','duration','resolution'):
            assert duplicate[field]==clip1[field]
        assert duplicate['result'] is None
        await c.patch(f"/api/clips/{duplicate['id']}",json={'prompt':'Different prompt','seed':42,'firstFrame':None,'endFrame':None})
        cards=(await c.get('/api/clips')).json()
        assert cards[0]['seed']==483729 and cards[0]['firstFrame']['id']==first['id']
        assert cards[1]['firstFrame'] is None
        assert cards[2]['firstFrame'] is None and cards[2]['seed']==42
        assert (await c.get(first['previewUrl'])).content==png()
        assert not a.uploads # selection persists locally without provider calls
    async with harness(tmp_path) as (e,a,c):
        cards=(await c.get('/api/clips')).json()
        assert cards[0]['seed']==483729 and cards[1]['seed']==original_seed
        assert cards[0]['firstFrame']['id']==first['id']


@pytest.mark.asyncio
async def test_per_clip_validation_and_invalid_uploads(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        first=await upload_reference(c)
        for patch in ({'seed':1.5},{'seed':True},{'seed':'12'},{'seed':None},{'seed':2**53},{'promptExpansionMode':'unknown'},
                      {'endFrame':first['id']},{'firstFrame':str(uuid4())},{'duration':4},{'resolution':'2K'}):
            r=await c.post('/api/clips',json={'prompt':'Test',**patch})
            assert r.status_code==400,r.text
        invalid=await c.post('/api/images',files={'image':('fake.png',b'not an image','image/png')})
        assert invalid.status_code==400
        oversized=await c.post('/api/images',files={'image':('big.png',b'x'*(10*1024*1024+1),'image/png')})
        assert oversized.status_code==400
        assert (await c.post('/api/images',data={})).status_code==400
        assert not a.submissions


@pytest.mark.asyncio
async def test_outgoing_h3_http_payloads_and_regeneration_for_all_frame_states(tmp_path):
    """Inspect serialized POST bodies at the installed HTTP adapter boundary."""
    calls=[]
    def provider(request):
        if request.method=='POST':
            calls.append((str(request.url),json.loads(request.content)))
            return httpx.Response(200,json={'request_id':f'paid-{len(calls)}'})
        if request.url.path.endswith('/status'):
            return httpx.Response(200,json={'status':'COMPLETED'})
        return httpx.Response(200,json={'video':{'url':'https://fal.media/result.mp4'}})
    adapter=FalAdapter('fake-secret',transport=httpx.MockTransport(provider))
    uploads=[]
    async def upload(image):
        uploads.append(image)
        return f'https://fal.media/upload-{len(uploads)}.png'
    adapter.upload=upload
    async with harness(tmp_path,adapter) as (e,a,c):
        initial=await upload_reference(c)
        end=await upload_reference(c,'blue')
        for frames in ({},{'firstFrame':initial['id']},{'firstFrame':initial['id'],'endFrame':end['id']}):
            card=(await c.post('/api/clips',json=dict(prompt='Test exact payload',seed=483729,promptExpansionMode='quality',duration=8,resolution='768P',**frames))).json()
            token=str(uuid4())
            r=await c.post(f"/api/clips/{card['id']}/generate",json={'token':token})
            assert r.status_code==202,r.text
            job=e.jobs[r.json()['id']]
            await until(lambda:job['status']=='completed')
            url,payload=calls[-1]
            assert url=='https://queue.fal.run/minimax/h3-max-turbo/'+('image-to-video' if frames else 'text-to-video')
            assert payload['prompt']=='Test exact payload' and payload['seed']==483729
            assert payload['prompt_expansion_mode']=='quality' and payload['duration']==8 and payload['resolution']=='768P'
            assert ('image_url' in payload)==('firstFrame' in frames)
            assert ('end_image_url' in payload)==('endFrame' in frames)
            assert 'firstFrame' not in payload and 'endFrame' not in payload
            assert job['seed']==483729 # documented result has no seed echo
            assert job['generationInput']==payload
            assert 'fake-secret' not in json.dumps(job)
            before=len(calls)
            retry=await c.post(f"/api/clips/{card['id']}/generate",json={'token':token})
            assert retry.json()['id']==job['id'] and len(calls)==before
            regenerated=await c.post(f"/api/clips/{card['id']}/generate",json={'token':str(uuid4())})
            new_job=e.jobs[regenerated.json()['id']]
            await until(lambda:new_job['status']=='completed')
            assert calls[-1]==(url,payload)
        assert len(uploads)==2 # persisted references reused for exact payloads
        saved=json.loads((tmp_path/'history.json').read_text())
        assert all(j['seed']==483729 and j.get('generationInput') for j in saved)


@pytest.mark.asyncio
async def test_independent_concurrent_clips_failure_and_restart_metadata(tmp_path):
    a=FakeAdapter();a.queued=True
    async with harness(tmp_path,a) as (e,a,c):
        cards=[(await c.post('/api/clips',json={'prompt':f'Clip {i}','seed':i})).json() for i in range(4)]
        accepted=await asyncio.gather(*(c.post(f"/api/clips/{card['id']}/generate",json={'token':str(uuid4())}) for card in cards))
        assert all(r.status_code==202 for r in accepted)
        await until(lambda:len(a.submissions)==4)
        assert (await c.patch(f"/api/clips/{cards[0]['id']}",json={'seed':9})).status_code==409
        assert (await c.post(f"/api/clips/{cards[0]['id']}/generate",json={'token':str(uuid4())})).status_code==409
        original_result=a.result
        async def result(model,request_id):
            if request_id=='request-3':raise FalError('Clip 3 rejected',422)
            return await original_result(model,request_id)
        a.result=result;a.queued=False
        await until(lambda:all(j['status'] in ('completed','failed') for j in e.jobs.values()))
        results=(await c.get('/api/clips')).json()
        assert [r['status'] for r in results]==['completed','completed','failed','completed']
        assert results[2]['result']['error']=='Clip 3 rejected'
        assert results[0]['result']['seed']==0 and results[3]['result']['seed']==3
    async with harness(tmp_path) as (e,a,c):
        results=(await c.get('/api/clips')).json()
        assert results[0]['result']['generationInput']['seed']==0
        assert results[2]['status']=='failed' and results[3]['status']=='completed'
        r=await c.post(f"/api/clips/{results[2]['id']}/generate",json={'token':str(uuid4())})
        assert r.status_code==202
        await until(lambda:e.jobs[r.json()['id']]['status']=='completed')


@pytest.mark.asyncio
async def test_old_prompt_strings_migrate_once_and_unknown_legacy_seed_is_not_claimed(tmp_path):
    (tmp_path/'history.json').write_text(json.dumps([dict(id='old-job',mode='frames',model=MODELS['frames'],prompt='Old prompt',status='completed',seed=None,duration=5,resolution='480P')]))
    (tmp_path/'sequences.json').write_text(json.dumps([dict(id='old-sequence',mode='keyframes',status='completed',prompts=['Old prompt'],duration=5,resolution='480P',clips=[{'index':0,'jobId':'old-job'}])]))
    async with harness(tmp_path) as (e,a,c):
        old=(await c.get('/api/clips')).json()[0]
        assert type(old['seed']) is int and old['promptExpansionMode']=='disabled'
        assert old['firstFrame'] is None and old['endFrame'] is None
        assert old['result']['seed'] is None and old['legacyMetadata']
        again=(await c.get('/api/clips')).json()[0]
        assert again['id']==old['id'] and again['seed']==old['seed']
        legacy=(await c.get('/api/sequences')).json()
        assert len(legacy)==1 and legacy[0]['id']=='old-sequence'
    async with harness(tmp_path) as (e,a,c):
        again=(await c.get('/api/clips')).json()[0]
        assert again['id']==old['id'] and again['seed']==old['seed']


@pytest.mark.asyncio
async def test_upload_failure_and_missing_provider_url_are_visible_per_clip(tmp_path):
    a=FakeAdapter()
    async def bad_upload(image):return None
    a.upload=bad_upload
    async with harness(tmp_path,a) as (e,a,c):
        image=await upload_reference(c)
        bad=(await c.post('/api/clips',json={'prompt':'With frame','firstFrame':image['id']})).json()
        r=await c.post(f"/api/clips/{bad['id']}/generate",json={'token':str(uuid4())})
        await until(lambda:e.jobs[r.json()['id']]['status']=='failed')
        assert 'no usable URL' in e.jobs[r.json()['id']]['error']
        assert not a.submissions
        good=(await c.post('/api/clips',json={'prompt':'Text only'})).json()
        r=await c.post(f"/api/clips/{good['id']}/generate",json={'token':str(uuid4())})
        assert r.status_code==202
        await until(lambda:e.jobs[r.json()['id']]['status']=='completed')


@pytest.mark.asyncio
async def test_replacing_same_image_keeps_duplicate_attachment_and_frozen_payload(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        first=await upload_reference(c)
        card=(await c.post('/api/clips',json={'prompt':'Frame test','seed':17,'firstFrame':first['id']})).json()
        generated=await c.post(f"/api/clips/{card['id']}/generate",json={'token':str(uuid4())})
        old_job=e.jobs[generated.json()['id']]
        await until(lambda:old_job['status']=='completed')
        old_payload=old_job['generationInput'].copy()
        duplicate=(await c.post(f"/api/clips/{card['id']}/duplicate")).json()
        replacement=await upload_reference(c)
        assert replacement['id']!=first['id'] and replacement['providerUrl'] is None
        await c.patch(f"/api/clips/{card['id']}",json={'firstFrame':replacement['id']})
        generated=await c.post(f"/api/clips/{card['id']}/generate",json={'token':str(uuid4())})
        new_job=e.jobs[generated.json()['id']]
        await until(lambda:new_job['status']=='completed')
        assert old_job['generationInput']==old_payload
        assert new_job['generationInput']['image_url']!=old_payload['image_url']
        assert old_job['seedSource']=='submitted' and new_job['seed']==17
        cards=(await c.get('/api/clips')).json()
        dup=next(record for record in cards if record['id']==duplicate['id'])
        assert dup['firstFrame']['id']==first['id'] and dup['firstFrame']['providerUrl']==old_payload['image_url']
