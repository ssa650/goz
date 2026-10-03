import asyncio
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import httpx
import pytest
from test_backend import harness, FakeAdapter, upload_reference, until
from backend.engine import Engine
from backend.fal_adapter import FalAdapter, FalError
from backend.config import MODELS
from backend.frames import ffmpeg, stitch_videos, media_metadata


class WireAdapter(FakeAdapter):
    """Exercise actual FalAdapter.submit HTTP serialization, no provider network."""
    def __init__(self):
        super().__init__()
        self.completed = []
        self.delays = [.08,.18,.001,.02]
        self.wire = FalAdapter('fake-test-key',transport=httpx.MockTransport(self.handler))
    async def handler(self,request):
        self.submissions.append((str(request.url).removeprefix('https://queue.fal.run/'),json.loads(request.content)))
        return httpx.Response(200,json={'request_id':f'request-{len(self.submissions)}'})
    async def submit(self,model,payload):
        return await self.wire.submit(model,payload)
    async def status(self,model,request_id):
        index=int(request_id.split('-')[-1])-1
        await asyncio.sleep(self.delays[index%4])
        return {'status':'COMPLETED'}
    async def result(self,model,request_id):
        self.completed.append(int(request_id.split('-')[-1])-1)
        return await super().result(model,request_id)
    async def close(self):
        await self.wire.close()


async def make_clips(client,mode):
    clips=[]
    colors=['red','green','blue','yellow','cyan','magenta','white','black']
    for index in range(4):
        body=dict(prompt=f'Scene {index+1}',seed=100+index,duration=15,resolution='480P',promptExpansionMode=['disabled','balanced','quality','disabled'][index])
        if mode in ('initial','both') or mode=='mixed' and index%2:
            body['firstFrame']=(await upload_reference(client,colors[index]))['id']
        if mode=='both' or mode=='mixed' and index==3:
            body['endFrame']=(await upload_reference(client,colors[index+4]))['id']
        response=await client.post('/api/clips',json=body)
        assert response.status_code==201,response.text
        clip=response.json()
        clips.append({**body,'id':clip['id'],'order':index,'firstFrame':body.get('firstFrame'),'endFrame':body.get('endFrame')})
    return clips


@pytest.fixture
def fake_stitch(monkeypatch):
    calls=[]
    async def stitch(clips,destination):
        calls.append(deepcopy(clips));destination.write_bytes(b'mock MP4')
    monkeypatch.setattr('backend.bundle_sequence.stitch_videos',stitch)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['text','initial','both','mixed'])
async def test_four_atomic_bundles_wire_payloads_out_of_order_and_stitching(tmp_path,mode,fake_stitch):
    async with harness(tmp_path,WireAdapter()) as (e,a,c):
        clips=await make_clips(c,mode)
        async def media_path(job): return Path(f"{job['clipId']}.mp4")
        e.media_path=media_path
        body=dict(id=str(uuid4()),clips=clips)
        response=await c.post('/api/sequences',json=body)
        assert response.status_code==202,response.text
        assert [clip['status'] for clip in response.json()['clips']]==['queued']*4
        run=e.sequences[body['id']]
        await until(lambda:run['status']=='completed')
        assert a.completed != list(range(4)), 'Deliberately finish out of order'
        assert len(fake_stitch)==1
        assert [item['clipId'] for item in fake_stitch[0]]==[clip['id'] for clip in clips]
        assert [item['order'] for item in fake_stitch[0]]==[0,1,2,3]
        assert [item['clipId'] for item in run['stitchInputs']]==[clip['id'] for clip in clips]
        assert not any(key in run for key in ('prompts','initialFrames','endFrames'))
        for clip in clips:
            job=e.jobs[next(x['jobId'] for x in run['clips'] if x['id']==clip['id'])]
            model,payload=next((m,p) for m,p in a.submissions if p['seed']==clip['seed'])
            assert payload['prompt']==clip['prompt'] and payload['duration']==15
            assert payload['prompt_expansion_mode']==clip['promptExpansionMode']
            assert model==MODELS['frames' if clip['firstFrame'] else 'text']
            assert payload.get('image_url')==(e.reference(clip['firstFrame'])['providerUrl'] if clip['firstFrame'] else None)
            assert payload.get('end_image_url')==(e.reference(clip['endFrame'])['providerUrl'] if clip['endFrame'] else None)
            assert job['settings']=={k:v for k,v in clip.items() if k not in ('id','order')}
        # Replay the same token even with stale editor data never makes another batch.
        assert (await c.post('/api/sequences',json=body)).json()['id']==run['id']
        assert len(a.submissions)==4
        snapshot=(await c.get(f"/api/sequences/{run['id']}")).json()
        assert [x['id'] for x in snapshot['clips']]==[x['id'] for x in clips]
        assert all(x['result']['video']['url']==x['generatedVideoUrl'] for x in snapshot['clips'])
        assert (await c.get(snapshot['finalVideoUrl'])).status_code==200
        assert 'final_video.mp4' in (await c.get(f"/api/sequences/{run['id']}/download")).headers['content-disposition']
    # Entire frozen run, results and editable ordering survive a backend restart.
    restored=Engine(FakeAdapter(),tmp_path)
    assert [x['id'] for x in restored.clips.list()]==[x['id'] for x in clips]
    assert restored.sequences[body['id']]['stitchInputs']==run['stitchInputs']
    await restored.close()


@pytest.mark.asyncio
async def test_sequence_exposes_completed_clips_while_later_clips_are_generating(tmp_path,fake_stitch):
    adapter=WireAdapter()
    adapter.delays=[.001,.25,.005,.005]
    async with harness(tmp_path,adapter) as (e,a,c):
        clips=await make_clips(c,'text')
        async def media_path(job): return Path(f"{job['clipId']}.mp4")
        e.media_path=media_path
        run_id=str(uuid4())
        await c.post('/api/sequences',json=dict(id=run_id,clips=clips))
        run=e.sequences[run_id]
        await until(lambda:run['clips'][0]['status']=='completed' and run['clips'][2]['status']=='completed')
        snapshot=(await c.get(f'/api/sequences/{run_id}')).json()
        assert snapshot['status']=='generating'
        assert snapshot['clips'][1]['generatedVideoUrl'] is None
        assert snapshot['clips'][0]['generatedVideoUrl']==f"/api/jobs/{snapshot['clips'][0]['jobId']}/video"
        assert snapshot['clips'][2]['generatedVideoUrl']==f"/api/jobs/{snapshot['clips'][2]['jobId']}/video"
        assert [clip['id'] for clip in snapshot['clips']]==[clip['id'] for clip in clips]
        assert snapshot['finalVideoUrl'] is None and not fake_stitch
        await until(lambda:run['status']=='completed')
        assert [clip['clipId'] for clip in fake_stitch[0]]==[clip['id'] for clip in clips]


@pytest.mark.asyncio
async def test_reorder_delete_duplicate_and_reload_preserve_whole_objects(tmp_path,fake_stitch):
    async with harness(tmp_path,WireAdapter()) as (e,a,c):
        clips=await make_clips(c,'both')
        ids=[clip['id'] for clip in clips]
        reordered=[ids[0],ids[2],ids[1],ids[3]]
        response=await c.post('/api/clips/order',json={'clipIds':reordered})
        assert response.status_code==200,response.text
        assert [x['id'] for x in response.json()]==reordered
        for item in response.json():
            original=next(x for x in clips if x['id']==item['id'])
            assert item['prompt']==original['prompt'] and item['seed']==original['seed']
            assert item['firstFrame']['id']==original['firstFrame'] and item['endFrame']['id']==original['endFrame']
        assert (await c.post('/api/clips/order',json={'clipIds':[ids[0]]*4})).status_code==400
        assert (await c.delete(f'/api/clips/{ids[1]}')).status_code==200
        duplicate=(await c.post(f'/api/clips/{ids[2]}/duplicate',json={})).json()
        assert duplicate['id'] not in ids
        assert duplicate['firstFrame']['id']==clips[2]['firstFrame'] and duplicate['endFrame']['id']==clips[2]['endFrame']
        assert duplicate['seed']==clips[2]['seed'] and duplicate['prompt']==clips[2]['prompt']
        assert duplicate['promptExpansionMode']==clips[2]['promptExpansionMode']
        # Restore 1,3,2,4 (new duplicate stands in for removed 2) and submit shuffled
        # JSON array; explicit order, rather than POST position, controls assembly.
        latest=(await c.get('/api/clips')).json()
        assert [x['id'] for x in latest]==[ids[0],ids[2],ids[3],duplicate['id']]
        assert [x['order'] for x in latest]==[0,1,2,3]
        ordered=[]
        for clip in latest:
            settings=e.clips.settings(e.clips.get(clip['id'])).model_dump(mode='json')
            ordered.append(dict(id=clip['id'],order=clip['order'],**settings))
        async def media_path(job): return Path(f"{job['clipId']}.mp4")
        e.media_path=media_path
        response=await c.post('/api/sequences',json=dict(id=str(uuid4()),clips=list(reversed(ordered))))
        assert response.status_code==202,response.text
        run=e.sequences[response.json()['id']]
        assert (await c.post('/api/clips/order',json={'clipIds':list(reversed([x['id'] for x in latest]))})).status_code==409
        assert (await c.patch(f"/api/clips/{ids[0]}",json={'prompt':'Do not mutate the snapshot'})).status_code==409
        assert (await c.post('/api/clips',json={'prompt':'extra'})).status_code==409
        await until(lambda:run['status']=='completed')
        assert [x['clipId'] for x in fake_stitch[0]]==[x['id'] for x in ordered]
    restored=Engine(FakeAdapter(),tmp_path)
    for before,after in zip(latest,restored.clips.list()):
        for key in ('id','order','prompt','seed','firstFrame','endFrame','promptExpansionMode','duration','resolution'):
            # Provider URLs are populated during generation; compare asset identity.
            assert (before[key]['id'] if key in ('firstFrame','endFrame') else before[key])==(after[key]['id'] if key in ('firstFrame','endFrame') else after[key])
    await restored.close()


@pytest.mark.asyncio
async def test_batch_validation_is_atomic_and_labels_each_bad_clip(tmp_path):
    async with harness(tmp_path) as (e,a,c):
        clips=await make_clips(c,'text')
        clips[0]['seed']=1.5
        clips[1]['prompt']='   '
        clips[2]['endFrame']=str(uuid4())
        clips[3]['firstFrame']=str(uuid4())
        response=await c.post('/api/sequences',json={'id':str(uuid4()),'clips':clips})
        assert response.status_code==400,response.text
        assert set(response.json()['clipErrors'])=={x['id'] for x in clips}
        assert 'Prompt is empty' in response.json()['clipErrors'][clips[1]['id']]
        assert not e.jobs and not a.submissions
        assert all(x['status']=='ready' for x in e.clips.list())


@pytest.mark.asyncio
async def test_failed_clip_does_not_shift_results_or_stitch_partial_sequence(tmp_path,fake_stitch):
    class FailThird(WireAdapter):
        async def result(self,model,request_id):
            if request_id=='request-3':raise FalError('Clip 3 provider error',422)
            return await super().result(model,request_id)
    async with harness(tmp_path,FailThird()) as (e,a,c):
        clips=await make_clips(c,'text')
        response=await c.post('/api/sequences',json=dict(id=str(uuid4()),clips=clips))
        run=e.sequences[response.json()['id']]
        await until(lambda:run['status']=='failed')
        assert [x['status'] for x in run['clips']]==['completed','completed','failed','completed']
        assert 'Clip 3 provider error' in run['clips'][2]['error']
        assert 'Clip 3 provider error' in run['error']
        assert not fake_stitch and not run['finalVideoUrl']
        assert [x['id'] for x in run['clips']]==[x['id'] for x in clips]


@pytest.mark.asyncio
async def test_account_rejection_reports_reason_and_does_not_submit_waiting_clips(tmp_path,fake_stitch):
    class EmptyBalance(WireAdapter):
        async def submit(self,model,payload):
            self.submissions.append((model,payload))
            raise FalError('User is locked. Reason: Exhausted balance.',403)
    async with harness(tmp_path,EmptyBalance()) as (e,a,c):
        clips=await make_clips(c,'text')
        response=await c.post('/api/sequences',json=dict(id=str(uuid4()),clips=clips))
        run=e.sequences[response.json()['id']]
        await until(lambda:run['status']=='failed')
        assert 1<=len(a.submissions)<=3
        assert all(x['status']=='failed' and 'Exhausted balance' in x['error'] for x in run['clips'])
        assert 'Exhausted balance' in run['error'] and 'Some clips' not in run['error']
        assert not fake_stitch and not run['finalVideoUrl'] and not e.busy()


@pytest.mark.asyncio
async def test_real_stitch_sorts_colors_and_preserves_audio_with_mixed_canvases(tmp_path):
    paths=[]
    # One silent portrait and one audible landscape exercise normalization.
    for index,(color,size,audio) in enumerate([('red','64x96',True),('blue','96x64',False),('green','64x96',True)]):
        path=tmp_path/f'source-{index}.mp4'
        args=['-f','lavfi','-i',f'color=c={color}:s={size}:r=24']
        if audio:args+=['-f','lavfi','-i','sine=frequency=440:sample_rate=48000']
        args+=['-t','0.3','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',path]
        await ffmpeg(*args);paths.append(path)
    clips=[dict(clipId=str(i),order=i,path=p,duration=.5) for i,p in enumerate(paths)]
    final=tmp_path/'final.mp4'
    await stitch_videos([clips[2],clips[0],clips[1]],final)
    meta=await asyncio.to_thread(media_metadata,final)
    assert abs(meta['duration']-1.5)<.1 and meta['size']==(64,96)
    assert meta['audio_codec']=='aac'
    import imageio_ffmpeg
    reader=imageio_ffmpeg.read_frames(str(final));next(reader)
    samples=[]
    try:
        for index,frame in enumerate(reader):
            if index in (3,15,27):
                pixel=(96//2*64+64//2)*3;samples.append(tuple(frame[pixel:pixel+3]))
    finally:reader.close()
    assert samples[0][0]>200 and samples[0][2]<30
    assert samples[1][2]>200 and samples[1][0]<30
    assert samples[2][1]>90 and samples[2][0]<30


@pytest.mark.asyncio
async def test_cancel_sequence_cancels_accepted_jobs_and_never_submits_waiting_clip(tmp_path,fake_stitch):
    adapter=FakeAdapter();adapter.queued=True
    async with harness(tmp_path,adapter) as (e,a,c):
        clips=await make_clips(c,'text')
        response=await c.post('/api/sequences',json={'id':str(uuid4()),'clips':clips})
        run=e.sequences[response.json()['id']]
        await until(lambda:len(a.submissions)==3)
        assert (await c.post(f"/api/sequences/{run['id']}/cancel")).json()['status']=='cancelled'
        await until(lambda:all(x['status']=='cancelled' for x in run['clips']))
        assert len(a.submissions)==3 and len(a.cancelled)==3 and not fake_stitch
        assert not e.bundles.active()


@pytest.mark.asyncio
async def test_restart_reconnects_existing_jobs_without_submitting_waiting_bundles(tmp_path,fake_stitch):
    adapter=FakeAdapter();adapter.queued=True
    async with harness(tmp_path,adapter) as (e,a,c):
        clips=await make_clips(c,'text')
        response=await c.post('/api/sequences',json={'id':str(uuid4()),'clips':clips})
        run_id=response.json()['id']
        await until(lambda:len(a.submissions)==3)
    async with harness(tmp_path) as (e,a,c):
        run=e.sequences[run_id]
        assert run['status']=='interrupted'
        await until(lambda:all(x['status'] in ('completed','failed') for x in run['clips']))
        assert not a.submissions and not fake_stitch and not run['finalVideoUrl']
        assert [x['id'] for x in run['clips']]==[x['id'] for x in clips]
        # An explicit retry of this run recovers its saved state, never restarts it.
        repeated=await c.post('/api/sequences',json={'id':run_id,'clips':clips})
        assert repeated.json()['status']=='interrupted' and not a.submissions
