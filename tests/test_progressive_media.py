"""Free completed-MP4 proxy/range and local-readiness lifecycle tests."""
import asyncio
import json
import struct
from pathlib import Path

import httpx
import pytest

from backend import progressive_media as progressive
from backend.adaptive.session import AdaptiveSession
from backend.adaptive.sensors import GazeFeed, EegFeed
from backend.engine import Engine
from backend.frames import ffmpeg, verify_image
from test_adaptive import png
from test_backend import FakeAdapter
from test_adaptive_http_ordering import local_session


def atom(kind, body=b''):
    return struct.pack('>I4s',len(body)+8,kind)+body


def sample():
    return atom(b'ftyp',b'isom0000')+atom(b'moov',b'metadata')+atom(b'mdat',b'payload'*6000)


def transport_for(data, *, status=206, content_type='video/mp4', redirect=None, malformed=False):
    def handler(request):
        if redirect:
            return httpx.Response(302,headers={'Location':redirect})
        byte_range=request.headers.get('range')
        start,end=progressive.range_bounds(byte_range,len(data))
        headers={'Content-Type':content_type,'Content-Length':str(end-start+1),
                 'Content-Range':f'bytes {start}-{end}/{len(data)}','Accept-Ranges':'bytes'}
        if malformed: headers['Content-Range']='bytes 1-9/10'
        return httpx.Response(status if byte_range else 200,headers=headers,content=data[start:end+1])
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_progressive_gate_requires_exact_range_and_early_complete_metadata():
    data=sample()
    result=await progressive.probe_video('https://fal.media/completed.mp4',transport=transport_for(data))
    assert result==dict(bytes=len(data),probeBytes=32768,metadata='early_moov',rangeSupported=True)
    for value in [atom(b'ftyp',b'isom')+atom(b'mdat',b'x'*40000)+atom(b'moov',b'end'),
                  b'<html>not media</html>',atom(b'ftyp',b'isom')+atom(b'moov',b'x'*40000)]:
        with pytest.raises(ValueError):
            await progressive.probe_video('https://fal.media/result.mp4',transport=transport_for(value))


@pytest.mark.asyncio
@pytest.mark.parametrize('options',[dict(status=200),dict(status=403),dict(content_type='text/html'),dict(malformed=True),
    dict(redirect='http://fal.media/insecure'),dict(redirect='https://localhost/private'),dict(redirect='https://evil.example/private')])
async def test_expired_wrong_media_and_unsafe_redirects_are_rejected_without_url_disclosure(options):
    with pytest.raises(ValueError) as error:
        await progressive.probe_video('https://fal.media/result.mp4?token=private',transport=transport_for(sample(),**options))
    assert 'token=private' not in str(error.value) and 'https://' not in str(error.value)


@pytest.mark.parametrize('value,expected',[('bytes=0-1',(0,1)),('bytes=5-',(5,99)),('bytes=-8',(92,99)),('bytes=90-200',(90,99)),(None,(0,99))])
def test_browser_single_ranges_include_safari_probe_suffix_and_open_end(value,expected):
    assert progressive.range_bounds(value,100)==expected


@pytest.mark.parametrize('value',['bytes=0-1,5-9','bytes=-0','bytes=100-','bytes=9-1','invalid','bytes=-'])
def test_invalid_ranges_never_reach_upstream(value):
    with pytest.raises(ValueError): progressive.range_bounds(value,100)


@pytest.mark.asyncio
async def test_real_mp4_faststart_prefix_gate_and_first_frame_validation(tmp_path):
    path=tmp_path/'real.mp4'
    await ffmpeg('-f','lavfi','-i','testsrc2=size=64x64:rate=10','-t','2','-pix_fmt','yuv420p','-movflags','+faststart',path)
    data=path.read_bytes()
    assert progressive.early_mp4_metadata(data[:32768])
    result=await progressive.probe_video('https://fal.media/real.mp4',transport=transport_for(data))
    assert result['bytes']==len(data)


@pytest.mark.asyncio
async def test_proxy_serves_safari_and_seek_ranges_before_local_download_finishes(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        data=sample(); engine.stream_transport=transport_for(data)
        job=engine.new_job(dict(mode='text',prompt='saved test',duration=15,resolution='480P'))
        job.update(status='completed',video=dict(url='https://fal.media/completed.mp4'))
        clip=s.clips[1]
        clip.update(jobId=job['id'],playbackDelivery='stream',streamMetadata=dict(bytes=len(data)),localMediaStatus='downloading')
        route=f'/api/adaptive/clips/{s.id}/1/stream'
        for byte_range in ['bytes=0-1','bytes=32-63','bytes=-20']:
            response=await client.get(route,headers={'Range':byte_range})
            assert response.status_code==206,response.text
            start,end=progressive.range_bounds(byte_range,len(data))
            assert response.content==data[start:end+1]
            assert response.headers['content-range']==f'bytes {start}-{end}/{len(data)}'
        assert clip['localMediaStatus']=='downloading' and 'path' not in clip
        response=await client.get(route,headers={'Range':'bytes=0-1,5-9'})
        assert response.status_code==400
        assert adapter.submissions==[]
        response=await client.get(f'/api/adaptive/clips/old-session/1/stream')
        assert response.status_code==400


@pytest.mark.asyncio
async def test_expired_proxy_uses_only_validated_local_fallback_and_never_submits(tmp_path,monkeypatch):
    async with local_session(tmp_path,monkeypatch) as (s,engine,adapter,client):
        engine.stream_transport=transport_for(sample(),status=403)
        job=engine.new_job(dict(mode='text',prompt='saved',duration=15,resolution='480P'))
        job.update(status='completed',video=dict(url='https://fal.media/expired.mp4?secret=private'))
        clip=s.clips[1]; clip.update(jobId=job['id'],playbackDelivery='stream',streamMetadata=dict(bytes=len(sample())),localMediaStatus='failed')
        response=await client.get(f'/api/adaptive/clips/{s.id}/1/stream',headers={'Range':'bytes=0-1'})
        assert response.status_code==400 and 'private' not in response.text
        assert (await client.get(f'/api/adaptive/clips/{s.id}/1/validated')).status_code==400
        local=tmp_path/'validated.mp4';local.write_bytes(sample())
        clip.update(path=str(local),localMediaStatus='validated')
        response=await client.get(f'/api/adaptive/clips/{s.id}/1/validated',headers={'Range':'bytes=0-1'})
        assert response.status_code==206 and response.content==sample()[:2]
        assert not adapter.submissions


@pytest.mark.asyncio
async def test_streaming_session_is_ready_before_slow_local_copy_and_persists_truthful_mode(tmp_path,monkeypatch):
    import backend.progressive_media
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'A shared scene.',[dict(name='Ana'),dict(name='Bea')],
                      15,'480P',verify_image(png()),playback_mode='stream')
    s.max_scenes=1
    entered,release=asyncio.Event(),asyncio.Event()
    async def media(job):
        entered.set();await release.wait();job['mediaReadyAt']=1791100000000
        return Path('fully-downloaded.mp4')
    async def probe(url): return dict(bytes=1000,metadata='early_moov',rangeSupported=True)
    monkeypatch.setattr(engine,'media_path',media)
    monkeypatch.setattr(backend.progressive_media,'probe_video',probe)
    monkeypatch.setattr(s,'start_detection',lambda *args:None)
    await s.make_scene({},s.opening,None,[])
    await entered.wait()
    clip=s.clips[0];job=engine.jobs[clip['jobId']]
    assert clip['status']=='ready' and clip['mediaReadiness']=='progressive_available'
    assert '/stream' in clip['url'] and '/validated' in clip['fallbackUrl']
    assert clip['localMediaStatus']=='downloading' and 'fullyValidatedAt' not in clip and 'path' not in clip
    assert job['playbackMode']=='stream' and 'streamReadyAt' in job and 'continuationReadyAt' not in job
    assert json.loads((s.dir/'session.json').read_text())['playbackMode']=='stream'
    release.set();await s.local_media_tasks[clip['id']]
    assert clip['localMediaStatus']=='validated' and clip['path']=='fully-downloaded.mp4'
    assert clip['fullyValidatedAt']>=clip['streamReadyAt']
    assert job['continuationReadyAt']>=job['streamReadyAt']
    assert len(engine.adapter.submissions)==1
    await engine.close()


@pytest.mark.asyncio
async def test_unavailable_stream_falls_back_to_full_download_without_extra_generation(tmp_path,monkeypatch):
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'A shared scene.',[dict(name='Ana'),dict(name='Bea')],
                      15,'480P',verify_image(png()),playback_mode='stream');s.max_scenes=1
    async def media(job): return Path('validated.mp4')
    async def probe(url): raise ValueError('Progressive range unsupported')
    monkeypatch.setattr(engine,'media_path',media)
    monkeypatch.setattr(progressive,'probe_video',probe)
    monkeypatch.setattr(s,'start_detection',lambda *args:None)
    await s.make_scene({},s.opening,None,[])
    clip=s.clips[0]
    assert clip['status']=='ready' and clip['playbackDelivery']=='download' and '/api/jobs/' in clip['url']
    assert clip['mediaReadiness']=='fully_validated' and clip['fallbackUrl'] is None
    assert any(e['kind']=='stream_unavailable' for e in s.events)
    assert len(engine.adapter.submissions)==1
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel',[False,True])
async def test_background_failed_or_cancelled_download_never_claims_validation(tmp_path,monkeypatch,cancel):
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'A shared scene.',[dict(name='Ana'),dict(name='Bea')],
                      15,'480P',verify_image(png()),playback_mode='stream');s.max_scenes=1
    release=asyncio.Event()
    async def media(job): await release.wait();raise ValueError('Downloaded media is corrupt')
    async def probe(url): return dict(bytes=1000,metadata='early_moov',rangeSupported=True)
    monkeypatch.setattr(engine,'media_path',media);monkeypatch.setattr(progressive,'probe_video',probe)
    await s.make_scene({},s.opening,None,[])
    clip=s.clips[0];task=s.local_media_tasks[clip['id']]
    if cancel:s.stop()
    else:release.set()
    await asyncio.gather(task,return_exceptions=True);await asyncio.sleep(0)
    assert clip.get('localMediaStatus') in ('failed','cancelled')
    assert 'fullyValidatedAt' not in clip and 'path' not in clip
    assert s.status==('stopped' if cancel else 'failed')
    assert len(engine.adapter.submissions)==1
    await engine.close()


@pytest.mark.asyncio
async def test_custom_continuation_freezes_then_waits_for_actual_local_end_frame(tmp_path,monkeypatch):
    import time
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'A shared scene.',[dict(name='Ana'),dict(name='Bea')],
                      15,'480P',verify_image(png()),playback_mode='stream');s.max_scenes=2
    release=asyncio.Event();actual=verify_image(png(),'actual-end.png')
    async def media(job): await release.wait();return Path('actual-local.mp4')
    async def probe(url): return dict(bytes=1000,metadata='early_moov',rangeSupported=True)
    async def end_frame(path): assert str(path)=='actual-local.mp4';return actual
    monkeypatch.setattr(engine,'media_path',media);monkeypatch.setattr(engine,'extractor',end_frame)
    monkeypatch.setattr(progressive,'probe_video',probe);monkeypatch.setattr(s,'start_detection',lambda *args:None)
    await s.make_scene({},s.opening,None,[])
    clip=s.clips[0];wall=time.time()
    s.tick(0,0,True,{},wall,clip_id=clip['id'])
    s.tick(0,5.0,True,{},wall+.001,clip_id=clip['id'])
    assert clip['frozenEvidence']['end']-clip['frozenEvidence']['start']<=5.0
    await asyncio.sleep(.01)
    assert len(engine.adapter.submissions)==1,'No custom continuation before actual final-frame extraction'
    made=[]
    async def make(decision,image,**kwargs):made.append(image)
    monkeypatch.setattr(s,'make_scene',make)
    release.set();await s.task
    assert made==[actual] and len(engine.adapter.submissions)==1
    s.stop();await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('playback,eeg_field,eeg_value',[('download','eegRunMode','cumulative_prior_clips'),
    ('stream','eegRunMode','baseline'),('download','eeg_run_mode','baseline'),(None,None,None)])
async def test_run_modes_round_trip_through_multipart_without_starting_generation(tmp_path,monkeypatch,playback,eeg_field,eeg_value):
    from test_backend import harness
    async def no_generation(self):pass
    monkeypatch.setattr(AdaptiveSession,'start',no_generation)
    async with harness(tmp_path) as (engine,adapter,client):
        form=dict(premise='Shared scene.',characters=json.dumps([dict(name='Ana'),dict(name='Bea')]),duration='15',resolution='480P')
        if playback is not None:form['playback_mode']=playback
        if eeg_field is not None:form[eeg_field]=eeg_value
        response=await client.post('/api/adaptive/sessions',data=form,files={'start':('frame.png',png(),'image/png')})
        assert response.status_code==202,response.text
        result=response.json()
        assert result['playbackMode']==(playback or 'download')
        assert result['eegRunMode']==(eeg_value or 'cumulative_prior_clips')
        persisted=json.loads((engine.directory/'adaptive'/result['id']/'session.json').read_text())
        assert persisted['playbackMode']==result['playbackMode'] and persisted['eegRunMode']==result['eegRunMode']
        assert not engine.jobs and not adapter.submissions and not adapter.uploads


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value',[('playback_mode','unfinished_stream'),('eegRunMode','emotion_detector')])
async def test_invalid_run_modes_are_rejected_before_generation(tmp_path,monkeypatch,field,value):
    from test_backend import harness
    async with harness(tmp_path) as (engine,adapter,client):
        form=dict(premise='Shared scene.',characters=json.dumps([dict(name='Ana'),dict(name='Bea')]),duration='15',resolution='480P')
        form[field]=value
        response=await client.post('/api/adaptive/sessions',data=form,files={'start':('frame.png',png(),'image/png')})
        assert response.status_code==400
        assert not engine.jobs and not adapter.submissions


@pytest.mark.asyncio
async def test_last_scene_finished_before_slow_local_copy_still_records_full_validation(tmp_path,monkeypatch):
    engine=Engine(FakeAdapter(),tmp_path,poll_seconds=.001)
    s=AdaptiveSession(engine,GazeFeed(),EegFeed(),tmp_path,'A shared scene.',[dict(name='Ana'),dict(name='Bea')],
                      15,'480P',verify_image(png()),playback_mode='stream');s.max_scenes=1
    release=asyncio.Event()
    async def media(job):await release.wait();return Path('validated-after-playback.mp4')
    async def probe(url):return dict(bytes=1000,metadata='early_moov',rangeSupported=True)
    monkeypatch.setattr(engine,'media_path',media);monkeypatch.setattr(progressive,'probe_video',probe)
    def forbidden(*args):raise AssertionError('Finished scenes must not start tracking again')
    monkeypatch.setattr(s,'start_detection',forbidden)
    await s.make_scene({},s.opening,None,[])
    clip=s.clips[0];clip['status']='watched';s.status='finished';release.set()
    await s.local_media_tasks[clip['id']]
    assert clip['localMediaStatus']=='validated' and clip['mediaReadiness']=='fully_validated'
    assert 'fullyValidatedAt' in clip and clip['path']=='validated-after-playback.mp4'
    assert len(engine.adapter.submissions)==1
    await engine.close()
