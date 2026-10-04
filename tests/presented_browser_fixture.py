"""Isolated cached-media QA of the actual adaptive.html/adaptive.js HTTP path.

Run with the existing Playwright CLI (no installs, sensors, provider calls or
live app changes): .venv/bin/python tests/presented_browser_fixture.py --cli PATH
"""
import os
os.environ.update(GOZ_GAZE='off',GOZ_EEG='off',GOZ_REQUIRE_SENSORS='0')
os.environ.pop('OPENAI_API_KEY',None)

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

import httpx
import uvicorn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from backend.app import create_app
from backend.engine import Engine
from backend.adaptive.session import AdaptiveSession
from backend.adaptive import director,tracking_diagnostics
from backend.adaptive.local_tracker import prewarm_local_tracker
from backend.progressive_media import range_bounds
from test_backend import FakeAdapter


INSTRUMENT = r"""() => {
 const qa=window.__presentedQA={ticks:[],inFlight:0,maxInFlight:0,encodes:0,firstPresentation:null,firstBox:null,strokes:0};
 const fetchOriginal=window.fetch;
 window.fetch=async function(url,options) {
  if(url!=='/api/adaptive/tick') return fetchOriginal.call(this,url,options);
  const body=JSON.parse(options.body);qa.inFlight++;qa.maxInFlight=Math.max(qa.maxInFlight,qa.inFlight);
  const row={at:performance.now(),time:body.video_t,epoch:body.epoch,playing:body.playing,
   frameBytes:body.tracking_frame?.length||0,bodyBytes:options.body.length,capture:body.tracking_capture};
  qa.ticks.push(row);
  try {
   const response=await fetchOriginal.call(this,url,options),reply=await response.clone().json();
   row.status=response.status;row.accepted=reply.trackingFrameAccepted;row.tickAccepted=reply.tickAccepted;
   // Hold only the first response to exercise production request backpressure.
   if(qa.ticks.length===1) await new Promise(resolve=>setTimeout(resolve,350));
   return response;
  } finally {qa.inFlight--;}
 };
 const encode=HTMLCanvasElement.prototype.toDataURL;
 HTMLCanvasElement.prototype.toDataURL=function(...args) {
  qa.encodes++;
  if(qa.encodes===1) throw new Error('isolated transient encode failure');
  if(qa.encodes===2 || qa.encodes===3) return 'data:image/jpeg;base64,'+'x'.repeat(80001);
  return encode.apply(this,args);
 };
 const frame=HTMLVideoElement.prototype.requestVideoFrameCallback;
 HTMLVideoElement.prototype.requestVideoFrameCallback=function(callback) {
  return frame.call(this,(now,meta)=>{
   if(!this.hidden && !this.paused) {
    qa.firstPresentation??=performance.now();
    // The queue starts with native readiness. Once presenting, force the
    // HAVE_CURRENT_DATA case through the real reportTick/capture hooks.
    Object.defineProperty(this,'readyState',{configurable:true,get:()=>2});
   }
   callback(now,meta);
  });
 };
 const stroke=CanvasRenderingContext2D.prototype.strokeRect;
 CanvasRenderingContext2D.prototype.strokeRect=function(...args) {
  if(this.canvas.id==='overlay-canvas') {qa.strokes++;qa.firstBox??=performance.now();}
  return stroke.apply(this,args);
 };
}"""


async def main(args):
    asset=ROOT/args.asset
    data=asset.read_bytes()
    output=ROOT/'output/playwright/presented-production';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='goz-production-') as temporary:
        engine=Engine(FakeAdapter(),Path(temporary),poll_seconds=.001)
        app=create_app(engine);original=app.router.lifespan_context
        release=asyncio.Event()
        class Slow(httpx.AsyncByteStream):
            def __init__(self,data):self.data=data
            async def __aiter__(self):
                for at in range(0,len(self.data),8192):
                    yield self.data[at:at+8192]
                    await asyncio.sleep(.002)
        def upstream(request):
            start,end=range_bounds(request.headers.get('range'),len(data))
            return httpx.Response(206 if request.headers.get('range') else 200,headers={
                'Content-Type':'video/mp4','Content-Length':str(end-start+1),
                'Content-Range':f'bytes {start}-{end}/{len(data)}'},stream=Slow(data[start:end+1]))
        engine.stream_transport=httpx.MockTransport(upstream)
        async def media(job):await release.wait();return asset
        engine.media_path=media
        @asynccontextmanager
        async def lifespan(application):
            async with original(application):
                s=AdaptiveSession(engine,app.state.sensors.gaze,app.state.sensors.eeg,engine.directory,
                    'Isolated cached actual frontend',[dict(name='SpongeBob'),dict(name='Patrick')],15,'480P',None,
                    tracker='color',playback_mode='stream')
                app.state.sensors.session=s;s.max_scenes=1
                s.local_worker=await prewarm_local_tracker('color');s.tracking_readiness['state']='ready'
                job=engine.new_job(dict(mode='text',prompt='offline cached, no submission',duration=15,resolution='480P'))
                job.update(status='completed',video=dict(url='https://fal.media/offline.mp4'))
                plan=director.template(s.story,{},15,s.names)
                clip=dict(id=str(uuid4()),sessionId=s.id,index=0,status='ready',duration=15.104,ticks=[],track=[],
                    jobId=job['id'],decision=plan['decision'],decisionId=str(uuid4()),plan=plan,writer='offline',
                    playbackDelivery='stream',streamMetadata=dict(bytes=len(data)),localMediaStatus='downloading',
                    url=f'/api/adaptive/clips/{s.id}/0/stream',fallbackUrl=f'/api/adaptive/clips/{s.id}/0/validated')
                s.clips.append(clip);s.start_detection(clip,None,None,None,15.104)
                s.local_media_tasks[clip['id']]=engine.spawn(s.prepare_local_media(clip,job,{},15.104,time.perf_counter()))
                yield
        app.router.lifespan_context=lifespan
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='error'))
        server_task=asyncio.create_task(server.serve())
        session=f'goz-presented-{args.port}'
        async def cli(*command):
            proc=await asyncio.create_subprocess_exec(args.cli,'--session',session,*command,
                cwd=output,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            stdout,_=await proc.communicate()
            (output/'cli-output.txt').open('a').write(stdout.decode())
            if proc.returncode:raise RuntimeError(stdout.decode())
            return stdout.decode()
        async def evaluate(code):
            result=await cli('--raw','run-code','async (page) => {'+code+'}')
            # --raw run-code prints the explicit console result from the snippet.
            return result
        try:
            async with asyncio.timeout(10):
                while not server.started:await asyncio.sleep(.02)
            await cli('open','about:blank')
            await cli('snapshot')
            # Instrumentation observes native production submission and drawing;
            # it never implements an alternate tick/capture/overlay path.
            await evaluate(f"await page.addInitScript({INSTRUMENT}); await page.goto('http://127.0.0.1:{args.port}/adaptive.html'); await page.waitForFunction(()=>window.__presentedQA?.firstBox!=null,{{timeout:8000}});")
            s=app.state.sensors.session;clip=s.clips[0]
            assert clip['localMediaStatus']=='downloading' and 'path' not in clip
            assert any(f.get('decode',{}).get('decoder')=='browser_presented_jpeg' and f['boxes'] for f in clip['track'])
            await evaluate("await page.waitForFunction(()=>document.querySelector('video:not([hidden])')?.currentTime>=3.7,{timeout:8000});")
            frozen=clip['frozenEvidence'];assert any(f['boxes'] for f in frozen['track'])
            browser_track=len(clip['track']);browser_run=clip['trackingRunId']
            await evaluate("await page.evaluate(()=>document.querySelector('video:not([hidden])').pause()); await page.waitForFunction(()=>window.__presentedQA.ticks.at(-1)?.playing===false,{timeout:2000});")
            paused_time=clip['ticks'][-1]['video_t'];assert not clip['ticks'][-1]['playing']
            # Resume then seek backwards: old browser epochs cannot reappear.
            await evaluate("await page.evaluate(async()=>{const v=document.querySelector('video:not([hidden])');v.currentTime=.25;await v.play();}); await page.waitForFunction(()=>window.__presentedQA.ticks.at(-1)?.epoch>2 && window.__presentedQA.ticks.at(-1)?.accepted===true,{timeout:4000});")
            epoch=clip['ticks'][-1]['epoch']
            assert all(f.get('playback_epoch',epoch)==epoch for f in clip['track'])
            # The deadline already completed without generation. Enable the
            # continuation boundary now to exercise real final-frame extraction.
            s.max_scenes=2
            release.set();await s.local_media_tasks[clip['id']];await s.detection_by_clip[clip['id']]
            assert clip['trackingRunId']!=browser_run and clip['detectionInput']=='validated_local_clip'
            assert clip['boundaryFrameStatus']=='ready' and s.boundary_frames.get(clip['id'])
            assert max(f['t'] for f in clip['track'])>=15
            browser=await cli('--raw','eval','() => window.__presentedQA')
            (output/'browser-counters.txt').write_text(browser)
            qa=json.loads(browser)
            (output/'browser-counters.json').write_text(json.dumps(qa,indent=2))
            assert qa['maxInFlight']==1 and qa['firstBox']-qa['firstPresentation']<3500
            assert all(t['frameBytes']<=80000 and t['bodyBytes']<112000 for t in qa['ticks'])
            assert any(t.get('capture',{}).get('counters',{}).get('encode_failed') for t in qa['ticks'])
            assert any(t.get('capture',{}).get('counters',{}).get('payload_limit') for t in qa['ticks'])
            # Supplemental native CORS check on an isolated diagnostic page.
            # Different loopback host names give cached pixels another origin.
            # Fulfill that one diagnostic request locally because the production
            # app correctly rejects cross-origin requests before decoding.
            cors=await evaluate(f"""await page.route('**/fixture-cors',route=>route.fulfill({{contentType:'text/html',body:'<!doctype html><body>CORS fixture</body>'}}));
                await page.route('http://localhost:{args.port}/native-cors.mp4',route=>route.fulfill({{contentType:'video/mp4',path:{json.dumps(str(asset))}}}));
                await page.goto('http://127.0.0.1:{args.port}/fixture-cors');
                return await page.evaluate(async()=>{{
                  window.__presentedQA.encodes=10; // Use the native encoder here.
                  const {{PresentedFrameCapture}}=await import('/presented-frame.js');
                  const v=document.createElement('video');v.muted=true;document.body.append(v);
                  v.src='http://localhost:{args.port}/native-cors.mp4';
                  await new Promise((resolve,reject)=>{{v.onloadeddata=resolve;v.onerror=reject;}});await v.play();
                  const c=new PresentedFrameCapture(document.createElement('canvas'));
                  const p={{sessionId:'cors',clipId:'cors',epoch:1,mediaTime:v.currentTime}};
                  const first=c.capture(v,p,true),reason=c.diagnostics().reason;
                  const second=c.capture(v,{{...p,mediaTime:p.mediaTime+.15}},true);v.pause();
                  return {{first,second,reason,disabledReason:c.diagnostics().reason}};
                }});""")
            (output/'native-cors.json').write_text(cors)
            assert 'cors_tainted' in cors and 'cors_disabled' in cors
            await tracking_diagnostics.for_clip(s,clip).flush()
            rows=tracking_diagnostics.read(engine.directory,s.id)
            (output/'tracking.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            proof=ROOT/'data/adaptive/ac2920b9-fcee-4bf8-a463-017d051c4202'
            report=dict(actualFrontend='adaptive.html + adaptive.js',actualEndpoint='/api/adaptive/tick',
                worker='prewarmed color detector',asset=str(asset.relative_to(ROOT)),assetSha256=hashlib.sha256(data).hexdigest(),
                firstBoxAfterPresentationMs=round(qa['firstBox']-qa['firstPresentation'],3),
                namedBeforeLocal=True,browserRecordsBeforeHandoff=browser_track,pausedAtS=paused_time,seekEpoch=epoch,
                maxInFlight=qa['maxInFlight'],frameQueueCapacity=1,localHandoff=clip['detectionLifecycle'],
                nativeCorsTaintHandled=True,
                fullClipThroughS=max(f['t'] for f in clip['track']),boundaryFrameStatus=clip['boundaryFrameStatus'],
                frozenNamedRecords=sum(bool(f['boxes']) for f in frozen['track']),freezeWorkMs=frozen['freezeWorkMs'],
                submissions=len(engine.adapter.submissions),uploads=len(engine.adapter.uploads),
                stages=clip['presentedFrameDiagnostics'],preservedPatrickProof={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in [proof/'story.json',ROOT/'data/adaptive/decision-traces.jsonl']},
                sourceFreezeSha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in (
                    'frontend/presented-frame.js','frontend/adaptive.js','backend/adaptive/routes.py',
                    'backend/adaptive/session.py','backend/adaptive/local_tracker.py','tests/test_presented_tracking.py',
                    'tests/presented-frame.test.js','tests/presented_browser_fixture.py')})
            assert report['submissions']==report['uploads']==0
            assert any(r['event']=='tracking_cancel_requested' and r['reason']=='local_handoff' for r in rows)
            (output/'production-report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
        finally:
            try:await cli('close')
            finally:
                app.state.sensors.session.stop();server.should_exit=True
                await asyncio.wait_for(server_task,20)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cli',required=True)
    parser.add_argument('--port',type=int,default=8782)
    parser.add_argument('--asset',default='data/media/83d10712-a108-4b69-833c-156a20dfbae7.mp4')
    asyncio.run(main(parser.parse_args()))
