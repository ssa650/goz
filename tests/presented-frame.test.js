import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PresentedFrameCapture,applyTrackingReply} from '../frontend/presented-frame.js';
const video=()=>({paused:false,seeking:false,ended:false,readyState:3,videoWidth:1280,videoHeight:800});
const context=()=>({sessionId:'s',clipId:'c',epoch:1,mediaTime:0});
test('presented video capture bounds pixels, bytes and cadence; CORS failure stays graceful',()=>{
 let draws=0;
 const canvas={getContext:()=>({drawImage:()=>draws++}),toDataURL:()=> 'data:image/jpeg;base64,YQ=='};
 const c=new PresentedFrameCapture(canvas),v=video(),p=context();
 assert.equal(c.capture(v,p,true),'YQ==');assert.equal(canvas.width,640);assert.equal(canvas.height,400);
 assert.equal(c.capture(v,{...p,mediaTime:.1},true),null);assert.equal(draws,1);
 assert.equal(c.capture(v,{...p,mediaTime:.125},true),'YQ==');
 assert.equal(c.capture(v,{...p,mediaTime:.3},false),null);
 canvas.toDataURL=()=> 'data:image/jpeg;base64,'+'x'.repeat(80001);
 assert.equal(c.capture(v,{...p,mediaTime:.3},true),null);
 canvas.toDataURL=()=>{throw new DOMException('Tainted canvas','SecurityError');};
 assert.equal(c.capture(v,{...p,mediaTime:.5},true),null);
 assert.equal(c.capture(v,{...p,mediaTime:.7},true),null);assert.equal(draws,4);
 assert.equal(c.capture({...v,paused:true},{...p,epoch:2},true),null);
});
test('transient encode errors recover; a delivered current frame does not need future data',()=>{
 let fail=true,encodes=0;
 const canvas={getContext:()=>({drawImage:()=>{}}),toDataURL:()=>{encodes++;if(fail)throw new Error('Encode failed');return 'data:image/jpeg;base64,YQ==';}};
 const c=new PresentedFrameCapture(canvas),v={...video(),readyState:2},p=context();
 assert.equal(c.capture(v,p,true),null);assert.equal(c.diagnostics().reason,'encode_failed');
 fail=false;assert.equal(c.capture(v,{...p,mediaTime:.125},true),'YQ==');assert.equal(encodes,2);
 for(const fields of [{paused:true},{hidden:true},{seeking:true},{ended:true},{readyState:1},{videoWidth:0}]) {
  assert.equal(c.capture({...v,...fields},{...p,mediaTime:.3},true),null);
 }
 assert.equal(c.diagnostics().reason,'no_dimensions');
 canvas.toDataURL=()=> 'data:,';
 assert.equal(c.capture(v,{...p,mediaTime:.4},true),null);assert.equal(c.diagnostics().reason,'encode_failed');
 canvas.getContext=()=>null;
 assert.equal(c.capture(v,{...p,mediaTime:.6},true),null);assert.equal(c.diagnostics().reason,'no_context');
});
test('oversized encodes try only two qualities; counters and CORS identity stay bounded',()=>{
 const qualities=[];
 const canvas={getContext:()=>({drawImage:()=>{}}),toDataURL:(_,q)=>{qualities.push(q);return 'data:image/jpeg;base64,'+'x'.repeat(q===.8?80001:80000);}};
 const c=new PresentedFrameCapture(canvas),v=video(),p=context();
 assert.equal(c.capture(v,p,true).length,80000);assert.deepEqual(qualities,[.8,.6]);
 canvas.toDataURL=()=>{throw new DOMException('CORS','SecurityError');};
 assert.equal(c.capture(v,{...p,mediaTime:.2},true),null);
 assert.equal(c.capture(v,{...p,mediaTime:.4},true),null);assert.equal(c.diagnostics().reason,'cors_disabled');
 canvas.toDataURL=()=> 'data:image/jpeg;base64,YQ==';
 assert.equal(c.capture(v,{...p,epoch:2,mediaTime:0},true),'YQ==');
 c.counters.encoded=1e6;c.count('encoded');assert.equal(c.counters.encoded,1e6);
 const snapshot=c.diagnostics();snapshot.counters.encoded=0;assert.equal(c.counters.encoded,1e6);
});
test('late frame replies cannot mutate a new clip, seek epoch or stopped session',()=>{
 const clip={id:'c',trackingGenerationId:'g',track:[],status:'playing'};
 const state={session:{id:'s',status:'running',clips:[clip]}};
 const p={session_id:'s',clip:0,clip_id:'c',epoch:1,playing:true};
 const f={session_id:'s',clip_id:'c',generation_id:'g',playback_epoch:1,t:0,boxes:{A:[0,0,1,1]},valid_until:.25};
 const reply={clipId:'c',generationId:'g',tracking:[f]};
 applyTrackingReply(state,p,reply,2);assert.equal(clip.track.length,0);
 applyTrackingReply(state,p,{...reply,generationId:'old'},1);assert.equal(clip.track.length,0);
 applyTrackingReply(state,p,reply,1);assert.equal(clip.track.length,1);
 state.session.status='stopped';clip.track=[];applyTrackingReply(state,p,reply,1);assert.equal(clip.track.length,0);
});
test('a browser reply from before local handoff cannot replace the new tracking run',()=>{
 const clip={id:'c',trackingGenerationId:'g',trackingRunId:'local',track:[],status:'playing'};
 const state={session:{id:'s',status:'running',clips:[clip]}};
 const p={session_id:'s',clip:0,clip_id:'c',epoch:1,playing:true,tracking_run_id:'browser'};
 const f={session_id:'s',clip_id:'c',generation_id:'g',playback_epoch:1,t:0,boxes:{},valid_until:.25};
 applyTrackingReply(state,p,{clipId:'c',generationId:'g',trackingRunId:'browser',tracking:[f]},1);
 assert.equal(clip.track.length,0);
});
