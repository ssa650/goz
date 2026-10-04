import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PlaybackQueue } from '../frontend/queue.js';
import { SequencePlayback } from '../frontend/sequence-playback.js';
class Video {
  constructor() { this.events = {}; this.readyState = 4; this.plays = 0; this.muted=false; }
  addEventListener(name, callback) { this.events[name] = callback; }
  pause() { this.paused = true; }
  load() {}
  removeAttribute(name) { delete this[name]; }
  async play() { this.plays++; this.paused = false; if (this.requiresVisible && this.hidden) throw new Error('Video must be visible'); if (this.blockAutoplay) throw Object.assign(new Error(), { name: 'NotAllowedError' }); }
}
const flush = () => new Promise(resolve => setTimeout(resolve, 0));
test('plays in order, buffers, resumes, and finishes without skipping', async () => {
  const videos = [new Video(), new Video()], states = [];
  const queue = new PlaybackQueue(videos, event => states.push(event.state));
  queue.update([{url:'/clip1'}], 'generating'); await flush();
  assert.equal(queue.current.src, '/clip1');
  queue.current.events.ended(); assert.equal(states.at(-1), 'buffering');
  queue.update([{url:'/clip1'}, {url:'/clip2'}], 'completed'); await flush();
  assert.equal(queue.current.src, '/clip2');
  queue.current.events.ended(); assert.equal(states.at(-1), 'finished');
});
test('cancel prevents delayed playback and a failed clip is never skipped', async () => {
  const videos = [new Video(), new Video()], states = [];
  const queue = new PlaybackQueue(videos, event => states.push(event.state));
  queue.update([{url:'/1'}, {url:'/2'}], 'completed'); await flush();
  queue.stop(); queue.current.events.ended(); assert.equal(states.at(-1), 'cancelled');
  assert.ok(videos.every(v => v.paused));
  queue.reset(); queue.update([{url:'/bad'}, {url:'/next'}], 'completed'); await flush();
  queue.current.events.error(); assert.equal(states.at(-1), 'error');
  queue.current.events.ended(); assert.equal(states.at(-1), 'error');
});
test('blocked autoplay can recover from a user gesture', async () => {
  const videos = [new Video(), new Video()], states = [];
  videos[0].blockAutoplay = true;
  const queue = new PlaybackQueue(videos, event => states.push(event.state));
  queue.update([{url:'/1'}], 'completed'); await flush();
  assert.equal(states.at(-1), 'gesture');
  videos[0].blockAutoplay = false; await queue.tryPlay();
  assert.equal(states.at(-1), 'playing');
});

function sequence(id='run') {
  return {id,status:'generating',clips:[0,1,2,3].map(order=>({
    id:`clip-${order}`,order,jobId:`job-${order}`,duration:5,status:'generating',generatedVideoUrl:null
  }))};
}
function ready(run,index) {
  run.clips.find(c=>c.order===index).status='completed';
  run.clips.find(c=>c.order===index).generatedVideoUrl=`/clip-${index}`;
}

test('live sequence starts before completion, keeps gaps, preloads and never restarts on polling or final stitching', async () => {
  const videos=[new Video(),new Video()], states=[];
  const queue=new PlaybackQueue(videos,event=>states.push(event));
  const live=new SequencePlayback(queue,()=>false), run=sequence();
  ready(run,2); run.clips.reverse(); // Wire results can arrive in a different order.
  live.update(run); await flush();
  assert.equal(queue.current,null); assert.equal(states.at(-1).index,0);
  ready(run,0);live.update(run); await flush();
  const first=queue.current; first.currentTime=2;
  assert.equal(first.src,'/clip-0');assert.equal(run.status,'generating');
  live.update(run);await flush();
  assert.equal(queue.current,first);assert.equal(first.plays,1);assert.equal(first.currentTime,2);
  assert.equal(queue.pending,null,'Clip 3 must wait for missing Clip 2');
  first.events.ended();assert.equal(states.at(-1).state,'buffering');assert.equal(states.at(-1).index,1);
  ready(run,1);live.update(run);await flush();
  assert.equal(queue.current.src,'/clip-1');assert.notEqual(queue.current,first);
  assert.equal(queue.pending.src,'/clip-2','The next ordered clip preloads in the other element');
  queue.current.events.ended();await flush();assert.equal(queue.current.src,'/clip-2');
  ready(run,3);run.status='stitching';live.update(run);await flush();
  assert.equal(queue.current.src,'/clip-2');assert.equal(queue.pending.src,'/clip-3');
  queue.current.events.ended();await flush();assert.equal(queue.current.src,'/clip-3');
  const last=queue.current;run.status='completed';run.finalVideoUrl='/final.mp4';
  live.update(run);await flush();assert.equal(queue.current,last);assert.equal(last.plays,2);
  last.events.ended();assert.equal(states.at(-1).state,'finished');
  assert.deepEqual(states.filter(e=>e.state==='playing').map(e=>e.index),[0,1,2,3]);
});

test('stop and explicit Watch remain respected while snapshots keep arriving; a new run starts a new queue', async () => {
  const queue=new PlaybackQueue([new Video(),new Video()],()=>{});
  const live=new SequencePlayback(queue,()=>true), run=sequence();ready(run,0);
  live.update(run);await flush();queue.stop();
  ready(run,1);live.update(run);await flush();assert.ok(queue.stopped);
  live.watch([{id:'manual',duration:10,video:{url:'/manual'}}]);await flush();
  live.update(run);await flush();assert.equal(queue.current.src,'/manual');
  const newer=sequence('new-run');ready(newer,0);live.update(newer);await flush();
  assert.equal(queue.current.src,'/clip-0');assert.equal(queue.next,0);assert.ok(queue.videos.every(v=>!v.muted));
});

test('failed or cancelled sequence stops at its missing clip without jumping to later results', async () => {
  for(const status of ['failed','cancelled','interrupted']) {
    const states=[],queue=new PlaybackQueue([new Video(),new Video()],e=>states.push(e));
    const live=new SequencePlayback(queue,()=>false),run=sequence();
    ready(run,0);ready(run,2);run.status=status;
    live.update(run);await flush();queue.current.events.ended();
    assert.equal(queue.current,null);assert.equal(states.at(-1).state,'stopped');assert.equal(queue.next,1);
    live.update(run);await flush();assert.equal(queue.current,null);
  }
});

test('autoplay blocked state survives repeated polls until the user starts viewing', async () => {
  const videos=[new Video(),new Video()],states=[];videos[0].blockAutoplay=true;
  const queue=new PlaybackQueue(videos,e=>states.push(e.state)),live=new SequencePlayback(queue,()=>false);
  const run=sequence();ready(run,0);live.update(run);await flush();
  assert.equal(states.at(-1),'gesture');assert.equal(videos[0].plays,1);
  live.update(run);queue.pending.events.canplay();await flush();
  assert.equal(states.at(-1),'gesture');assert.equal(videos[0].plays,1);
  videos[0].blockAutoplay=false;await queue.tryPlay();assert.equal(states.at(-1),'playing');
});

test('playback can finish while the final MP4 is still stitching', async () => {
  const states=[],queue=new PlaybackQueue([new Video(),new Video()],e=>states.push(e.state));
  queue.update([{url:'/only-clip'}],'stitching');await flush();queue.current.events.ended();
  assert.equal(states.at(-1),'finished');
});

test('audio stays enabled in both buffers across ordered clip swaps and regeneration', async () => {
  const videos=[new Video(),new Video()];videos.forEach(v=>v.muted=true);
  const queue=new PlaybackQueue(videos,()=>{}),live=new SequencePlayback(queue,()=>true);
  const run=sequence();run.status='completed';run.clips.forEach((_,index)=>ready(run,index));
  live.update(run);await flush();
  for(let index=0;index<4;index++) {
    assert.equal(queue.current.src,`/clip-${index}`);
    assert.ok(videos.every(v=>v.muted===false));
    queue.current.events.ended();await flush();
  }
  const newer=sequence('regenerated');ready(newer,0);live.update(newer);await flush();
  assert.ok(videos.every(v=>v.muted===false));
  assert.equal(queue.current.src,'/clip-0');
});

test('next decoded video is visible before play is requested, including after a generation gap', async () => {
  const videos=[new Video(),new Video()],states=[];videos.forEach(v=>v.requiresVisible=true);
  const queue=new PlaybackQueue(videos,e=>states.push(e.state));
  queue.update([{url:'/first'}],'generating');await flush();
  assert.equal(states.at(-1),'playing');const first=queue.current;
  first.events.ended();assert.equal(first.hidden,false);
  queue.update([{url:'/first'},{url:'/second'}],'completed');await flush();
  assert.equal(states.at(-1),'playing');assert.equal(queue.current.hidden,false);assert.equal(first.hidden,true);
});

test('sustained slow production holds ending frames and uses only two media buffers', async()=>{
  const videos=[new Video(),new Video()],events=[];
  const q=new PlaybackQueue(videos,e=>events.push(e));
  const clips=[];
  for(let index=0;index<4;index++) {
    clips.push({url:`/slow-${index}`});
    q.update(clips,'running');await flush();
    assert.equal(q.current.src,`/slow-${index}`);
    const shown=q.current;
    shown.events.ended();
    assert.equal(events.at(-1).state,'buffering');
    assert.equal(q.displayed,shown);
    assert.equal(q.current,null);assert.equal(q.pending,null);
    for(let i=0;i<100;i++)q.update(clips,'running');
    assert.equal(q.displayed,shown);assert.equal(videos.length,2);
  }
  q.update(clips,'completed');assert.equal(events.at(-1).state,'finished');
});


test('transition diagnostics distinguish production gaps from preload/decode and stay bounded across polls', async () => {
  const videos=[new Video(),new Video()], diagnostics=[], states=[];
  const q=new PlaybackQueue(videos,e=>states.push(e),d=>diagnostics.push(d));
  q.update([{url:'/one'}],'running'); await flush();
  const first=q.current; first.events.ended();
  for(let i=0;i<200;i++) q.update([{url:'/one'}],'running');
  assert.equal(diagnostics.filter(d=>d.index===1 && d.reason==='buffering').length,1);
  assert.equal(q.displayed,first); assert.equal(first.hidden,false); assert.equal(q.current,null);
  videos[1].readyState=1; videos[1].networkState=2;
  q.update([{url:'/one'},{url:'/two'}],'running'); await flush();
  assert.equal(q.current,null); assert.equal(states.at(-1).state,'loading');
  assert.equal(diagnostics.find(d=>d.index===1 && d.phase==='preload').networkState,2);
  videos[1].readyState=2; videos[1].events.loadeddata(); await flush();
  assert.equal(q.current,null,'A decoded frame alone must not be reported as playing');
  videos[1].readyState=3; videos[1].events.canplay(); await flush();
  const played=diagnostics.find(d=>d.index===1 && d.phase==='playing');
  assert.equal(played.previousIndex,0); assert.ok(played.holdMs>=0);
  assert.equal(q.current.src,'/two'); assert.equal(q.current.plays,1);
  assert.ok(diagnostics.length<=48);
  assert.ok(diagnostics.every(d=>!JSON.stringify(d).includes('/one') && !JSON.stringify(d).includes('/two')));
});

test('unconfirmed play, rejection and held frames never notify sensor viewing', async () => {
  const videos=[new Video(),new Video()], states=[], diagnostics=[];
  let resolve, reject;
  videos[0].play=()=>new Promise((ok,no)=>{resolve=ok;reject=no;});
  const q=new PlaybackQueue(videos,e=>states.push(e.state),d=>diagnostics.push(d));
  q.update([{url:'/one'}],'running');
  videos[0].events.playing();
  assert.equal(states.at(-1),'loading');
  reject(Object.assign(new Error('blocked'),{name:'NotAllowedError'})); await flush();
  assert.equal(states.at(-1),'gesture'); assert.ok(!states.includes('playing'));
  const failed=diagnostics.find(d=>d.phase==='play_rejected');
  assert.equal(failed.errorName,'NotAllowedError');
  const retry=q.tryPlay(); resolve(); await retry;
  assert.equal(states.at(-1),'playing');
});

test('cancel and reset invalidate pending play completion without restarting or notifying playing', async () => {
  const videos=[new Video(),new Video()], states=[];
  let resolve;
  videos[0].play=()=>new Promise(ok=>{resolve=ok;});
  const q=new PlaybackQueue(videos,e=>states.push(e.state));
  q.update([{url:'/old'}],'completed'); q.stop(); resolve(); await flush();
  assert.equal(states.at(-1),'cancelled'); assert.ok(!states.includes('playing'));
  q.reset(); q.update([{url:'/repeat'}],'completed');
  videos[0].events.playing(); assert.equal(states.at(-1),'loading');
  resolve(); await flush(); assert.equal(states.at(-1),'playing');
  assert.equal(q.next,0); assert.equal(q.current.src,'/repeat');
});

test('completed queue with a missing ordered clip stops while holding the predecessor', async () => {
  const videos=[new Video(),new Video()],states=[];
  const q=new PlaybackQueue(videos,e=>states.push(e.state));
  q.update([{url:'/one'},null,{url:'/three'}],'completed'); await flush();
  const first=q.current; first.events.ended();
  assert.equal(states.at(-1),'stopped'); assert.equal(q.next,1); assert.equal(q.displayed,first);
  assert.equal(q.current,null); assert.equal(q.pending,null);
});

test('media error during pending play remains an error when play promise resolves', async () => {
  const videos=[new Video(),new Video()],states=[],diagnostics=[];
  let resolve; videos[0].play=()=>new Promise(ok=>{resolve=ok;});
  const q=new PlaybackQueue(videos,e=>states.push(e.state),d=>diagnostics.push(d));
  q.update([{url:'/broken'}],'completed'); videos[0].error={code:3}; videos[0].events.error();
  resolve(); await flush();
  assert.equal(states.at(-1),'error'); assert.ok(!states.includes('playing')); assert.ok(videos[0].paused);
  assert.equal(diagnostics.find(d=>d.phase==='media_error').mediaErrorCode,3);
});


test('failed progressive playback falls back once at the same position and preserves ordered dual-buffer playback',async()=>{
  const videos=[new Video(),new Video()],states=[],diagnostics=[];
  const q=new PlaybackQueue(videos,e=>states.push(e),d=>diagnostics.push(d));
  q.update([{url:'/stream-0',fallbackUrl:'/validated-0'},{url:'/stream-1',fallbackUrl:'/validated-1'}],'completed');await flush();
  const broken=q.current;broken.currentTime=6;broken.error={code:2};broken.events.error();await flush();
  assert.equal(q.next,0);assert.equal(q.current.src,'/validated-0');assert.equal(q.current.currentTime,6);
  assert.equal(diagnostics.filter(d=>d.phase==='fallback_requested').length,1);
  assert.ok(diagnostics.some(d=>d.phase==='fallback_playing'));
  q.current.events.ended();await flush();assert.equal(q.current.src,'/stream-1');assert.equal(q.next,1);
  q.current.events.ended();assert.equal(states.at(-1).state,'finished');
  assert.equal(videos.length,2);
});

test('fallback decode wait holds predecessor and a failed validated fallback never repeats or skips',async()=>{
  const videos=[new Video(),new Video()],states=[];
  const q=new PlaybackQueue(videos,e=>states.push(e.state));
  q.update([{url:'/one'}],'running');await flush();const held=q.current;held.events.ended();
  videos[1].readyState=1;
  q.update([{url:'/one'},{url:'/expired',fallbackUrl:'/validated'},{url:'/three'}],'completed');
  videos[1].events.error();assert.equal(q.current,null);assert.equal(q.displayed,held);assert.equal(held.hidden,false);
  assert.equal(q.pending.src,'/validated');assert.equal(q.next,1);assert.equal(states.at(-1),'loading');
  q.pending.events.error();assert.equal(states.at(-1),'error');
  for(let i=0;i<20;i++)q.update(q.clips,'completed');
  assert.equal(q.pending.src,'/validated');assert.equal(q.next,1);assert.equal(states.at(-1),'error');
});

test('a stale failed stream play promise cannot overwrite a fallback or restarted run',async()=>{
  const videos=[new Video(),new Video()],states=[];let resolve;
  videos[0].play=()=>new Promise(ok=>{resolve=ok;});
  const q=new PlaybackQueue(videos,e=>states.push(e));
  q.update([{url:'/stream',fallbackUrl:'/validated'}],'completed');videos[0].events.error();await flush();
  assert.equal(q.current.src,'/validated');const valid=q.current;
  resolve();await flush();assert.equal(q.current,valid);assert.equal(q.displayed,valid);
  assert.equal(states.filter(e=>e.state==='playing').length,1);
  q.stop();q.reset();videos[0].play=Video.prototype.play;
  q.update([{url:'/new'}],'completed');await flush();
  assert.equal(q.current.src,'/new');assert.equal(q.fallbacks.size,0);
});

test('autoplay denial waits for a gesture without triggering progressive fallback',async()=>{
  const videos=[new Video(),new Video()],states=[];videos[0].blockAutoplay=true;
  const q=new PlaybackQueue(videos,e=>states.push(e.state));
  q.update([{url:'/stream',fallbackUrl:'/validated'}],'completed');await flush();
  assert.equal(states.at(-1),'gesture');assert.equal(q.pending.src,'/stream');assert.equal(q.fallbacks.size,0);
  videos[0].blockAutoplay=false;await q.tryPlay();assert.equal(q.current.src,'/stream');assert.equal(states.at(-1),'playing');
});
