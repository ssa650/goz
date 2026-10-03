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
