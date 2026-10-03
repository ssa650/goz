import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PlaybackQueue } from '../frontend/queue.js';
class Video {
  constructor() { this.events = {}; this.readyState = 4; this.plays = 0; }
  addEventListener(name, callback) { this.events[name] = callback; }
  pause() { this.paused = true; }
  load() {}
  removeAttribute(name) { delete this[name]; }
  async play() { this.plays++; this.paused = false; if (this.blockAutoplay) throw Object.assign(new Error(), { name: 'NotAllowedError' }); }
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
