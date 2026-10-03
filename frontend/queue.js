// A strict, two-element playback queue. Generating and watching are independent.
export class PlaybackQueue {
  constructor(videos, notify) {
    this.videos = videos; this.notify = notify; this.epoch = 0;
    for (const video of videos) {
      video.addEventListener('canplay', () => { if (video === this.pending) this.tryPlay(); });
      video.addEventListener('playing', () => { if (video === this.current) this.emit('playing'); });
      video.addEventListener('waiting', () => { if (video === this.current) this.emit('loading'); });
      video.addEventListener('ended', () => { if (video === this.current && !this.stopped) {
        this.current = null; this.next++; this.stage();
      } });
      video.addEventListener('error', () => {
        if (!this.stopped && (video === this.pending || video === this.current)) {
          this.blocked = true; this.emit('error');
        }
      });
    }
    this.reset();
  }
  reset() {
    this.epoch++; this.stopped = false; this.clips = []; this.next = 0; this.current = null;
    this.pending = null; this.complete = false; this.failed = false; this.blocked = false; this.starting = false;
    for (const video of this.videos) { video.pause(); video.hidden = true; video.removeAttribute('src'); video.load(); }
  }
  stop() { this.epoch++; this.stopped = true; for (const video of this.videos) video.pause(); this.emit('cancelled'); }
  setSound(enabled) { for (const video of this.videos) video.muted = !enabled; }
  update(clips, status) {
    if (this.stopped) return;
    this.clips = clips; this.complete = status === 'completed'; this.failed = ['failed','interrupted'].includes(status);
    this.stage();
  }
  emit(state) { this.notify({ state, index: this.next, count: this.clips.length }); }
  stage() {
    if (this.stopped || this.blocked) return;
    const index = this.current ? this.next + 1 : this.next;
    const clip = this.clips[index];
    if (clip && (!this.pending || this.pending.clipIndex !== index)) {
      this.pending = this.videos.find(v => v !== this.current);
      this.pending.clipIndex = index; this.pending.src = clip.url; this.pending.load();
    }
    if (!this.current) {
      if (clip) { this.emit('loading'); this.tryPlay(); }
      else this.emit(this.complete ? 'finished' : this.failed ? 'stopped' : 'buffering');
    }
  }
  async tryPlay() {
    if (this.current || !this.pending || this.pending.clipIndex !== this.next || this.pending.readyState < 3 || this.starting || this.blocked || this.stopped) return;
    this.starting = true;
    const video = this.pending, epoch = this.epoch;
    this.current = video; this.pending = null;
    try {
      await video.play();
      if (epoch !== this.epoch) return;
      // Swap only after play has succeeded. Keep the prior ending frame underneath while loading.
      for (const v of this.videos) v.hidden = v !== video;
      this.emit('playing'); this.stage();
    } catch (error) {
      if (epoch !== this.epoch) return;
      this.current = null; this.pending = video;
      this.emit(error.name === 'NotAllowedError' ? 'gesture' : 'error');
    } finally { if (epoch === this.epoch) this.starting = false; }
  }
}
