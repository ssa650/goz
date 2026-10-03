// @ts-check
// A strict, two-element playback queue. Generating and watching are independent.
/** @typedef {{url:string,index?:number,jobId?:string,duration?:number}} PlaybackClip */
/** @typedef {HTMLVideoElement & {clipIndex?:number}} QueueVideo */
/** @typedef {'playing'|'loading'|'buffering'|'finished'|'cancelled'|'stopped'|'gesture'|'error'} PlaybackState */
export class PlaybackQueue {
  /** @param {QueueVideo[]} videos @param {(event:{state:PlaybackState,index:number,count:number})=>void} notify */
  constructor(videos, notify) {
    this.videos = videos; this.notify = notify; this.epoch = 0; this.next=0;
    /** @type {(PlaybackClip|null)[]} */ this.clips=[];
    /** @type {QueueVideo|null} */ this.current=null;
    /** @type {QueueVideo|null} */ this.pending=null;
    /** @type {QueueVideo|null} */ this.displayed=null;
    for (const video of videos) {
      video.addEventListener('canplay', () => { if (video === this.pending) void this.tryPlay(false); });
      video.addEventListener('playing', () => { if (video === this.current && !this.stopped && !this.blocked) this.emit('playing'); });
      video.addEventListener('waiting', () => { if (video === this.current && !this.stopped && !this.blocked) this.emit('loading'); });
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
    this.pending = null; this.displayed = null; this.complete = false; this.failed = false; this.blocked = false; this.starting = false;
    this.autoplayBlocked = false;
    for (const video of this.videos) { video.pause(); video.hidden = true; video.removeAttribute('src'); video.load(); }
  }
  stop() { this.epoch++; this.stopped = true; for (const video of this.videos) video.pause(); this.emit('cancelled'); }
  /** @param {boolean} enabled */
  setSound(enabled) { for (const video of this.videos) if(video.muted===enabled) video.muted = !enabled; }
  /** @param {(PlaybackClip|null)[]} clips @param {string} status */
  update(clips, status) {
    if (this.stopped) return;
    // Stitching means every clip has generated; playback need not await final encoding.
    this.clips = clips; this.complete = ['completed','stitching'].includes(status); this.failed = ['failed','interrupted','cancelled'].includes(status);
    this.stage();
  }
  /** @param {PlaybackState} state */
  emit(state) { this.notify({ state, index: this.next, count: this.clips.length }); }
  stage() {
    if (this.stopped || this.blocked) return;
    const index = this.current ? this.next + 1 : this.next;
    const clip = this.clips[index];
    if (clip && (!this.pending || this.pending.clipIndex !== index)) {
      // Preserve the last displayed frame when the next clip arrives after a gap.
      this.pending = this.videos.find(v => v !== (this.current||this.displayed))||null;
      if(!this.pending) return;
      this.pending.clipIndex = index; this.pending.src = clip.url; this.pending.load();
    }
    if (!this.current) {
      if (clip) { if(!this.autoplayBlocked) {this.emit('loading'); void this.tryPlay(false);} }
      else this.emit(this.complete ? 'finished' : this.failed ? 'stopped' : 'buffering');
    }
  }
  /** @param {boolean} [userGesture] */
  async tryPlay(userGesture=true) {
    if(userGesture) this.autoplayBlocked=false;
    if(this.autoplayBlocked) return;
    if (this.current || !this.pending || this.pending.clipIndex !== this.next || this.pending.readyState < 3 || this.starting || this.blocked || this.stopped) return;
    this.starting = true;
    const video = this.pending, epoch = this.epoch;
    this.current = video; this.pending = null;
    try {
      // Make the decoded next frame visible before requesting playback. Safari
      // can suppress muted playback of display:none media for power saving.
      for (const v of this.videos) v.hidden = v !== video;
      await video.play();
      if (epoch !== this.epoch) return;
      // Loading keeps the previous ending frame visible until this ready swap.
      this.displayed=video;
      this.emit('playing'); this.stage();
    } catch (error) {
      if (epoch !== this.epoch) return;
      this.current = null; this.pending = video;
      for (const v of this.videos) v.hidden = v !== this.displayed;
      this.autoplayBlocked=error instanceof Error && error.name === 'NotAllowedError';
      this.blocked=!this.autoplayBlocked;
      this.emit(this.autoplayBlocked ? 'gesture' : 'error');
    } finally { if (epoch === this.epoch) this.starting = false; }
  }
}
