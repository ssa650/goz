// @ts-check
// A strict, two-element playback queue. Generating and watching are independent.
/** @typedef {{url:string,index?:number,jobId?:string,duration?:number,fallbackUrl?:string|null}} PlaybackClip */
/** @typedef {HTMLVideoElement & {clipIndex?:number,resumeTime?:number}} QueueVideo */
/** @typedef {'playing'|'loading'|'buffering'|'finished'|'cancelled'|'stopped'|'gesture'|'error'} PlaybackState */
/** @typedef {{phase:string,index:number,previousIndex:number|null,epoch:number,elapsedMs:number,holdMs:number|null,readyState:number|null,networkState:number|null,videoTime:number|null,mediaErrorCode:number|null,errorName?:string,reason?:PlaybackState}} TransitionDiagnostic */
export class PlaybackQueue {
  /** @param {QueueVideo[]} videos @param {(event:{state:PlaybackState,index:number,count:number})=>void} notify @param {(event:TransitionDiagnostic)=>void} [diagnostic] */
  constructor(videos, notify, diagnostic=()=>{}) {
    this.diagnostic = diagnostic; this.playAttempt = 0;
    /** @type {Set<number>} */ this.fallbacks = new Set();
    /** @type {number|null} */ this.holdStartedAt = null;
    /** @type {Map<number,{startedAt:number,count:number,seen:Set<string>}>} */ this.diagnostics = new Map();
    this.videos = videos; this.notify = notify; this.epoch = 0; this.next=0;
    /** @type {(PlaybackClip|null)[]} */ this.clips=[];
    /** @type {QueueVideo|null} */ this.current=null;
    /** @type {QueueVideo|null} */ this.pending=null;
    /** @type {QueueVideo|null} */ this.displayed=null;
    for (const video of videos) {
      video.addEventListener('loadeddata', () => { if (video === this.pending) this.trace('loadeddata', video); });
      video.addEventListener('canplay', () => { if (video === this.pending) { this.trace('canplay', video); void this.tryPlay(false); } });
      video.addEventListener('playing', () => { if (video === this.current && !this.starting && !this.stopped && !this.blocked) this.emit('playing'); });
      video.addEventListener('waiting', () => { if (video === this.current && !this.stopped && !this.blocked) { this.trace('waiting', video); this.emit('loading'); } });
      video.addEventListener('ended', () => { if (video === this.current && !this.stopped) {
        this.holdStartedAt = Date.now(); this.trace('ended', video);
        this.current = null; this.next++; this.stage();
      } });
      video.addEventListener('error', () => {
        if (!this.stopped && (video === this.pending || video === this.current)) {
          this.trace('media_error', video); if (!this.fallback(video)) { this.blocked = true; this.emit('error'); }
        }
      });
    }
    this.reset();
  }
  reset() {
    this.epoch++; this.stopped = false; this.clips = []; this.next = 0; this.current = null;
    this.pending = null; this.displayed = null; this.complete = false; this.failed = false; this.blocked = false; this.starting = false;
    this.autoplayBlocked = false; this.holdStartedAt = null;
    this.diagnostics.clear(); this.fallbacks.clear(); this.playAttempt++;
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
  /** @param {string} phase @param {QueueVideo|null} [video] @param {PlaybackState} [reason] @param {string} [errorName] */
  trace(phase, video=this.pending||this.current, reason, errorName) {
    const index = phase === 'state' ? this.next : video?.clipIndex ?? this.next;
    let record = this.diagnostics.get(index);
    if (!record) {
      record = {startedAt:Date.now(),count:0,seen:new Set()}; this.diagnostics.set(index,record);
      if (this.diagnostics.size > 128) this.diagnostics.delete(this.diagnostics.keys().next().value ?? -1);
    }
    const key = `${phase}:${reason ?? ''}`;
    if (record.seen.has(key) || record.count >= 24) return;
    record.seen.add(key); record.count++;
    const at = Date.now();
    try { this.diagnostic({phase,index,previousIndex:this.displayed?.clipIndex ?? null,epoch:this.epoch,
      elapsedMs:Math.max(0,at-record.startedAt),holdMs:this.holdStartedAt === null ? null : Math.max(0,at-this.holdStartedAt),
      readyState:video?.readyState ?? null,networkState:video?.networkState ?? null,
      videoTime:video && Number.isFinite(video.currentTime) ? video.currentTime : null,mediaErrorCode:video?.error?.code ?? null,
      ...(reason ? {reason} : {}),...(errorName ? {errorName:errorName.slice(0,80)} : {})}); }
    catch { /* Diagnostics must not interrupt ordered playback. */ }
  }
  /** @param {PlaybackState} state */
  emit(state) { this.trace('state',this.current||this.pending,state); this.notify({ state, index: this.next, count: this.clips.length }); }
  stage() {
    if (this.stopped || this.blocked) return;
    const index = this.current ? this.next + 1 : this.next;
    const clip = this.clips[index];
    if (clip && (!this.pending || this.pending.clipIndex !== index)) {
      // Preserve the last displayed frame when the next clip arrives after a gap.
      this.pending = this.videos.find(v => v !== (this.current||this.displayed))||null;
      if(!this.pending) return;
      this.pending.clipIndex = index; this.pending.src = clip.url; this.trace('preload',this.pending); this.pending.load();
    }
    if (!this.current) {
      if (clip) { if(!this.autoplayBlocked) {this.emit('loading'); void this.tryPlay(false);} }
      else this.emit(this.failed || (this.complete && this.next < this.clips.length) ? 'stopped' : this.complete ? 'finished' : 'buffering');
    }
  }
  /** @param {QueueVideo} video */
  fallback(video) {
    const index = video.clipIndex ?? this.next, clip = this.clips[index];
    if (!clip?.fallbackUrl || this.fallbacks.has(index) || this.stopped) return false;
    this.fallbacks.add(index); this.playAttempt++; this.starting = false;
    const active = video === this.current;
    const resume = active && Number.isFinite(video.currentTime) ? video.currentTime : 0;
    video.pause(); if (active) this.current = null;
    this.pending = this.videos.find(v=>v !== this.displayed) || video;
    this.pending.clipIndex = index; this.pending.resumeTime = resume; this.pending.src = clip.fallbackUrl;
    this.blocked = false; this.trace('fallback_requested',this.pending); this.pending.load();
    if (!this.current) { this.emit('loading'); void this.tryPlay(false); }
    return true;
  }
  /** @param {boolean} [userGesture] */
  async tryPlay(userGesture=true) {
    if(userGesture) this.autoplayBlocked=false;
    if(this.autoplayBlocked) return;
    if (this.current || !this.pending || this.pending.clipIndex !== this.next || this.pending.readyState < 3 || this.starting || this.blocked || this.stopped) return;
    this.starting = true;
    const video = this.pending, epoch = this.epoch, attempt = ++this.playAttempt;
    this.current = video; this.pending = null;
    try {
      // Make the decoded next frame visible before requesting playback. Safari
      // can suppress muted playback of display:none media for power saving.
      for (const v of this.videos) v.hidden = v !== video;
      if (video.resumeTime) {
        video.currentTime = Number.isFinite(video.duration) ? Math.min(video.resumeTime,Math.max(0,video.duration-.05)) : video.resumeTime;
      }
      video.resumeTime = 0;
      this.trace('play_requested',video); await video.play();
      if (epoch !== this.epoch || attempt !== this.playAttempt) return;
      if (this.blocked) { video.pause(); return; }
      // Loading keeps the previous ending frame visible until this ready swap.
      this.trace(this.fallbacks.has(video.clipIndex ?? this.next) ? 'fallback_playing' : 'playing',video); this.holdStartedAt = null; this.displayed=video;
      // An ended event may have advanced the queue while play() was pending.
      if (this.current === video) this.emit('playing'); this.stage();
    } catch (error) {
      if (epoch !== this.epoch || attempt !== this.playAttempt) return;
      this.trace('play_rejected',video,undefined,error instanceof Error ? error.name : 'UnknownError');
      this.current = null; this.pending = video;
      for (const v of this.videos) v.hidden = v !== this.displayed;
      this.autoplayBlocked=error instanceof Error && error.name === 'NotAllowedError';
      this.blocked=!this.autoplayBlocked;
      if (this.autoplayBlocked || !this.fallback(video)) this.emit(this.autoplayBlocked ? 'gesture' : 'error');
    } finally {
      if (epoch === this.epoch && attempt === this.playAttempt) {
        this.starting = false;
        if (!this.current && !this.blocked && !this.autoplayBlocked && !this.stopped) this.stage();
      }
    }
  }
}
