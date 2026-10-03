// @ts-check
/** @typedef {import('./clip-state.js').SequenceRun} SequenceRun */
/** @typedef {import('./clip-state.js').GenerationResult} GenerationResult */
/** @typedef {import('./queue.js').PlaybackClip} PlaybackClip */

/** Keep empty positions: a later completion must never shift into an earlier clip.
 * @param {SequenceRun} run @returns {(PlaybackClip|null)[]} */
export function playbackClips(run) {
  return [...run.clips].sort((a,b)=>a.order-b.order).map(clip=>
    clip.status==='completed' && clip.generatedVideoUrl
      ? {index:clip.order,jobId:clip.jobId,url:clip.generatedVideoUrl,duration:clip.duration}
      : null);
}

export class SequencePlayback {
  /** @param {import('./queue.js').PlaybackQueue} player @param {()=>boolean} sound */
  constructor(player,sound) {
    this.player=player;this.sound=sound;
    /** @type {string|null} */ this.runId=null;
    this.manual=false;
  }
  /** Repeated snapshots append available clips without restarting current playback.
   * @param {SequenceRun} run */
  update(run) {
    if(run.id!==this.runId) {
      this.runId=run.id;this.manual=false;
      this.player.reset();this.player.setSound(this.sound());
    }
    if(!this.manual) this.player.update(playbackClips(run),run.status);
  }
  /** Explicit Watch actions take precedence until a new sequence starts.
   * @param {GenerationResult[]} results */
  watch(results) {
    this.manual=true;this.player.reset();this.player.setSound(this.sound());
    this.player.update(results.map((job,index)=>({index,jobId:job.id,url:job.video?.url||'',duration:job.duration})),'completed');
  }
}
