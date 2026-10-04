// Only generated video pixels are captured. At most two bounded encodes per
// cadence; CORS failures abstain until the clip/seek identity changes.
export class PresentedFrameCapture {
  constructor(canvas) { this.canvas=canvas; this.lastKey=null; this.lastTime=-1; this.failedKey=null; this.counters={}; this.reason='not_attempted'; }
  count(reason) {
    this.reason=reason;
    this.counters[reason]=Math.min(1e6,(this.counters[reason]||0)+1);
    return null;
  }
  diagnostics() { return {reason:this.reason,counters:{...this.counters}}; }
  capture(video, {sessionId,clipId,epoch,mediaTime}, enabled) {
    const key=`${sessionId}:${clipId}:${epoch}`;
    if (key!==this.lastKey) { this.lastKey=key; this.lastTime=-1; this.counters={}; }
    if (!enabled) return this.count('local_or_inactive');
    if (!Number.isFinite(mediaTime)) return this.count('invalid_time');
    if (video.paused || video.seeking || video.ended || video.hidden) return this.count('inactive_video');
    // A delivered rVFC frame only requires HAVE_CURRENT_DATA. Progressive
    // playback can briefly lack a future frame while these pixels are valid.
    if (video.readyState<2) return this.count('no_current_frame');
    if (key===this.failedKey) return this.count('cors_disabled');
    if (mediaTime-this.lastTime<.12) return this.count('cadence');
    const sw=video.videoWidth,sh=video.videoHeight;
    if (!sw || !sh) return this.count('no_dimensions');
    const width=Math.min(640,sw),height=Math.round(sh*width/sw);
    if (height>1280 || height<1) return this.count('dimensions_limit');
    this.lastTime=mediaTime;
    try {
      this.canvas.width=width;this.canvas.height=height;
      const context=this.canvas.getContext('2d');
      if (!context) return this.count('no_context');
      context.drawImage(video,0,0,width,height);
      const prefix='data:image/jpeg;base64,';
      for (const quality of [.8,.6]) {
        const value=this.canvas.toDataURL('image/jpeg',quality);
        if (!value.startsWith(prefix) || value.length===prefix.length) return this.count('encode_failed');
        if (value.length-prefix.length<=80000) { this.count('encoded'); return value.slice(prefix.length); }
      }
      return this.count('payload_limit');
    } catch (error) {
      if (error?.name==='SecurityError') { this.failedKey=key;return this.count('cors_tainted'); }
      return this.count('encode_failed');
    }
  }
}

export function applyTrackingReply(state, payload, reply, currentEpoch) {
  const s=state?.session,clip=s?.clips?.[payload.clip];
  if (!reply?.tracking || !payload.playing || currentEpoch!==payload.epoch || s?.id!==payload.session_id ||
      clip?.id!==payload.clip_id || reply.clipId!==clip.id || reply.generationId!==clip.trackingGenerationId ||
      (reply.trackingRunId!=null && reply.trackingRunId!==clip.trackingRunId) ||
      (payload.tracking_run_id!=null && payload.tracking_run_id!==clip.trackingRunId) ||
      ['stopped','finished','failed'].includes(s.status) || clip.status==='watched') return;
  const frames=new Map((clip.track||[]).map(f=>[f.t,f]));
  for (const f of reply.tracking) {
    if (f.session_id===s.id && f.clip_id===clip.id && f.generation_id===reply.generationId &&
        (f.playback_epoch==null || f.playback_epoch===payload.epoch)) frames.set(f.t,f);
  }
  clip.track=[...frames.values()].sort((a,b)=>a.t-b.t);
}
