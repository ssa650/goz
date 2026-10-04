/** Provider selection is fixed for a session; each result reports its true source. */
export const TRACKER_LABELS = Object.freeze({
  fal: 'Florence + local reference tracking',
  opencv: 'Local OpenCV reference tracking',
  people: 'Person boxes (identity unavailable)',
});

export function trackerValue(value) {
  return Object.hasOwn(TRACKER_LABELS, value) ? value : 'fal';
}

export function trackingSummary(session, clip, mediaTime) {
  const provider = TRACKER_LABELS[trackerValue(session?.tracker)];
  if (!clip) return `${provider} · awaiting clip`;
  const frame = [...(clip.track || [])].reverse().find(f => f.t <= mediaTime + .000001);
  const inactive = ['stopped','finished','failed'].includes(session?.status) || clip.status === 'watched' || clip.detectionLifecycle === 'cancelled';
  const fresh = !inactive && frame && mediaTime <= (frame.valid_until ?? frame.t + .8) + .000001;
  const names = fresh ? Object.keys(frame.boxes || {}) : [];
  const phase = ['stopped','finished','failed'].includes(session?.status) ? session.status :
    clip.detectionLifecycle === 'cancelled' ? 'cancelled' : clip.detectionStatus || 'waiting';
  const sources = fresh && frame.box_sources ? [...new Set(Object.values(frame.box_sources))].join(', ') : null;
  const source = fresh ? sources || frame.source || 'unspecified' : 'no fresh frame';
  return `${provider} · ${phase} · ${names.length ? names.join(', ') : 'identity unknown'} · ${source}`;
}
