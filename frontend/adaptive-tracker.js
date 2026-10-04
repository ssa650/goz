/** Provider selection is fixed for a session; each result reports its true source. */
export const TRACKER_LABELS = Object.freeze({
  color: 'Experimental color + shape heuristic',
  fal: 'Florence + local reference tracking',
  opencv: 'Local OpenCV reference tracking',
  people: 'Person boxes (identity unavailable)',
  yoloe: 'YOLOE visual reference experiment',
});

export function trackerValue(value) {
  return Object.hasOwn(TRACKER_LABELS, value) ? value : 'color';
}

// Display the detector's existing heuristic score; acceptance still uses its
// frozen visual-support/ambiguity gates rather than an invented probability.
export function colorMatchSummary(frame) {
  if (frame?.source !== 'opencv_color_shape_demo') return '';
  const regions = frame.regions || [];
  const names = [...new Set([...Object.keys(frame.boxes || {}), ...(frame.unknown || [])])];
  const labels = {
    unknown_insufficient_support: 'insufficient visual support',
    ambiguous_color_locations: 'ambiguous color locations',
    blank_frame: 'blank frame',
    no_unambiguous_color_shape_match: 'no unambiguous color/shape match',
  };
  return names.map(name => {
    const detected = Object.hasOwn(frame.boxes || {}, name);
    const candidates = regions.filter(r => (detected ? r.identity === name : r.candidate_identity === name) &&
      r.verification?.score_kind === 'uncalibrated_heuristic_score' && Number.isFinite(r.verification?.score));
    const candidate = candidates.reduce((best, r) => !best || r.verification.score > best.verification.score ? r : best, null);
    const score = candidate ? `${Math.round(100 * Math.max(0, Math.min(1, candidate.verification.score)))}%` : 'unavailable';
    const reason = candidate?.identity_status || frame.abstention_reason;
    return `${name}: color match score (uncalibrated) ${score}${detected ? '' : ` · not confidently detected (${labels[reason] || 'visual acceptance gates not met'})`}`;
  }).join(' · ');
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
  const match = fresh ? colorMatchSummary(frame) : '';
  return `${provider} · ${phase} · ${names.length ? names.join(', ') : 'identity unknown'} · ${source}${match ? ` · ${match}` : ''}`;
}
