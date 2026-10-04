/** The same CSS coordinate transform used to draw actual boxes. No raw sensors. */
export function overlayDiagnostic({clip, mediaTime, enabled, content, canvas, video, active=true, dpr=1}) {
  const rect = value => value ? {x:value.left,y:value.top,w:value.w ?? value.width,h:value.h ?? value.height} : null;
  const frame = [...(clip?.track || [])].reverse().find(f => f.t <= mediaTime + .000001);
  const expiry = frame ? (frame.valid_until ?? frame.t + .8) : null;
  const fresh = !!frame && mediaTime <= expiry + .000001;
  const status = !enabled ? 'disabled' : !active ? 'inactive' : !content ? 'content_bounds_unavailable' :
    !frame ? 'no_frame' : !fresh ? 'expired_frame' : !Object.keys(frame.boxes || {}).length ? 'no_named_boxes' : 'rendered';
  const ox = content && canvas ? content.left-canvas.left : 0;
  const oy = content && canvas ? content.top-canvas.top : 0;
  return {enabled,status,mediaTime,frameMediaTime:frame?.t ?? null,validUntil:expiry,
    devicePixelRatio:dpr,contentRect:rect(content),canvasRect:rect(canvas),videoRect:rect(video),
    boxes:status==='rendered' ? Object.entries(frame.boxes).slice(0,6).map(([name,b]) => ({name,
      normalized:b,canvas:[ox+b[0]*content.w,oy+b[1]*content.h,(b[2]-b[0])*content.w,(b[3]-b[1])*content.h]})) : []};
}
