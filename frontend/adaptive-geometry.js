/** Media bounds in viewport CSS pixels. object-position is the centered position
 * used by this player. Cropped content stays unbounded so hit tests never clamp.
 * @param {{left:number,top:number,width:number,height:number}} rect
 * @param {number} width @param {number} height @param {string} [fit]
 */
export function videoContentRect(rect, width, height, fit = 'contain') {
  if (!(width > 0 && height > 0 && rect.width > 0 && rect.height > 0)) return null;
  const scale = fit === 'cover' ? Math.max(rect.width / width, rect.height / height) : Math.min(rect.width / width, rect.height / height);
  const w = fit === 'fill' ? rect.width : width * scale, h = fit === 'fill' ? rect.height : height * scale;
  return { left: rect.left + (rect.width - w) / 2, top: rect.top + (rect.height - h) / 2, w, h };
}

/** Convert viewport CSS rectangles to the tracker's screen logical-pixel space.
 * Browser chrome cannot be measured exactly by web standards; fullscreen removes
 * that uncertainty. No DPR conversion: macOS gaze and screenX/Y use points.
 * @param {{left:number,top:number,w:number,h:number}} rect
 * @param {{screenX:number,screenY:number,outerWidth:number,innerWidth:number,outerHeight:number,innerHeight:number,fullscreen:boolean,zoom?:number}} win
 */
export function screenContentRect(rect, win) {
  const z = win.zoom || 1;
  const side = win.fullscreen ? 0 : Math.max(0, (win.outerWidth-win.innerWidth*z)/2);
  const top = win.fullscreen ? 0 : Math.max(0, win.outerHeight-win.innerHeight*z-side);
  return {x:win.screenX+side+rect.left*z, y:win.screenY+top+rect.top*z, w:rect.w*z, h:rect.h*z};
}

/** @param {{t:number, boxes:Record<string,number[]>,valid_until?:number}[]} track @param {number} t */
export function boxesAt(track, t) {
  if (!track?.length) return {};
  const best = [...track].reverse().find(f => f.t <= t + .000001);
  if (!best) return {};
  return t <= (best.valid_until ?? best.t + .8) + .000001 ? best.boxes : {};
}
