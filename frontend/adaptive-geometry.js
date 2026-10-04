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


/** Measured viewport CSS -> OS screen points. DPR belongs only to canvas pixels.
 * Two separated pointer observations fit each axis; one point cannot prove scale.
 * Window/display/zoom changes invalidate the fit. Calibration of eyes is separate.
 */
export class ScreenPointMapping {
  constructor() {
    this.key = '';
    /** @type {{clientX:number,clientY:number,screenX:number,screenY:number}[]} */
    this.samples = [];
    /** @type {{x:number,y:number,scale:number}|null} */
    this.transform = null;
  }
  /** @param {{clientX:number,clientY:number,screenX:number,screenY:number}} p @param {string} key */
  observe(p, key) {
    if (key !== this.key) { this.key = key; this.samples = []; this.transform = null; }
    if (![p.clientX,p.clientY,p.screenX,p.screenY].every(Number.isFinite)) return;
    const first = this.samples[0];
    this.samples.push({clientX:p.clientX,clientY:p.clientY,screenX:p.screenX,screenY:p.screenY});
    if (this.samples.length > 64) this.samples.splice(1,1);
    if (!first) return;
    const fit = (/** @type {'clientX'|'clientY'} */ client, /** @type {'screenX'|'screenY'} */ screen) => {
      const far = this.samples.reduce((a,b) => Math.abs(b[client]-first[client]) > Math.abs(a[client]-first[client]) ? b : a, first);
      if (Math.abs(far[client]-first[client]) < 40) return null;
      const scale = (far[screen]-first[screen])/(far[client]-first[client]);
      const offset = first[screen]-first[client]*scale;
      if (scale < .25 || scale > 4 || this.samples.some(s => Math.abs(s[screen]-offset-s[client]*scale) > 2)) return null;
      return {scale,offset};
    };
    const x=fit('clientX','screenX'), y=fit('clientY','screenY');
    this.transform = x && y && Math.abs(x.scale-y.scale) <= .04 ? {x:x.offset,y:y.offset,scale:(x.scale+y.scale)/2} : null;
  }
  /** @param {string} key */
  valid(key) { return key === this.key && !!this.transform; }
  /** @param {{left:number,top:number,w:number,h:number}} r @param {string} key */
  rect(r,key) {
    if (!this.valid(key) || !this.transform) return null;
    const {x,y,scale}=this.transform;
    return {x:x+r.left*scale,y:y+r.top*scale,w:r.w*scale,h:r.h*scale};
  }
  /** @param {string} key */
  audit(key) {
    return {valid:this.valid(key),method:'pointer-affine-screen-points',coordinateSpace:'screen-points',
      scale:this.valid(key) ? this.transform?.scale : null,
      reason:this.valid(key) ? null : key !== this.key ? 'window-display-or-zoom-changed' : 'move-pointer-across-both-axes',
      eyeCalibrationVerified:false};
  }
}
