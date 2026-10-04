/** Media bounds in viewport CSS pixels, including object-fit and object-position.
 * Cropped content stays unbounded so hit tests never clamp.
 * @param {{left:number,top:number,width:number,height:number}} rect
 * @param {number} width @param {number} height @param {string} [fit] @param {string} [position]
 */
export function videoContentRect(rect, width, height, fit = 'contain', position = '50% 50%') {
  if (![rect.left,rect.top,rect.width,rect.height,width,height].every(Number.isFinite) ||
      !(width > 0 && height > 0 && rect.width > 0 && rect.height > 0)) return null;
  if (!['contain','cover','fill','none','scale-down'].includes(fit)) return null;
  const contain = Math.min(rect.width / width, rect.height / height);
  const scale = fit === 'none' ? 1 : fit === 'scale-down' ? Math.min(1,contain) :
    fit === 'cover' ? Math.max(rect.width / width, rect.height / height) : contain;
  const w = fit === 'fill' ? rect.width : width * scale, h = fit === 'fill' ? rect.height : height * scale;
  const parts = position.trim().split(/\s+/);
  if (parts.length !== 2) return null; // Unsupported CSS positions abstain.
  const offset = (/** @type {string} */ part, /** @type {number} */ free, /** @type {boolean} */ horizontal) => {
    if (part === 'center') return free / 2;
    if (part === (horizontal ? 'left' : 'top')) return 0;
    if (part === (horizontal ? 'right' : 'bottom')) return free;
    if (/^-?\d+(\.\d+)?%$/.test(part)) return free * parseFloat(part) / 100;
    if (/^-?\d+(\.\d+)?px$/.test(part)) return parseFloat(part);
    return null;
  };
  const x=offset(parts[0],rect.width-w,true), y=offset(parts[1],rect.height-h,false);
  return x == null || y == null ? null : {left:rect.left+x,top:rect.top+y,w,h};
}

/** Intersect content with its element, viewport and axis-specific ancestor clips.
 * Keep the full content rectangle separate: cropped/hidden gaze is never clamped.
 * @typedef {{left:number,top:number,w:number,h:number}} CSSRect
 * @param {CSSRect} content @param {CSSRect[]} bounds
 */
export function visibleContentRect(content, bounds) {
  let {left,top,w,h}=content, right=left+w, bottom=top+h;
  for (const b of bounds) {
    left=Math.max(left,b.left); top=Math.max(top,b.top);
    right=Math.min(right,b.left+b.w); bottom=Math.min(bottom,b.top+b.h);
  }
  return {left,top,w:Math.max(0,right-left),h:Math.max(0,bottom-top)};
}

/** Browser-observable identity only. No guessed browser chrome, DPR conversion or
 * cached transform reuse, even when returning to a previously seen display.
 * @param {{screenX:number,screenY:number,innerWidth:number,innerHeight:number,outerWidth:number,outerHeight:number,devicePixelRatio:number,screen:{width:number,height:number,availWidth?:number,availHeight?:number,availLeft?:number,availTop?:number,left?:number,top?:number},fullscreen:boolean,visualViewport?:{width:number,height:number,offsetLeft:number,offsetTop:number,scale:number}|null}} win
 */
export function mappingIdentity(win) {
  const s=win.screen, v=win.visualViewport;
  return JSON.stringify([win.screenX,win.screenY,win.innerWidth,win.innerHeight,win.outerWidth,win.outerHeight,
    win.devicePixelRatio,s.width,s.height,s.availWidth,s.availHeight,s.availLeft,s.availTop,s.left,s.top,
    win.fullscreen,v?.width,v?.height,v?.offsetLeft,v?.offsetTop,v?.scale]);
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
    this.reason = 'screen-mapping-unmeasured';
    this.revision = 0;
    this.recoveryActive = false;
    this.recoverySamples = 0;
  }
  /** @param {string} key */
  sync(key) {
    if (key !== this.key) this.reset(key, this.key ? 'window-display-or-zoom-changed' : 'screen-mapping-unmeasured');
  }
  /** @param {string} key @param {string} reason */
  reset(key, reason) {
    this.key=key; this.samples=[]; this.transform=null; this.reason=reason;
    this.recoveryActive=false; this.recoverySamples=0; this.revision++;
  }
  /** @param {string} key */
  beginRecovery(key) {
    this.reset(key,'mapping-targets-required'); this.recoveryActive=true;
  }
  /** Three explicit mouse/pen targets: two measure both axes, the third checks
   * their prediction. Mapping remains unusable throughout recovery.
   * @param {{clientX:number,clientY:number,screenX:number,screenY:number}} p @param {string} key
   */
  acceptAnchor(p,key) {
    this.sync(key);
    if (!this.recoveryActive) return false;
    this.observe(p,key);
    this.recoverySamples=this.samples.length;
    if (this.recoverySamples < 3) return false;
    this.recoveryActive=false;
    if (!this.transform) { this.samples=[]; return false; }
    this.reason=''; return true;
  }
  /** @param {{clientX:number,clientY:number,screenX:number,screenY:number}} p @param {string} key */
  observe(p, key) {
    this.sync(key);
    if (![p.clientX,p.clientY,p.screenX,p.screenY].every(Number.isFinite)) { this.reason='pointer-coordinates-unavailable'; return; }
    const first = this.samples[0];
    this.samples.push({clientX:p.clientX,clientY:p.clientY,screenX:p.screenX,screenY:p.screenY});
    if (this.samples.length > 64) this.samples.splice(1,1);
    if (!first) { this.reason='mapping-targets-required'; return; }
    const fit = (/** @type {'clientX'|'clientY'} */ client, /** @type {'screenX'|'screenY'} */ screen) => {
      const far = this.samples.reduce((a,b) => Math.abs(b[client]-first[client]) > Math.abs(a[client]-first[client]) ? b : a, first);
      if (Math.abs(far[client]-first[client]) < 40) return null;
      const scale = (far[screen]-first[screen])/(far[client]-first[client]);
      const offset = first[screen]-first[client]*scale;
      if (scale < .25 || scale > 4 || this.samples.some(s => Math.abs(s[screen]-offset-s[client]*scale) > 2)) return null;
      return {scale,offset};
    };
    const x=fit('clientX','screenX'), y=fit('clientY','screenY');
    const scale=x && y ? (x.scale+y.scale)/2 : 0;
    this.transform = x && y && Math.abs(x.scale-y.scale) <= .04 && this.samples.every(s =>
      Math.abs(s.screenX-x.offset-s.clientX*scale)<=2 && Math.abs(s.screenY-y.offset-s.clientY*scale)<=2) ?
      {x:x.offset,y:y.offset,scale} : null;
    const spans=['clientX','clientY'].map(k => Math.max(...this.samples.map(s=>s[/** @type {'clientX'|'clientY'} */ (k)]))-
      Math.min(...this.samples.map(s=>s[/** @type {'clientX'|'clientY'} */ (k)])));
    this.reason=this.transform ? '' : spans.some(n=>n<40) ? 'targets-too-close' : 'pointer-transform-inconsistent';
  }
  /** Check a fresh pointer against an existing fit; never infer new scale/origin.
   * @param {{clientX:number,clientY:number,screenX:number,screenY:number}} p @param {string} key
   */
  validatePointer(p,key) {
    this.sync(key);
    if (!this.valid(key) || !this.transform) return false;
    const {x,y,scale}=this.transform;
    if (![p.clientX,p.clientY,p.screenX,p.screenY].every(Number.isFinite) ||
        Math.abs(p.screenX-x-p.clientX*scale)>2 || Math.abs(p.screenY-y-p.clientY*scale)>2) {
      this.reset(key,'pointer-transform-changed'); return false;
    }
    return true;
  }
  /** @param {string} key */
  valid(key) { this.sync(key); return !!this.transform && !this.recoveryActive; }
  /** @param {{left:number,top:number,w:number,h:number}} r @param {string} key */
  rect(r,key) {
    if (!this.valid(key) || !this.transform || ![r.left,r.top,r.w,r.h].every(Number.isFinite) || r.w<=0 || r.h<=0) return null;
    const {x,y,scale}=this.transform;
    return {x:x+r.left*scale,y:y+r.top*scale,w:r.w*scale,h:r.h*scale};
  }
  /** @param {string} key */
  audit(key) {
    return {valid:this.valid(key),method:'pointer-affine-screen-points',coordinateSpace:'screen-points',
      scale:this.valid(key) ? this.transform?.scale : null,
      reason:this.valid(key) ? null : key !== this.key ? 'window-display-or-zoom-changed' : this.reason,
      eyeCalibrationVerified:false,revision:this.revision,samples:this.samples.length,
      recoveryActive:this.recoveryActive,recoverySamples:this.recoverySamples};
  }
}
