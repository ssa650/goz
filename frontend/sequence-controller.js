// @ts-check
import { api, jsonRequest, sequenceFor } from './clip-state.js';
/** @typedef {import('./clip-state.js').ClipRecord} ClipRecord */
/** @typedef {import('./clip-state.js').SequenceRequest} SequenceRequest */
/** @typedef {import('./clip-state.js').SequenceRun} SequenceRun */
/** @typedef {{configured:boolean,demo:boolean,pollMs:number}} PlayerConfig */
/** @typedef {{getItem:(key:string)=>string|null,setItem:(key:string,value:string)=>void,removeItem:(key:string)=>void}} RequestStorage */

/** Regenerate the frozen inputs, including the exact seeds and asset IDs.
 * @param {SequenceRun} run @param {string} id @returns {SequenceRequest} */
export function regenerateRequest(run,id) {
  return {id,clips:[...run.clips].sort((a,b)=>a.order-b.order).map(clip=>({
    id:clip.id,order:clip.order,prompt:clip.prompt,seed:clip.seed,
    firstFrame:clip.firstFrame,endFrame:clip.endFrame,
    promptExpansionMode:clip.promptExpansionMode,duration:clip.duration,resolution:clip.resolution
  }))};
}
/** @param {SequenceRun|null} run @returns {string} */
export function runError(run) {
  if(!run || !['failed','interrupted','cancelled'].includes(run.status)) return '';
  const errors=run.clips.filter(c=>c.error||c.result?.error);
  const unique=[...new Set(errors.map(c=>c.error||c.result?.error))];
  if(unique.length===1) return unique[0]||run.error||'Generation stopped.';
  return errors.length?errors.map(c=>`Clip ${c.order+1}: ${c.error||c.result?.error}`).join('\n')
    :run.error||'Generation stopped.';
}
/** @param {number|undefined|null} ms @returns {string} */
export function formatTime(ms) {
  if(ms==null||!Number.isFinite(ms)) return '—';
  return ms<60000?`${(ms/1000).toFixed(1)}s`:`${Math.floor(ms/60000)}m ${Math.round(ms%60000/1000)}s`;
}
/** Small controller over the existing ordered bundle API. */
export class SequenceController {
  /** @param {(controller:SequenceController)=>void} changed
   * @param {(run:SequenceRun)=>void} playback
   * @param {typeof api} [request] @param {RequestStorage} [storage] */
  constructor(changed,playback,request=api,storage=sessionStorage) {
    this.changed=changed;this.playback=playback;this.request=request;this.storage=storage;
    /** @type {PlayerConfig|null} */ this.config=null;
    /** @type {ClipRecord[]} */ this.clips=[];
    /** @type {SequenceRun|null} */ this.run=null;
    /** @type {SequenceRequest|null} */ this.pending=null;
    this.starting=false;this.error='';this.connectionError='';
  }
  get busy() {return this.starting||!!this.run?.generationBusy||!!this.run&&!['completed','failed','cancelled','interrupted'].includes(this.run.status);}
  async init() {
    try {
      const [config,clips,runs]=await Promise.all([
        this.request('/api/config'),this.request('/api/clips'),this.request('/api/sequences')
      ]);
      this.config=/** @type {PlayerConfig} */(config);this.clips=/** @type {ClipRecord[]} */(clips);
      this.run=/** @type {SequenceRun[]} */(runs).find(run=>run.mode==='bundles')||null;
      const saved=this.storage.getItem('bundle-request');
      if(saved) {
        const pending=JSON.parse(saved);
        if(typeof pending.id==='string'&&Array.isArray(pending.clips)) {
          this.pending=pending;
          const existing=/** @type {SequenceRun[]} */(runs).find(run=>run.id===pending.id);
          if(existing) {this.run=existing;this.clearPending();}
        }
      }
      if(this.run) this.playback(this.run);
    } catch(error) {this.connectionError=error instanceof Error?error.message:String(error);}
    this.changed(this);
  }
  clearPending() {this.pending=null;this.storage.removeItem('bundle-request');}
  /** @param {boolean} [regenerate] */
  async generate(regenerate=false) {
    if(this.busy||!this.config?.configured) return;
    this.starting=true;this.error='';this.changed(this);
    try {
      if(!this.pending) {
        if(regenerate&&this.run) this.pending=regenerateRequest(this.run,crypto.randomUUID());
        else {
          this.clips=await this.request('/api/clips');
          if(!this.clips.length) throw new Error('The bundled sequence is missing. Restart the app with a fresh data directory.');
          this.pending=sequenceFor(this.clips,crypto.randomUUID());
        }
        this.storage.setItem('bundle-request',JSON.stringify(this.pending));
      }
      /** @type {SequenceRun} */ const run=await this.request('/api/sequences',jsonRequest(this.pending));
      this.run=run;this.clearPending();this.connectionError='';this.playback(run);
    } catch(error) {
      this.error=error instanceof Error?error.message:String(error);
      if(error instanceof Error&&'clipErrors' in error&&error.clipErrors&&typeof error.clipErrors==='object') {
        this.error=Object.entries(error.clipErrors).map(([id,detail])=>
          `Clip ${this.clips.findIndex(c=>c.id===id)+1}: ${String(detail)}`).join('\n');
      }
      if(error instanceof Error&&'status' in error) this.clearPending();
      if(this.pending) this.error+=' Click Generate to reconnect to this same run.';
    } finally {this.starting=false;this.changed(this);}
  }
  async refresh() {
    if(!this.run||this.starting) return;
    try {
      /** @type {SequenceRun} */ const run=await this.request(`/api/sequences/${this.run.id}`);
      this.run=run;this.connectionError='';this.playback(run);
    } catch(error) {this.connectionError=error instanceof Error?error.message:String(error);}
    this.changed(this);
  }
}
