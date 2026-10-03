// @ts-check
import { api, jsonRequest, isGenerating, parseSeed, randomSeed, settingsFor, sequenceFor, missingFrameMessage } from './clip-state.js';
/** @typedef {import('./clip-state.js').ClipRecord} ClipRecord */
/** @typedef {import('./clip-state.js').ReferenceFrame} ReferenceFrame */
/** @typedef {import('./clip-state.js').GenerationResult} GenerationResult */
/** @typedef {import('./clip-state.js').PromptExpansionMode} PromptExpansionMode */
/** @typedef {import('./clip-state.js').SequenceRun} SequenceRun */
/** @typedef {import('./clip-state.js').SequenceRequest} SequenceRequest */
/** @typedef {{durations:number[], maxClips:number, maxImageBytes:number, configured:boolean, demo:boolean,canUndoImport?:boolean,presetAvailable?:boolean}} EditorConfig */

/** @param {unknown} error */
const message = error => error instanceof Error ? error.message : 'The request failed.';
/** @template {Element} T @param {ParentNode} root @param {string} selector @returns {T} */
function element(root, selector) {
  const found = root.querySelector(selector);
  if (!found) throw new Error(`Missing editor control: ${selector}`);
  return /** @type {T} */ (found);
}

export class ClipEditor {
  /** @param {EditorConfig} config @param {(results:GenerationResult[])=>void} watch @param {(busy:boolean)=>void} busyChanged @param {(run:SequenceRun)=>void} [sequenceChanged] */
  constructor(config, watch, busyChanged, sequenceChanged=()=>{}) {
    this.config=config; this.watch=watch; this.busyChanged=busyChanged;
    this.sequenceChanged=sequenceChanged;
    this.refreshError='';
    this.batchStarting=false; this.reordering=false;this.importing=false;
    /** @type {SequenceRun|null} */ this.run=null;
    /** @type {SequenceRequest|null} */ this.pending=null;
    try {this.pending=JSON.parse(sessionStorage.getItem('bundle-request')||'null');} catch {sessionStorage.removeItem('bundle-request');}
    /** @type {Map<string, ClipCard>} */ this.cards=new Map();
    /** @type {HTMLElement} */ this.list=element(document,'#clip-cards');
    /** @type {HTMLButtonElement} */ this.add=element(document,'#add-clip');
    /** @type {HTMLInputElement} */ this.importFile=element(document,'#clip-prompt-file');
    /** @type {HTMLSelectElement} */ this.importMode=element(document,'#clip-import-mode');
    /** @type {HTMLInputElement} */ this.bulkFrames=element(document,'#clip-frame-files');
    /** @type {HTMLElement} */ this.frameImportStatus=element(document,'#frame-import-status');
    /** @type {HTMLButtonElement} */ this.undoImport=element(document,'#undo-import');
    /** @type {HTMLButtonElement} */ this.loadPreset=element(document,'#load-preset');
    this.loadPreset.hidden=!config.presetAvailable;
    this.undoImport.hidden=!config.canUndoImport;
    /** @type {HTMLElement} */ this.error=element(document,'#clip-editor-error');
    /** @type {HTMLButtonElement} */ this.runButton=element(document,'#generate-sequence');
    /** @type {HTMLButtonElement} */ this.cancelButton=element(document,'#cancel-sequence');
    /** @type {HTMLElement} */ this.runStatus=element(document,'#sequence-status');
    /** @type {HTMLElement} */ this.summary=element(document,'#sequence-summary');
    /** @type {HTMLAnchorElement} */ this.finalDownload=element(document,'#final-download');
    /** @type {HTMLButtonElement} */ this.finalWatch=element(document,'#watch-final');
    /** @type {HTMLDialogElement} */ this.dialog=element(document,'#frame-dialog');
    /** @type {HTMLImageElement} */ this.fullImage=element(this.dialog,'img');
    /** @type {ReturnType<typeof setTimeout>|undefined} */ this.timer=undefined;
    this.add.addEventListener('click',()=>void this.addClip());
    this.importFile.addEventListener('change',()=>void this.importPrompts());
    this.bulkFrames.addEventListener('change',()=>void this.importFrames());
    this.undoImport.addEventListener('click',()=>void this.restoreImport());
    this.loadPreset.addEventListener('click',()=>void this.restorePreset());
    this.runButton.addEventListener('click',()=>void this.generateSequence());
    this.cancelButton.addEventListener('click',()=>void this.cancelSequence());
    this.finalWatch.addEventListener('click',()=>this.watchFinal());
    element(this.dialog,'button').addEventListener('click',()=>this.dialog.close());
  }
  async init() {
    try {
      /** @type {ClipRecord[]} */ const records=await api('/api/clips');
      if (!records.length) records.push(await api('/api/clips',jsonRequest({starter:true})));
      records.forEach(record=>this.append(record));
      /** @type {SequenceRun[]} */ const runs=(await api('/api/sequences'));
      const bundles=runs.filter(run=>run.mode==='bundles');
      this.run=bundles.find(run=>!['completed','failed','cancelled','interrupted'].includes(run.status))||bundles.find(run=>run.id===(this.pending?.id||sessionStorage.getItem('bundle-run-id')))||bundles[0]||null;
      if(this.run?.id===this.pending?.id) {this.pending=null;sessionStorage.removeItem('bundle-request');}
      this.update(); this.poll();
    } catch(error) { this.showError(message(error)); }
  }
  /** @param {string} text */ showError(text) { this.error.textContent=text; this.error.hidden=!text; }
  /** @param {ClipRecord} record */ append(record) {
    const card=new ClipCard(this,record); this.cards.set(record.id,card); this.list.append(card.root); this.update();
  }
  async addClip() {
    this.add.disabled=true;
    try { /** @type {ClipRecord} */ const record=await api('/api/clips',jsonRequest({})); this.append(record); this.showError(''); }
    catch(error) { this.showError(message(error)); }
    finally { this.update(); }
  }
  update() {
    const cards=[...this.cards.values()], locked=this.editingLocked();
    this.add.disabled=locked||cards.length>=this.config.maxClips;
    this.importFile.disabled=locked||cards.some(c=>isGenerating(c.record)||c.uploading||c.posting);
    this.importMode.disabled=this.importFile.disabled;this.undoImport.disabled=this.importFile.disabled;
    this.loadPreset.disabled=this.importFile.disabled;
    this.bulkFrames.disabled=this.importFile.disabled||!cards.some(c=>c.record.initialFrameFile||c.record.endFrameFile);
    cards.forEach((card,order)=>{card.record.order=order;card.title.textContent=`Clip ${order+1}`;card.paint();});
    const busy=cards.some(card=>isGenerating(card.record)||card.posting||card.uploading);
    this.runButton.disabled=this.importing||this.batchStarting||this.reordering||busy||!this.config.configured||!cards.length||this.activeRun();
    this.runButton.textContent=this.pending?'Reconnect Sequence':'Generate Sequence';
    this.cancelButton.hidden=!this.activeRun();this.cancelButton.disabled=this.batchStarting;
    this.runStatus.textContent=this.run?`Last run: ${this.run.status[0].toUpperCase()+this.run.status.slice(1)} · ${this.run.clips.filter(c=>c.status==='completed').length}/${this.run.clips.length} clips completed${this.run.error?` · ${this.run.error}`:''}${this.run.warning?` · ${this.run.warning}`:''}`:'Ready · clips generate and stitch in the order shown.';
    const final=Boolean(this.run?.status==='completed'&&this.run.finalVideoUrl);
    this.finalDownload.hidden=!final;this.finalWatch.hidden=!final;
    if(this.run) this.finalDownload.href=`/api/sequences/${this.run.id}/download`;
    this.summary.replaceChildren(...cards.map((card,order)=>{
      const item=document.createElement('div');item.className='sequence-item';
      const label=document.createElement('span');label.textContent=`${order?'→ ':''}Clip ${order+1}`;item.append(label);
      for(const field of /** @type {const} */(['firstFrame','endFrame'])) {
        const ref=card.record[field];if(ref) {const image=document.createElement('img');image.src=ref.previewUrl;image.alt=`Clip ${order+1} ${field==='firstFrame'?'initial':'end'} frame`;item.append(image);}
      }
      return item;
    }));
    const expected=new Set(cards.flatMap(c=>[c.record.initialFrameFile,c.record.endFrameFile].filter(Boolean)));
    const remaining=cards.filter(c=>missingFrameMessage(c.record));
    if(!this.importing) this.frameImportStatus.textContent=expected.size?`${cards.length} clips · ${expected.size} distinct frame filenames · ${remaining.length?`${remaining.length} clips waiting for frames`:'all required frames attached'}. Files match by name, regardless of selection order.`:'For clips JSON with frame filenames, select all matching images together. Other clips use the upload controls inside each card.';
    this.busyChanged(busy||this.activeRun());
    if(this.run) this.sequenceChanged(this.run);
  }
  activeRun() {return Boolean(this.run&&!['completed','failed','cancelled','interrupted'].includes(this.run.status));}
  editingLocked() {return this.importing||this.batchStarting||this.reordering||this.activeRun()||Boolean(this.pending)||Boolean(this.run?.generationBusy&&this.run.status==='cancelled');}
  /** @param {ClipCard} card @param {number} delta */
  async move(card,delta) {
    if(this.editingLocked()) return;
    const cards=[...this.cards.values()], from=cards.indexOf(card), to=from+delta;
    if(to<0||to>=cards.length) return;
    this.reordering=true;this.update();
    try {
      [cards[from],cards[to]]=[cards[to],cards[from]];
      /** @type {ClipRecord[]} */ const records=await api('/api/clips/order',jsonRequest({clipIds:cards.map(c=>c.record.id)}));
      this.cards=new Map(records.map(remote=>{const existing=this.cards.get(remote.id);if(!existing) throw new Error('Sequence changed; reload the page.');return [remote.id,existing];}));
      this.list.replaceChildren(...[...this.cards.values()].map(c=>c.root));
    } catch(error) {card.showError(message(error));}
    finally {this.reordering=false;this.update();}
  }
  async validateAndSave() {
    let valid=true;
    for(const card of this.cards.values()) {
      try {card.validate();await card.save();}
      catch(error) {card.showError(`Clip ${card.record.order+1}: ${message(error)}`);valid=false;}
    }
    return valid;
  }
  async generateSequence() {
    if(this.batchStarting||this.activeRun()) return;
    this.batchStarting=true;this.update();
    try {
      if(!this.pending) {
        if(!await this.validateAndSave()) return;
        this.pending=sequenceFor([...this.cards.values()].map(c=>c.record),crypto.randomUUID());
        sessionStorage.setItem('bundle-request',JSON.stringify(this.pending));
      }
      /** @type {SequenceRun} */ const run=await api('/api/sequences',jsonRequest(this.pending));
      this.run=run;sessionStorage.setItem('bundle-run-id',run.id);
      this.pending=null;sessionStorage.removeItem('bundle-request');this.showError('');
      for(const clip of run.clips) {
        const card=this.cards.get(clip.id);if(card) {card.record.status=clip.status;card.record.jobId=clip.jobId;card.record.result=clip.result;}
      }
    } catch(error) {
      if(error instanceof Error && 'clipErrors' in error && error.clipErrors && typeof error.clipErrors==='object') {
        for(const [id,detail] of Object.entries(error.clipErrors)) this.cards.get(id)?.showError(`Clip ${(this.cards.get(id)?.record.order??0)+1}: ${String(detail)}`);
      }
      if(error instanceof Error && 'status' in error) {this.pending=null;sessionStorage.removeItem('bundle-request');}
      this.showError(message(error)+(this.pending?' Reconnect Sequence recovers this exact batch without submitting a second run.':''));
    } finally {this.batchStarting=false;this.update();}
  }
  async cancelSequence() {
    if(!this.run) return;
    try {this.run=await api(`/api/sequences/${this.run.id}/cancel`,jsonRequest({}));}
    catch(error) {this.showError(message(error));}
    this.update();
  }
  watchFinal() {
    if(!this.run?.finalVideoUrl) return;
    // Use the existing player; the final MP4 is one already-assembled result.
    this.watch([{id:this.run.id,status:'completed',model:'stitched sequence',prompt:'',seed:null,duration:this.run.clips.reduce((sum,c)=>sum+c.duration,0),resolution:'',video:{url:this.run.finalVideoUrl}}]);
  }
  async importPrompts() {
    const file=this.importFile.files?.[0]; if (!file) return;
    this.importing=true;this.update();
    try {
      if(file.size>100000) throw new Error('Prompt file must be at most 100 KB.');
      await Promise.all([...this.cards.values()].filter(c=>c.revision>c.savedRevision).map(c=>c.save()));
      /** @type {ClipRecord[]} */ const records=await api('/api/clips/import',jsonRequest({text:(await file.text()).replace(/^\uFEFF/,''),mode:this.importMode.value}));
      this.replaceCards(records);this.undoImport.hidden=false;this.showError('');
    } catch(error) { this.showError(message(error)); }
    finally { this.importing=false;this.importFile.value=''; this.update(); }
  }
  /** @param {ClipRecord[]} records */ replaceCards(records) {
    for(const card of this.cards.values()) card.dispose();
    this.cards.clear();this.list.replaceChildren();records.forEach(record=>this.append(record));
  }
  async restoreImport() {
    this.importing=true;this.update();
    try {/** @type {ClipRecord[]} */ const records=await api('/api/clips/import/undo',jsonRequest({}));this.replaceCards(records);this.undoImport.hidden=true;this.showError('');}
    catch(error) {this.showError(message(error));}
    finally {this.importing=false;this.update();}
  }
  async restorePreset() {
    this.importing=true;this.update();
    try {
      await Promise.all([...this.cards.values()].filter(c=>c.revision>c.savedRevision).map(c=>c.save()));
      /** @type {ClipRecord[]} */ const records=await api('/api/clips/preset',jsonRequest({}));
      this.replaceCards(records);this.undoImport.hidden=false;this.showError('');
    } catch(error) {this.showError(message(error));}
    finally {this.importing=false;this.update();}
  }
  async importFrames() {
    const files=Array.from(this.bulkFrames.files||[]);if(!files.length) return;
    this.importing=true;this.frameImportStatus.textContent=`Uploading and matching ${files.length} frame files…`;this.update();
    try {
      if(files.length>this.config.maxClips*2) throw new Error(`Select at most ${this.config.maxClips*2} frame files.`);
      for(const file of files) {
        if(file.size>this.config.maxImageBytes) throw new Error(`${file.name}: each image must be at most 10 MB.`);
        if(!['image/png','image/jpeg','image/webp'].includes(file.type)) throw new Error(`${file.name}: use a PNG, JPEG or WebP image.`);
      }
      await Promise.all([...this.cards.values()].map(c=>c.save()));
      const body=new FormData();files.forEach(file=>body.append('frames',file));
      /** @type {ClipRecord[]} */ const records=await api('/api/clips/frames',{method:'POST',body});
      this.replaceCards(records);this.showError('');
    } catch(error) {
      if(error instanceof Error && 'clipErrors' in error && error.clipErrors && typeof error.clipErrors==='object') {
        for(const [id,detail] of Object.entries(error.clipErrors)) this.cards.get(id)?.showError(`Clip ${(this.cards.get(id)?.record.order??0)+1}: ${String(detail)}`);
      }
      this.showError(message(error));
    } finally {this.importing=false;this.bulkFrames.value='';this.update();}
  }
  poll() {
    clearTimeout(this.timer);
    this.timer=setTimeout(async()=>{
      try {
        /** @type {ClipRecord[]} */ const records=await api('/api/clips');
        if(!this.importing) for (const remote of records) this.cards.get(remote.id)?.refresh(remote);
        if(this.run) this.run=await api(`/api/sequences/${this.run.id}`);
        if(this.refreshError && this.error.textContent===this.refreshError) this.showError('');
        this.refreshError='';
      } catch(error) {
        this.refreshError=`Cannot refresh clip status: ${message(error)}. No generation was resubmitted.`;
        this.showError(this.refreshError);
      }
      this.update(); this.poll();
    },800);
  }
  /** @param {ReferenceFrame} reference */ preview(reference) {
    this.fullImage.src=reference.previewUrl; this.fullImage.alt=reference.name; this.dialog.showModal();
  }
}

class ClipCard {
  /** @param {ClipEditor} editor @param {ClipRecord} record */
  constructor(editor, record) {
    this.editor=editor; this.record=record; this.revision=0; this.savedRevision=0;
    this.posting=false; this.uploading=false; this.disposed=false;
    /** @type {Partial<Record<'firstFrame'|'endFrame',string>>} */ this.uploadErrors={};
    /** @type {Promise<void>|null} */ this.savePromise=null;
    /** @type {ReturnType<typeof setTimeout>|undefined} */ this.saveTimer=undefined;
    this.token=sessionStorage.getItem(`clip-token:${record.id}`);
    this.root=document.createElement('article'); this.root.className='clip-card'; this.root.dataset.clipId=record.id;
    this.root.innerHTML=`
      <div class="clip-card-heading"><h3></h3><span class="clip-mode"></span><span class="clip-source"></span><div class="clip-tools"><button type="button" class="secondary move-up">↑ Move Up</button><button type="button" class="secondary move-down">↓ Move Down</button><button type="button" class="secondary duplicate">Duplicate</button><button type="button" class="secondary remove-card">Remove</button></div></div>
      <fieldset><label>Prompt<textarea class="clip-prompt" rows="4" maxlength="8000" placeholder="Describe this clip…"></textarea></label>
      <div class="clip-controls"><label>Seed<div class="seed-control"><input class="clip-seed" type="text" inputmode="numeric" spellcheck="false" aria-label="Seed"><button type="button" class="secondary randomize">Randomize Seed</button></div></label>
      <label>Prompt Expansion<select class="clip-expansion"><option value="disabled">Disabled</option><option value="balanced">Balanced</option><option value="quality">Quality</option></select></label>
      <label>Duration<select class="clip-duration"></select></label><label>Resolution<select class="clip-resolution"><option value="480P">480p</option><option value="768P">768p</option><option value="1080P">1080p</option></select></label></div>
      <p class="frame-heading">Reference Frames <span>Optional · PNG, JPEG or WebP · 10 MB each</span></p><div class="clip-frames">
      <div class="frame-field first-frame"><label>Initial Frame<input class="frame-upload" type="file" accept="image/png,image/jpeg,image/webp"></label><div class="frame-content" hidden><button class="frame-thumb" type="button" aria-label="View initial frame"><img alt="Initial frame preview"></button><div><span class="frame-name"></span><button class="secondary remove-frame" type="button">Remove initial frame</button></div></div><p class="frame-note">No initial frame · Text to Video</p></div>
      <div class="frame-field end-frame"><label>End Frame<input class="frame-upload" type="file" accept="image/png,image/jpeg,image/webp"></label><div class="frame-content" hidden><button class="frame-thumb" type="button" aria-label="View end frame"><img alt="End frame preview"></button><div><span class="frame-name"></span><button class="secondary remove-frame" type="button">Remove end frame</button></div></div><p class="frame-note">Add an initial frame before adding an end frame.</p></div>
      </div><button type="button" class="secondary previous-frame" hidden>← Use Previous End Frame</button></fieldset>
      <div class="clip-actions"><button type="button" class="generate">Generate</button><button type="button" class="secondary cancel-clip" hidden>Cancel</button><span class="clip-state" role="status" aria-live="polite"></span><span class="save-state"></span></div>
      <p class="clip-error" role="alert" hidden></p><div class="clip-result" hidden><div class="clip-result-actions"><button class="secondary watch" type="button">Watch clip</button><a class="download">Download MP4</a><span class="actual-seed"></span></div><details><summary>Generation metadata</summary><pre class="clip-metadata"></pre></details></div>`;
    /** @type {HTMLElement} */ this.title=element(this.root,'h3');
    /** @type {HTMLFieldSetElement} */ this.fields=element(this.root,'fieldset');
    /** @type {HTMLTextAreaElement} */ this.prompt=element(this.root,'.clip-prompt');
    /** @type {HTMLInputElement} */ this.seed=element(this.root,'.clip-seed');
    /** @type {HTMLSelectElement} */ this.expansion=element(this.root,'.clip-expansion');
    /** @type {HTMLSelectElement} */ this.duration=element(this.root,'.clip-duration');
    /** @type {HTMLSelectElement} */ this.resolution=element(this.root,'.clip-resolution');
    /** @type {HTMLButtonElement} */ this.generateButton=element(this.root,'.generate');
    /** @type {HTMLElement} */ this.status=element(this.root,'.clip-state');
    /** @type {HTMLElement} */ this.saved=element(this.root,'.save-state');
    /** @type {HTMLElement} */ this.error=element(this.root,'.clip-error');
    this.prompt.value=record.prompt; this.seed.value=String(record.seed); this.expansion.value=record.promptExpansionMode;
    this.duration.replaceChildren(...editor.config.durations.map(n=>new Option(`${n}s`,String(n))));
    this.duration.value=String(record.duration); this.resolution.value=record.resolution;
    for(const control of [this.prompt,this.seed]) control.addEventListener('input',()=>this.changed());
    for(const control of [this.expansion,this.duration,this.resolution]) control.addEventListener('change',()=>this.changed());
    element(this.root,'.randomize').addEventListener('click',()=>{this.seed.value=String(randomSeed()); this.changed(); void this.save().catch(error=>this.showError(message(error)));});
    this.generateButton.addEventListener('click',()=>void this.generate());
    element(this.root,'.duplicate').addEventListener('click',()=>void this.duplicate());
    element(this.root,'.remove-card').addEventListener('click',()=>void this.remove());
    element(this.root,'.move-up').addEventListener('click',()=>void this.editor.move(this,-1));
    element(this.root,'.move-down').addEventListener('click',()=>void this.editor.move(this,1));
    element(this.root,'.previous-frame').addEventListener('click',()=>{
      const previous=[...this.editor.cards.values()][this.record.order-1]?.record.endFrame;
      if(previous&&!this.record.firstFrame) {this.record.firstFrame=previous;delete this.uploadErrors.firstFrame;this.changed();void this.save().catch(error=>this.showError(message(error)));}
    });
    element(this.root,'.cancel-clip').addEventListener('click',()=>void this.cancel());
    element(this.root,'.watch').addEventListener('click',()=>{if(this.record.result?.video) this.editor.watch([this.record.result]);});
    this.wireFrame('firstFrame','.first-frame'); this.wireFrame('endFrame','.end-frame');
    this.saved.textContent='Saved'; this.paint();
  }
  dispose() { this.disposed=true; clearTimeout(this.saveTimer); }
  /** @param {'firstFrame'|'endFrame'} field @param {string} selector */
  wireFrame(field, selector) {
    const box=element(this.root,selector);
    /** @type {HTMLInputElement} */ const input=element(box,'.frame-upload');
    input.addEventListener('change',()=>{const file=input.files?.[0]; if(file) void this.upload(field,file,input);});
    element(box,'.remove-frame').addEventListener('click',()=>{
      this.record[field]=null; input.value='';
      delete this.uploadErrors[field];
      if(field==='firstFrame') {this.record.endFrame=null;delete this.uploadErrors.endFrame; /** @type {HTMLInputElement} */(element(this.root,'.end-frame input')).value='';}
      this.changed(); this.paint(); void this.save().catch(error=>this.showError(message(error)));
    });
    element(box,'.frame-thumb').addEventListener('click',()=>{const ref=this.record[field]; if(ref) this.editor.preview(ref);});
  }
  /** @param {string} text */ showError(text) {this.error.textContent=text; this.error.hidden=!text;}
  readControls() {
    this.record.prompt=this.prompt.value; this.record.seed=parseSeed(this.seed.value);
    this.record.promptExpansionMode=/** @type {PromptExpansionMode} */(this.expansion.value);
    this.record.duration=Number(this.duration.value);
    this.record.resolution=/** @type {'480P'|'768P'|'1080P'} */(this.resolution.value);
  }
  validate() {
    this.readControls();
    if(this.uploading) throw new Error('Wait for the image upload to finish.');
    const failed=Object.values(this.uploadErrors)[0];if(failed) throw new Error(failed);
    if(!this.record.prompt.trim()) throw new Error('Prompt is empty.');
    const missing=missingFrameMessage(this.record);if(missing) throw new Error(missing);
    if(this.record.endFrame&&!this.record.firstFrame) throw new Error('End frame cannot be used without an initial frame.');
  }
  changed() {
    clearTimeout(this.saveTimer); this.revision++; this.saved.textContent='Unsaved changes';
    try {this.readControls(); this.showError('');}
    catch(error) {this.showError(message(error));return;}
    this.saveTimer=setTimeout(()=>void this.save().catch(error=>this.showError(message(error))),350);
    this.editor.update();
  }
  /** @returns {Promise<void>} */
  async save() {
    clearTimeout(this.saveTimer); this.readControls();
    if(this.savePromise) {await this.savePromise; if(this.revision>this.savedRevision) return this.save(); return;}
    if(this.revision<=this.savedRevision) return;
    this.savePromise=(async()=>{
      while(this.revision>this.savedRevision && !this.disposed) {
        const revision=this.revision; this.saved.textContent='Saving…';
        /** @type {ClipRecord} */ const remote=await api(`/api/clips/${this.record.id}`,{...jsonRequest(settingsFor(this.record)),method:'PATCH'});
        this.savedRevision=revision;
        if(revision===this.revision) {this.record=remote; this.saved.textContent='Saved';this.paint();}
      }
    })();
    try {await this.savePromise;} catch(error) {this.saved.textContent='Not saved';throw error;} finally {this.savePromise=null;}
  }
  /** @param {'firstFrame'|'endFrame'} field @param {File} file @param {HTMLInputElement} input */
  async upload(field,file,input) {
    this.uploading=true; this.status.textContent=`Uploading ${field==='firstFrame'?'initial':'end'} frame…`; this.editor.update();
    try {
      if(file.size>this.editor.config.maxImageBytes) throw new Error('Each image must be at most 10 MB.');
      if(!['image/png','image/jpeg','image/webp'].includes(file.type)) throw new Error('Use a PNG, JPEG, or WebP image.');
      if(field==='endFrame'&&!this.record.firstFrame) throw new Error('Add an initial frame before adding an end frame.');
      const body=new FormData(); body.append('image',file);
      /** @type {ReferenceFrame} */ const reference=await api('/api/images',{method:'POST',body});
      if(!reference.id||!reference.previewUrl) throw new Error('Image upload returned no usable reference. Try again.');
      this.record[field]=reference; delete this.uploadErrors[field];this.changed(); await this.save();
    } catch(error) {this.uploadErrors[field]=`${field==='firstFrame'?'Initial':'End'} frame upload failed: ${message(error)}`;this.showError(this.uploadErrors[field]||'Upload failed.');}
    finally {this.uploading=false; input.value=''; this.editor.update();}
  }
  /** @param {ClipRecord} remote */ refresh(remote) {
    // Poll only the outcome; do not overwrite text or seed being edited locally.
    if(this.posting||this.uploading||this.savePromise||this.revision>this.savedRevision) return;
    this.record.status=remote.status; this.record.result=remote.result; this.record.jobId=remote.jobId;
    this.record.jobIds=remote.jobIds;this.record.error=remote.error;
    this.record.initialFrameFile=remote.initialFrameFile;this.record.endFrameFile=remote.endFrameFile;
    for(const field of /** @type {const} */(['firstFrame','endFrame'])) {
      if(this.record[field]?.id===remote[field]?.id) this.record[field]=remote[field];
    }
    if(remote.result && this.token && remote.result.id===remote.jobId && isGenerating(remote)) {
      sessionStorage.removeItem(`clip-token:${this.record.id}`); this.token=null;
    }
    this.paint();
  }
  async generate() {
    if(this.posting||this.uploading||isGenerating(this.record)||this.editor.editingLocked()) return;
    this.posting=true; this.paint(); this.editor.update();
    try {
      this.validate(); await this.save();
      this.token ||= crypto.randomUUID(); sessionStorage.setItem(`clip-token:${this.record.id}`,this.token);
      /** @type {GenerationResult} */ const job=await api(`/api/clips/${this.record.id}/generate`,jsonRequest({token:this.token}));
      this.record.result=job;this.record.jobId=job.id;this.record.status=job.status;
      sessionStorage.removeItem(`clip-token:${this.record.id}`);this.token=null;this.showError('');
    } catch(error) {
      if(error instanceof Error && 'status' in error) {this.token=null;sessionStorage.removeItem(`clip-token:${this.record.id}`);}
      this.showError(message(error)+(this.token?' Use Reconnect to recover this same submission.':''));
    } finally {this.posting=false;this.paint();this.editor.update();}
  }
  async duplicate() {
    try {await this.save(); /** @type {ClipRecord} */ const record=await api(`/api/clips/${this.record.id}/duplicate`,jsonRequest({}));this.editor.append(record);}
    catch(error) {this.showError(message(error));}
  }
  async remove() {
    try {await this.save();await api(`/api/clips/${this.record.id}`,{method:'DELETE'});this.dispose();this.root.remove();this.editor.cards.delete(this.record.id);this.editor.update();}
    catch(error) {this.showError(message(error));}
  }
  async cancel() {
    try {/** @type {ClipRecord} */ const remote=await api(`/api/clips/${this.record.id}/cancel`,jsonRequest({}));this.refresh(remote);}
    catch(error) {this.showError(message(error));}
  }
  paint() {
    const generating=isGenerating(this.record), locked=generating||this.posting||this.uploading||this.editor.editingLocked();
    this.fields.disabled=locked;
    this.generateButton.disabled=locked||!this.editor.config.configured;
    this.generateButton.textContent=this.token?'Reconnect':this.record.jobId?'Regenerate':'Generate';
    /** @type {HTMLButtonElement} */(element(this.root,'.duplicate')).disabled=locked||this.editor.cards.size>=this.editor.config.maxClips;
    /** @type {HTMLButtonElement} */(element(this.root,'.remove-card')).disabled=locked;
    /** @type {HTMLButtonElement} */(element(this.root,'.move-up')).disabled=locked||this.record.order===0;
    /** @type {HTMLButtonElement} */(element(this.root,'.move-down')).disabled=locked||this.record.order===this.editor.cards.size-1;
    /** @type {HTMLButtonElement} */ const previous=element(this.root,'.previous-frame');
    previous.hidden=this.record.order===0;
    previous.disabled=locked||Boolean(this.record.firstFrame)||![...this.editor.cards.values()][this.record.order-1]?.record.endFrame;
    previous.title=this.record.firstFrame?'Remove your initial frame first to use the previous end frame.':'Use the previous clip’s selected end-frame asset.';
    /** @type {HTMLButtonElement} */ const cancel=element(this.root,'.cancel-clip');cancel.hidden=!generating;cancel.disabled=this.posting;
    element(this.root,'.clip-mode').textContent=this.record.firstFrame?'Image to Video':this.record.initialFrameFile?'Image to Video · waiting for frame':'Text to Video';
    element(this.root,'.clip-source').textContent=this.record.sourceId||'';
    if(!this.uploading) this.status.textContent=this.posting?'Starting…':this.record.status==='queued'?'Queued':this.record.status==='generating'?'Generating…':this.record.status==='uploading'?'Uploading image…':this.record.status==='submitting'?'Submitting…':this.record.status==='ready'?(missingFrameMessage(this.record)?'Waiting for frames':'Ready'):this.record.status[0].toUpperCase()+this.record.status.slice(1);
    for(const [field,selector] of /** @type {const} */([['firstFrame','.first-frame'],['endFrame','.end-frame']])) {
      const box=element(this.root,selector), ref=this.record[field];
      /** @type {HTMLInputElement} */(element(box,'.frame-upload')).disabled=locked||(field==='endFrame'&&!this.record.firstFrame);
      /** @type {HTMLElement} */(element(box,'.frame-content')).hidden=!ref;
      if(ref) {/** @type {HTMLImageElement} */(element(box,'img')).src=ref.previewUrl;element(box,'.frame-name').textContent=ref.name;}
      const expected=field==='firstFrame'?this.record.initialFrameFile:this.record.endFrameFile;
      element(box,'.frame-note').textContent=ref?`Attached: ${ref.name}. Choose a file to replace this image.`:expected?`Required file: ${expected}${field==='endFrame'&&!this.record.firstFrame?' · upload all frames together or add the initial frame first.':''}`:field==='endFrame'&&!this.record.firstFrame?'Add an initial frame before adding an end frame.':field==='firstFrame'?'No initial frame · Text to Video':'Optional last frame';
    }
    const job=this.record.result, completed=job?.status==='completed'&&Boolean(job.video);
    /** @type {HTMLElement} */ const result=element(this.root,'.clip-result');
    result.hidden=!job;
    /** @type {HTMLButtonElement} */(element(result,'.watch')).hidden=!completed;
    /** @type {HTMLAnchorElement} */ const download=element(result,'.download');download.hidden=!completed;
    if(job) {
      download.href=`/api/jobs/${job.id}/download`;
      element(result,'.actual-seed').textContent=`Seed used${job.seedSource==='submitted'?' (submitted)':''}: ${job.seed ?? 'Unknown (older generation)'}`;
      const payload=job.generationInput;
      element(result,'.clip-metadata').textContent=JSON.stringify({model:job.model,prompt:payload?.prompt??job.prompt,
        seedUsed:job.seed,seedSource:job.seedSource??(job.seed==null?'unknown':'submitted'),requestedSeed:payload?.seed,promptExpansionMode:payload?.prompt_expansion_mode??job.promptExpansionMode??'disabled',
        initialFrameUsed:payload?.image_url??null,endFrameUsed:payload?.end_image_url??null,
        duration:payload?.duration??job.duration,resolution:payload?.resolution??job.resolution,
        requestId:job.requestId??null,jobId:job.id,expandedPrompt:job.expandedPrompt??null,
        ...(this.record.legacyMetadata&&!payload?{note:'This older run did not retain all generation inputs. Re-upload its frames before reproducing it.'}:{})},null,2);
      if(job.error && this.record.status==='failed') this.showError(job.error);
      else if(job.connectionWarning) this.showError(job.connectionWarning);
    }
  }
}
