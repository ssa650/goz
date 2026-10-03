// @ts-check
/** @typedef {'disabled'|'balanced'|'quality'} PromptExpansionMode */
/** @typedef {{id:string, name:string, previewUrl:string, contentType:string, size:number, providerUrl:string|null}} ReferenceFrame */
/** @typedef {{prompt:string, seed:number, firstFrame:ReferenceFrame|null, endFrame:ReferenceFrame|null, promptExpansionMode:PromptExpansionMode, duration:number, resolution:'480P'|'768P'|'1080P'}} ClipGenerationSettings */
/** @typedef {{id:string, status:string, model:string, prompt:string, seed:number|null, seedSource?:'provider'|'submitted'|'unknown', promptExpansionMode?:PromptExpansionMode, duration:number, resolution:string, requestId?:string, firstFrame?:string|null, endFrame?:string|null, error?:string|null, connectionWarning?:string|null, video?:{url:string}, generationInput?:{prompt:string, seed?:number, prompt_expansion_mode:PromptExpansionMode, duration:number, resolution:string, image_url?:string, end_image_url?:string}, expandedPrompt?:string|null, totalElapsedMs?:number, apiReadyMs?:number, apiElapsedMs?:number|null, uploadElapsedMs?:number, requestUncertain?:boolean}} GenerationResult */
/** @typedef {ClipGenerationSettings & {id:string, order:number, status:string, jobId:string|null, jobIds:string[], result:GenerationResult|null, error?:string|null, legacyMetadata?:boolean,sourceId?:string|null,initialFrameFile?:string|null,endFrameFile?:string|null}} ClipRecord */
/** @typedef {Omit<ClipGenerationSettings,'firstFrame'|'endFrame'> & {firstFrame:string|null,endFrame:string|null}} ClipWriteSettings */
/** @typedef {ClipWriteSettings & {id:string,order:number}} ClipBundle */
/** @typedef {{id:string,clips:ClipBundle[]}} SequenceRequest */
/** @typedef {{id:string,mode:'bundles',status:string,clips:(ClipBundle & {status:string,jobId:string,generatedVideoUrl:string|null,result:GenerationResult|null,error?:string|null})[],finalVideoUrl:string|null,error?:string,warning?:string,generationBusy:boolean}} SequenceRun */

/** Array order is the editor order. Copy complete bundles, never independent lists.
 * @param {ClipRecord[]} clips @param {string} id @returns {SequenceRequest} */
export function sequenceFor(clips,id) {
  return {id,clips:clips.map((clip,order)=>({id:clip.id,order,...settingsFor(clip)}))};
}

/** @param {ClipRecord} clip @returns {string|null} */
export function missingFrameMessage(clip) {
  const missing=[];
  if(clip.initialFrameFile&&!clip.firstFrame) missing.push(`Initial frame ${clip.initialFrameFile} is missing.`);
  if(clip.endFrameFile&&!clip.endFrame) missing.push(`End frame ${clip.endFrameFile} is missing.`);
  return missing.length?`${missing.join(' ')} Upload matching frame files.`:null;
}

/** Randomize only on an explicit button click. @returns {number} */
export function randomSeed() {
  return crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
}

/** @param {string} value @returns {number} */
export function parseSeed(value) {
  if (!/^-?\d+$/.test(value.trim()) || !Number.isSafeInteger(Number(value))) {
    throw new Error('Seed must be an integer between −9007199254740991 and 9007199254740991.');
  }
  return Number(value);
}

/** @param {ClipGenerationSettings} clip @returns {ClipWriteSettings} */
export function settingsFor(clip) {
  return {prompt:clip.prompt, seed:clip.seed, firstFrame:clip.firstFrame?.id || null,
    endFrame:clip.endFrame?.id || null, promptExpansionMode:clip.promptExpansionMode,
    duration:clip.duration, resolution:clip.resolution};
}

/** @param {ClipRecord} clip @returns {boolean} */
export function isGenerating(clip) {
  return ['uploading','submitting','queued','generating'].includes(clip.status);
}

/** @template T @param {string} path @param {RequestInit} [options] @returns {Promise<T>} */
export async function api(path, options) {
  const response = await fetch(path, options);
  /** @type {unknown} */
  const value = await response.json();
  if (!response.ok) {
    const message = value && typeof value === 'object' && 'error' in value && typeof value.error === 'string' ? value.error : 'The request failed.';
    const clipErrors=value && typeof value==='object' && 'clipErrors' in value ? value.clipErrors : undefined;
    throw Object.assign(new Error(message), {status:response.status,clipErrors});
  }
  return /** @type {T} */ (value);
}

/** @param {unknown} value @returns {RequestInit} */
export function jsonRequest(value) {
  return {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(value)};
}
