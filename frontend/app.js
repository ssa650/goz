// @ts-check
import { PlaybackQueue } from './queue.js';
import { SequencePlayback } from './sequence-playback.js';
import { SequenceController, formatTime, runError } from './sequence-controller.js';
/** @template {HTMLElement} T @param {string} id @returns {T} */
function element(id) {
  const found=document.getElementById(id);
  if(!found) throw new Error(`Missing player element ${id}`);
  return /** @type {T} */(found);
}
/** @type {HTMLVideoElement[]} */ const videos=[element('video-a'),element('video-b')];
/** @type {HTMLButtonElement} */ const generate=element('generate');
/** @type {HTMLButtonElement} */ const regenerate=element('regenerate');
/** @type {HTMLButtonElement} */ const download=element('download');
/** @type {HTMLButtonElement} */ const fullscreen=element('fullscreen');
const player=new PlaybackQueue(videos,({state,index})=>{
  const messages={playing:`Playing clip ${index+1}.`,loading:`Loading clip ${index+1}…`,buffering:`Waiting for clip ${index+1} to finish generating…`,
    finished:'Sequence finished.',cancelled:'Playback paused.',stopped:'Generation stopped. See the error below.',
    gesture:'Click Fullscreen to start playback.',error:'This clip could not play. Try refreshing the page.'};
  element('playback-status').textContent=messages[state];
  element('overlay').hidden=state==='playing';element('overlay-message').textContent=messages[state];
});
player.setSound(true);
for(const video of videos) video.volume=1;
const playback=new SequencePlayback(player,()=>true);
const screen=element('screen');
// Fullscreen the container, so browser video controls stay absent across swaps.
screen.addEventListener('contextmenu',event=>event.preventDefault());
fullscreen.addEventListener('click',async()=>{
  // Also recover blocked autoplay from this explicit user gesture.
  void player.tryPlay();
  try {
    if(document.fullscreenElement===screen) await document.exitFullscreen();
    else await screen.requestFullscreen();
  } catch(error) {
    element('playback-status').textContent=`Fullscreen is unavailable: ${error instanceof Error?error.message:String(error)}`;
  }
});
/** @param {SequenceController} controller */
function render(controller) {
  const {run,config}=controller,completed=run?.clips.filter(c=>c.status==='completed').length||0;
  element('backend-status').textContent=config?(config.demo?'Local demo':'Connected'):'Disconnected';
  generate.disabled=!config?.configured||controller.busy;
  regenerate.disabled=!config?.configured||controller.busy||!run;
  download.disabled=!run?.finalVideoUrl||run.status!=='completed';
  generate.textContent=controller.starting?'Starting…':controller.pending?'Reconnect':'Generate';
  const status=!config?'Connecting to the backend…':!config.configured?'Set FAL_KEY in .env, then restart the app.'
    :controller.starting?'Starting your sequence…':!run?`${controller.clips.length} clips ready.`
    :run.status==='completed'?`${completed}/${run.clips.length} clips completed · Your video is ready to download.`
    :run.status==='stitching'?`${completed}/${run.clips.length} clips completed · Stitching your download…`
    :`${run.status.charAt(0).toUpperCase()+run.status.slice(1)} · ${completed}/${run.clips.length} clips completed`;
  element('generation-status').textContent=status;
  const error=controller.error||controller.connectionError||runError(run);
  element('generation-error').textContent=error;element('generation-error').hidden=!error;
  element('timing-summary').textContent=run?`${completed}/${run.clips.length}`:'';
  const clips=run?[...run.clips].sort((a,b)=>a.order-b.order):controller.clips;
  element('timing-rows').replaceChildren(...clips.map(clip=>{
    const row=document.createElement('tr'),result=clip.result;
    const values=[`Clip ${clip.order+1}`,clip.status,
      formatTime(result?.totalElapsedMs),formatTime(result?.apiReadyMs??result?.apiElapsedMs)];
    for(const value of values) {const cell=document.createElement('td');cell.textContent=value;row.append(cell);}
    const error=clip.error||result?.error;
    if(error) {const cell=document.createElement('td');cell.className='timing-error';cell.textContent=error;cell.colSpan=4;
      const detail=document.createElement('tr');detail.append(cell);return [row,detail];}
    return [row];
  }).flat());
}
const controller=new SequenceController(render,run=>playback.update(run));
generate.addEventListener('click',()=>void controller.generate());
regenerate.addEventListener('click',()=>void controller.generate(true));
download.addEventListener('click',()=>{
  if(!controller.run?.finalVideoUrl||controller.run.status!=='completed') return;
  const link=document.createElement('a');link.href=`/api/sequences/${controller.run.id}/download`;
  link.download='final_video.mp4';link.click();
});
async function poll() {
  await controller.refresh();
  window.setTimeout(()=>void poll(),Math.max(controller.config?.pollMs||500,800));
}
await controller.init();void poll();
