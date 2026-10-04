import { setupTracePanel } from './decision-trace.js';
import { overlayDiagnostic } from './tracking-diagnostics.js';
import { museProgress } from './muse-status.js';
import { currentObservation } from './adaptive-observation.js';
import { trackerValue, trackingSummary } from './adaptive-tracker.js';
import { PlaybackQueue } from './queue.js';
import { videoContentRect, ScreenPointMapping, boxesAt } from './adaptive-geometry.js';
const $ = id => document.getElementById(id);
const COLORS = ['#e07a5f', '#3d85c6', '#81b29a', '#f2cc8f'];
let video = $('video');
const videos = [video, $('video-next')];
const canvas = $('overlay-canvas'), ctx = canvas.getContext('2d');
let state = null, sessionId = null, playingIndex = null, waitingFor = 0, sound = false, lastChanges = '', running = false;
let providerConfigured = false, submitting = false;
let clockOffset = 0, tickPending = false, pendingStop = null;
let endingIndex = null;
let playbackEpoch = 0, presented = null, lastTickAt = 0, playerMode = 'idle';
const screenMapping = new ScreenPointMapping();
let lastPresentedClip = null;
const ending = new Set();
setupTracePanel($('trace-panel'), () => sessionId);
const windowKey = () => [window.screenX, window.screenY, window.innerWidth, window.innerHeight, devicePixelRatio, screen.width, screen.height, !!document.fullscreenElement].join(':');
window.addEventListener('pointermove', e => {
  screenMapping.observe(e, windowKey());
});

async function request(path, options) {
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),8000);
  let r;try {r=await fetch(path,{...options,signal:controller.signal});}finally {clearTimeout(timeout);}
  const value = await r.json();
  if (!r.ok) throw new Error(value.error || 'Request failed.');
  return value;
}
const post = (path, body) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const error = message => { $('error').textContent = message || ''; $('error').hidden = !message; };
const color = name => COLORS[Math.max(0, (state?.session?.names || []).indexOf(name)) % COLORS.length];
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// -- setup form ---------------------------------------------------------------
function addCharacter(name = '', description = '') {
  const rows = $('characters').children.length; if (rows >= 4) return;
  const row = el('div', 'row');
  const n = el('input'); n.placeholder = `Character ${String.fromCharCode(65 + rows)}`; n.value = name; n.maxLength = 40; n.required = true;
  const d = el('input'); d.placeholder = 'Look (e.g. green octopus with a long nose)'; d.value = description; d.maxLength = 200;
  row.append(n, d); $('characters').append(row);
}
$('add-character').addEventListener('click', () => addCharacter());
function syncSequenceSettings() {
  $('duration').disabled = $('resolution').disabled = $('use-sequence').checked;
}
$('use-sequence').addEventListener('change', syncSequenceSettings);
$('setup').addEventListener('submit', async event => {
  event.preventDefault(); error('');
  if (state?.setup && !state.setup.generationReady) { error(state.setup.error || state.setup.message); return; }
  const characters = [...$('characters').children].map(r => ({ name: r.children[0].value.trim(), description: r.children[1].value.trim() })).filter(c => c.name);
  const body = new FormData(), file = $('opening').files[0];
  if (file) body.append(file.type.startsWith('video/') ? 'opening' : 'start', file);
  body.append('use_saved_sequence', $('use-sequence').checked ? '1' : '0');
  body.append('premise', $('premise').value);
  body.append('timeline', $('timeline').value);
  body.append('tracker', trackerValue($('tracker').value));
  body.append('playback_mode', $('stream-playback').checked ? 'stream' : 'download');
  body.append('eegRunMode', $('eeg-run-mode').value);
  body.append('characters', JSON.stringify(characters)); body.append('duration', $('duration').value); body.append('resolution', $('resolution').value);
  body.append('objects', JSON.stringify($('objects').value.split(',').map(n => n.trim()).filter(Boolean).map(n => ({name:n, description:n}))));
  submitting = true; $('start').disabled = true;
  try { await request('/api/adaptive/sessions', { method: 'POST', body }); }
  catch (e) { error(e.message); }
  finally { submitting = false; $('start').disabled = !providerConfigured || !state?.setup?.generationReady; }
});
for (const [id,recenter] of [['gaze-check',false],['gaze-recenter',true]]) $(id).addEventListener('click',async()=>{try {await post('/api/sensors/gaze-check',{recenter});}catch(e){error(e.message);}});
let eegCalibrationPending = false;
for (const [id, action] of [['eeg-calibration-start','start_or_recalibrate'],['eeg-gaze-only','gaze_only']]) {
  $(id).addEventListener('click', async () => {
    if (eegCalibrationPending) return;
    eegCalibrationPending = true; $(id).disabled = true;
    try { await post('/api/adaptive/muse/calibration',{action}); }
    catch (e) { error(e.message); }
    finally { eegCalibrationPending = false; }
  });
}
$('sensor-retry').addEventListener('click', async () => {
  try { await request('/api/sensors/setup', {method:'POST'}); } catch (e) { error(e.message); }
});
$('muse-connect').addEventListener('click', async () => {
  $('muse-connect').disabled = true;
  try { await post('/api/adaptive/muse/connect', {}); } catch (e) { error(e.message); }
});
$('muse-disconnect').addEventListener('click', async () => {
  $('muse-disconnect').disabled = true;
  try { await post('/api/adaptive/muse/disconnect', {}); } catch (e) { error(e.message); }
});
$('gaze-calibration-remove').addEventListener('click', async () => {
  try { await request('/api/sensors/gaze-calibration', {method:'DELETE'}); } catch (e) { error(e.message); }
});
$('stop').addEventListener('click', async () => {
  player.stop(); playingIndex = null; endingIndex = null; ending.clear();
  try { render({ ...state, session: await post('/api/adaptive/stop', {session_id:sessionId}) }); } catch (e) { error(e.message); }
});
$('fullscreen').addEventListener('click', () => {
  const change = document.fullscreenElement ? document.exitFullscreen() : $('screen').requestFullscreen();
  change.catch(e => error(e.message));
});
$('sound').addEventListener('click', () => { sound = !sound; player.setSound(sound); $('sound').textContent = sound ? 'Mute sound' : 'Enable sound'; });

// -- player -------------------------------------------------------------------
function contentRect() {
  return videoContentRect(video.getBoundingClientRect(), video.videoWidth, video.videoHeight, getComputedStyle(video).objectFit);
}
function screenRect() {
  const c = contentRect(); if (!c || !screenMapping.valid(windowKey())) return null;
  const convert = r => screenMapping.rect(r,windowKey());
  const r = video.getBoundingClientRect();
  const left = Math.max(c.left,r.left,0), top = Math.max(c.top,r.top,0);
  const right = Math.min(c.left+c.w,r.right,innerWidth), bottom = Math.min(c.top+c.h,r.bottom,innerHeight);
  return {rect:convert(c), visible_rect:convert({left,top,w:Math.max(0,right-left),h:Math.max(0,bottom-top)})};
}
function transitionDiagnostic(diagnostic) {
  const s = state?.session, sid = sessionId;
  if (!s || s.id !== sid) return;
  const source = diagnostic.previousIndex == null ? null : s.clips[diagnostic.previousIndex];
  const target = s.clips[diagnostic.index], anchor = target || source;
  if (!anchor) return;
  const body = {session_id:sid,clip:anchor.index,clip_id:anchor.id,kind:'diagnostic',wall:Date.now()+clockOffset,
    diagnostic:{...diagnostic,sourceClipId:source?.id ?? null,targetClipId:target?.id ?? null}};
  // Capture identity/time now. A late response must never relabel another run.
  void post('/api/adaptive/playback-event',body).catch(() => {});
}
async function playbackEvent(kind, v) {
  const clip = state?.session?.clips[v.clipIndex], sid = sessionId;
  if (!clip) return;
  try { await post('/api/adaptive/playback-event', {session_id:sid, clip:v.clipIndex, clip_id:clip.id, kind, wall:Date.now()+clockOffset}); }
  catch { /* Polling retains connection diagnostics. */ }
}
for (const v of videos) {
  v.addEventListener('ended', () => {
    if (v !== player.current || !sessionId) return;
    ending.add(v.clipIndex); void playbackEvent('ended', v); playingIndex = null; presented = null;
  });
  v.addEventListener('canplay', () => void playbackEvent('ready', v));
  v.addEventListener('seeking', () => { if (v === video) { playbackEpoch++; presented=null; void reportTick(false); } });
  v.addEventListener('pause', () => { if (v === video) void reportTick(false); });
  v.addEventListener('waiting', () => { if (v === video) { void reportTick(false); void playbackEvent('waiting',v); } });
  if ('requestVideoFrameCallback' in v) {
    const frame = (now, meta) => {
      if (v === player.current) {
        presented = {time:meta.mediaTime, wall:performance.timeOrigin+meta.expectedDisplayTime+clockOffset, at:performance.now()};
        void reportTick(true);
      }
      v.requestVideoFrameCallback(frame);
    };
    v.requestVideoFrameCallback(frame);
  }
}
const player = new PlaybackQueue(videos, event => {
  playerMode = event.state;
  if (event.state === 'playing') {
    const changedVideo = video !== player.current;
    video = player.current; playingIndex = event.index; waitingFor = event.index;
    $('overlay').hidden = true;
    const key = `${sessionId}:${event.index}`;
    if (lastPresentedClip !== key || changedVideo) { lastPresentedClip=key; playbackEpoch++; presented=null; void playbackEvent('playing',video); }
  } else if (event.state === 'gesture') {
    $('overlay').hidden=false; $('overlay').textContent='Click the video to start playback.';
  } else if (['buffering','loading','stopped','error','finished','cancelled'].includes(event.state)) {
    if (!player.current) playingIndex=null;
    $('overlay').hidden=false;
    $('overlay').textContent = event.state === 'finished' ? 'Story finished.' : event.state === 'cancelled' ? 'Stopped by viewer.' :
      event.state === 'error' ? 'Media could not play. Stop and restart to retry.' : event.state === 'stopped' ? 'Generation failed · holding last frame. Stop and restart when the provider is available.' :
      event.index ? 'BRIDGE · holding the last frame while the next generated scene becomes ready. No new viewer evidence is collected.' : 'Preparing the opening scene…';
  }
}, transitionDiagnostic);
$('screen').addEventListener('click', () => { if (player.autoplayBlocked) void player.tryPlay(); });
function managePlayer(s) {
  if (s.status === 'stopped') { if (!player.stopped) player.stop(); return; }
  player.update(s.clips.map(c => c.url && ['ready','playing','watched'].includes(c.status) ? c : null), s.status === 'finished' ? 'completed' : s.status);
}
async function reportTick(playing) {
  if (playingIndex == null || !state?.session || (playing && (tickPending || performance.now()-lastTickAt < 80))) return;
  const mapping = screenRect() || {rect:null,visible_rect:null};
  const sid=sessionId, index=playingIndex, clip=state.session.clips[index];
  const fresh=presented && performance.now()-presented.at < 250;
  const hasFrames='requestVideoFrameCallback' in video;
  const active=playing && !video.paused && !video.seeking && !video.ended && video.readyState>=3 && (!hasFrames || fresh);
  const payload={session_id:sid,clip:index,clip_id:clip.id,epoch:playbackEpoch,
      video_t:fresh && active ? presented.time : video.currentTime,playing:!!active,playback_rate:video.playbackRate,
      ...mapping,mapping: {...screenMapping.audit(windowKey()),devicePixelRatio,screenWidth:screen.width,screenHeight:screen.height},wall:fresh && active ? presented.wall : Date.now()+clockOffset};
  if (tickPending) { if (!playing && !pendingStop) pendingStop=payload; return; }
  lastTickAt=performance.now(); await sendCapturedTick(payload);
}
async function sendCapturedTick(payload) {
  tickPending=true;
  try {await post('/api/adaptive/tick',payload);}
  catch(e) {$('backend-status').textContent=`Playback sync: ${e.message}`;}
  finally {tickPending=false;if(pendingStop){const next=pendingStop;pendingStop=null;void sendCapturedTick(next);}}
}
setInterval(() => { if (!('requestVideoFrameCallback' in video) || !presented || performance.now()-presented.at > 300) void reportTick(!('requestVideoFrameCallback' in video)); }, 150);

// -- overlay ------------------------------------------------------------------
let lastOverlayDiagnosticAt = 0, overlayDiagnosticPending = false;
function draw() {
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * devicePixelRatio) || canvas.height !== Math.round(r.height * devicePixelRatio)) { canvas.width = Math.round(r.width * devicePixelRatio); canvas.height = Math.round(r.height * devicePixelRatio); }
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
  const c = contentRect(), s = state?.session;
  if (s && playingIndex != null && s.clips[playingIndex] && s.status === 'running' && !overlayDiagnosticPending && Date.now()-lastOverlayDiagnosticAt >= 1000) {
      lastOverlayDiagnosticAt=Date.now(); overlayDiagnosticPending=true;
      const clip=s.clips[playingIndex], vr=video.getBoundingClientRect();
      const diagnostic=overlayDiagnostic({clip,mediaTime:video.currentTime,enabled:$('show-boxes').checked,
        content:c,canvas:r,video:vr,active:clip?.status!=='watched',dpr:devicePixelRatio});
      void post('/api/adaptive/overlay-diagnostic',{session_id:sessionId,clip_id:clip?.id,
        wall:Date.now()+clockOffset,diagnostic}).catch(()=>{}).finally(()=>{overlayDiagnosticPending=false;});
    }
  if (c && s && playingIndex != null) {
    ctx.save(); const vr=video.getBoundingClientRect(); ctx.beginPath(); ctx.rect(vr.left-r.left,vr.top-r.top,vr.width,vr.height);ctx.clip();
    const ox = c.left - r.left, oy = c.top - r.top;
    const currentPoint=state.gaze.point;
    const matchingPoint=currentObservation(currentPoint,sessionId,s.clips[playingIndex]?.id,(Date.now()+clockOffset)/1000);
    const target = matchingPoint?.target;
    if ($('show-boxes').checked) {
      for (const [name, b] of Object.entries(boxesAt(s.clips[playingIndex]?.track, video.currentTime))) {
        ctx.strokeStyle = color(name); ctx.lineWidth = name === target ? 4 : 2;
        ctx.strokeRect(ox + b[0] * c.w, oy + b[1] * c.h, (b[2] - b[0]) * c.w, (b[3] - b[1]) * c.h);
        ctx.fillStyle = color(name); ctx.font = '600 13px system-ui'; ctx.fillText(name, ox + b[0] * c.w + 6, oy + b[1] * c.h + 17);
      }
    }
    const g = matchingPoint;
    if ($('show-gaze').checked && g && g.valid && g.on_video && Date.now()+clockOffset-g.t*1000 < 500) {
      ctx.beginPath(); ctx.arc(ox + g.nx * c.w, oy + g.ny * c.h, 11, 0, 2 * Math.PI);
      ctx.strokeStyle = g.target ? '#baf5ce' : '#ffffff'; ctx.lineWidth = 3; ctx.stroke();
    }
    ctx.restore();
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// -- dashboard ----------------------------------------------------------------
function pill(id, text, kind) { $(id).textContent = text; $(id).className = `pill ${kind}`; }
function bar(label, value, { signed = false, flash = false, swatch } = {}) {
  const row = el('div', `bar${flash ? ' flash' : ''}`), name = el('span', null, label), track = el('div', 'track'), fill = el('div', `fill${signed ? ' signed' : ''}`);
  if (signed) { const mid = el('div', 'mid'); track.append(mid); fill.style.left = `${50 + Math.min(0, value) * 50}%`; fill.style.width = `${Math.abs(value) * 50}%`; }
  else { fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`; fill.style.background = swatch || '#baecd0'; }
  track.append(fill); row.append(name, track, el('span', 'val', signed ? (value >= 0 ? '+' : '') + value.toFixed(2) : value.toFixed(2)));
  return row;
}
function drawEeg(series) {
  const c = $('eeg-chart'), g = c.getContext('2d'), w = c.width, h = c.height;
  g.clearRect(0, 0, w, h); g.strokeStyle = '#2c313a'; g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
  if (!series.length) return;
  g.strokeStyle = '#abedc5'; g.lineWidth = 2; g.beginPath();
  series.forEach((p, i) => { const x = w + (p.t / 30) * w, y = h / 2 - Math.max(-3, Math.min(3, p.z)) / 3 * (h / 2 - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}
function render(st) {
  state = st; const s = st.session;
  const setup = st.setup;
  $('sensor-setup').hidden = !setup?.required && !setup?.enabled && setup?.eegMode !== 'muse';
  $('sensor-message').textContent = (setup?.error || setup?.message || '') +
    (setup && (setup.muse.source === 'mindmonitor' || setup.phase === 'calibrating_muse') ? ` (${museProgress(setup.muse)})` : '');
  const museState = setup?.museConnection || {state:'disconnected'};
  $('muse-connection-status').textContent = museState.state === 'error' ? `Muse connection error: ${museState.error || 'Unknown error'}` :
    museState.state === 'connecting' ? 'Connecting to Muse 2…' : museState.state === 'connected' ? 'Muse 2 connected.' : 'Muse 2 disconnected. Gaze and playback remain available.';
  $('muse-connect').hidden = !setup || setup.eegMode !== 'muse';
  $('muse-connect').textContent = museState.state === 'error' ? 'Retry Muse 2' : 'Connect Muse 2';
  $('muse-disconnect').hidden = !setup || !['connecting','connected','error'].includes(museState.state);
  $('muse-connect').disabled = submitting || museState.state === 'connecting' || museState.state === 'connected';
  $('muse-disconnect').disabled = submitting || museState.state === 'disconnected';
  $('sensor-retry').hidden = !setup?.canRetry; $('sensor-retry').disabled = s?.status === 'running';
  const eegCalibration = setup?.eegCalibration;
  $('eeg-calibration-progress').value = eegCalibration?.cleanSeconds || 0;
  $('eeg-calibration-status').textContent = eegCalibration ?
    `${eegCalibration.gazeOnly ? 'Gaze-only selected · ' : ''}${eegCalibration.ready ? 'EEG baseline ready' : eegCalibration.state} · ${eegCalibration.cleanSeconds.toFixed(1)} / 60 unique clean seconds · ${eegCalibration.reason}` : 'EEG calibration status unavailable.';
  $('eeg-calibration-start').hidden = !eegCalibration?.supported;
  $('eeg-calibration-start').textContent = eegCalibration?.ready ? 'Recalibrate for a different wearer or setup' : 'Start / restart 60-second EEG calibration';
  $('eeg-calibration-start').disabled = eegCalibrationPending || submitting || s?.status === 'running' || !setup?.muse?.live;
  $('eeg-gaze-only').disabled = eegCalibrationPending || submitting || s?.status === 'running' || !!eegCalibration?.gazeOnly;
  $('gaze-calibration-status').textContent = `Eye calibration: ${setup?.gaze?.calibrationState || 'required'}${setup?.gaze?.failureReason ? ' · '+setup.gaze.failureReason : ''}`;
  $('calibration-details').textContent=[setup?.gaze?.calibrationPath,setup?.gaze?.checkCommand,setup?.gaze?.checkGuidance].filter(Boolean).join('\n');
  $('gaze-calibration-remove').hidden = !setup?.gaze?.savedCalibration;
  for (const id of ['gaze-check','gaze-recenter']) {$(id).hidden=!setup?.gaze?.savedCalibration;$(id).disabled=submitting || s?.status==='running';}
  $('gaze-calibration-remove').disabled = submitting || s?.status === 'running';
  $('start').disabled = submitting || !providerConfigured || !setup?.generationReady;
  const gz = st.gaze, ee = st.eeg;
  pill('gaze-source', gz.error ? 'port busy' : gz.source === 'sim' ? 'SIM' : gz.live ? 'gazekit live' : 'no gaze', gz.source === 'sim' ? 'sim' : gz.live ? 'live' : 'off');
  pill('eeg-source', ee.source === 'sim' ? 'SIM' : ['muse', 'mindmonitor'].includes(ee.source) ? (ee.connectionState || (ee.live ? 'streaming' : 'disconnected')).replaceAll('_',' ') : 'off', ee.source === 'sim' ? 'sim' : ee.live ? 'live' : 'off');
  $('eeg-state').textContent = (ee.source === 'mindmonitor' ? `${ee.state} · smoothed α/β ${ee.alphaBetaRatio?.toFixed(2) ?? '—'}` : `${ee.state} · relative β/(α+θ)`) + ` · signal quality ${Math.round(100 * (ee.confidence || 0))}%`;
  $('eeg-quality').textContent = (!ee.live || !(ee.confidence > 0) ? 'EEG unavailable — gaze-only mode · ' : '') + (ee.qualityError || (ee.selectedChannels?.length ? `Clean channels: ${ee.selectedChannels.join(', ')}${ee.qualityWarning ? ' · '+ee.qualityWarning : ''}` : ee.goodChannels ? `Good channels: ${ee.goodChannels.join(', ')}` : ''));
  $('gaze-hz').textContent = gz.hz ?? '—'; $('blinks').textContent = gz.blinks_per_min ?? '—'; $('yaw').textContent = gz.yaw != null ? Math.round(gz.yaw) : '—';
  const point=currentObservation(gz.point,sessionId,s?.clips[playingIndex]?.id,(Date.now()+clockOffset)/1000);
  const target = point?.valid ? point.target : null, tg = $('gaze-target'); tg.replaceChildren();
  if (target) { const sw = el('span', 'swatch'); sw.style.background = color(target); tg.append(sw, target); }
  else tg.textContent = !point ? (s?.status !== 'running' ? 'No active video frame' : 'Tracking unavailable / stale') : !point.valid ? 'Tracking invalid' : !point.on_video ? 'Outside video' : point.state === 'ambiguous' ? 'Ambiguous target' : point.state === 'unavailable' ? 'Attribution unavailable' : 'Background';
  $('dwell').textContent = (st.response?.characters?.[target]?.dwell_s || 0).toFixed(1);
  $('response-strength').textContent = `${Math.round(100 * (st.response?.response_strength || 0))}%`;
  $('gaze-confidence').textContent = `${Math.round(100 * (st.response?.gaze_confidence || 0))}%`;
  const z = ee.live && ee.confidence > 0 ? ee.series.at(-1)?.z : null; $('eeg-z').textContent = z != null ? `${z >= 0 ? '+' : ''}${z.toFixed(2)}${ee.source === 'mindmonitor' ? '' : 'σ'}` : '—'; drawEeg(ee.series);
  const yoloOption=$('tracker').querySelector('option[value="yoloe"]');
  const yolo=st.trackerAvailability?.yoloe || s?.trackerAvailability?.yoloe;
  if(yoloOption) {yoloOption.disabled=!yolo?.available;yoloOption.textContent=yolo?.available ? 'YOLOE visual reference experiment' : `YOLOE experiment (${yolo?.reason || 'runtime unavailable'})`;}
  if (!s) {
    if (sessionId) player.reset();
    sessionId = null; playingIndex = null; endingIndex = null;
    $('setup').hidden = false; $('stop').disabled = true;
    $('stage').textContent = 'Idle.';
    $('tracker').disabled = false;
    $('tracking-status').textContent = trackingSummary(null,null,0);
    return;
  }
  if (s.id !== sessionId) {
    sessionId = s.id; playingIndex = null; endingIndex = null; ending.clear(); playbackEpoch++; presented=null; player.reset(); player.setSound(sound);
    waitingFor = s.clips.findIndex(c => c.status !== 'watched');
    if (waitingFor < 0) waitingFor = s.clips.length;
    lastChanges = ''; player.next=waitingFor;
  }
  running = s.status === 'running'; $('stop').disabled = !running; $('setup').hidden = running;
  $('stage').textContent = `${s.demo ? 'DEMO · ' : st.simulation ? 'SIMULATED SENSORS · LIVE VIDEO · ' : ''}${s.status} · ${s.stage}${s.error ? ` · ${s.error}` : ''}`;
  $('warnings').textContent = s.warnings?.map(w => `${w.component}: ${w.message}`).join('\n') || '';
  $('warnings').hidden = !s.warnings?.length;
  $('story-state').textContent = s.story.current_event || s.story.scenes.at(-1)?.summary || s.story.premise;
  $('profile-clips').textContent = `${s.profile.clips} scene${s.profile.clips === 1 ? '' : 's'} measured`;
  const lastChange = [...s.clips].reverse().find(c => c.profileChanges?.length)?.profileChanges || [...s.clips].reverse().find(c => c.changes?.length)?.changes || [];
  const changedKeys = new Set(lastChange.map(c => c.key));
  $('affinity').replaceChildren(el('h4', null, 'Observed attention'), ...Object.entries(s.profile.characters).map(([n, v]) => bar(n, v, { swatch: color(n), flash: changedKeys.has(`character:${n}`) })));
  $('prefs').replaceChildren(el('h4', null, 'Tentative preferences'), bar('pacing', s.profile.pacing, { signed: true, flash: changedKeys.has('pacing') }),
    bar('dialogue', s.profile.dialogue, { signed: true, flash: changedKeys.has('dialogue') }),
    ...Object.entries(s.profile.genres).filter(([, v]) => v !== 0).map(([g, v]) => bar(g, v, { signed: true, flash: changedKeys.has(`genre:${g}`) })));
  const key = JSON.stringify([lastChange,s.profile.clips > 0]);
  if (key !== lastChanges) {
    lastChanges = key;
    $('changes').replaceChildren(...(lastChange.length ? lastChange.map(c => {
      const li = el('li', c.strong ? 'strong' : null);
      li.append(el('span', null, `${c.key.replace('character:', '')} `), el('span', 'delta', `${c.before.toFixed(2)} → ${c.after.toFixed(2)}`), el('div', 'hint', c.why + (c.strong ? ' · CONSISTENT ATTENTION' : '')));
      return li;
    }) : [el('li', 'hint', s.profile.clips ? 'No preference change: observations have not met the adaptation thresholds.' : 'Nothing yet: watch the first scene.')]));
  }
  const latest = s.clips.at(-1);
  const current=s.clips[playingIndex ?? s.playing];
  $('tracker').disabled = running;
  if (running) $('tracker').value = trackerValue(s.tracker);
  $('tracking-status').textContent = trackingSummary(s,current,video.currentTime);
  const currentChange=current?.decision.focus || (current?.decision.pacing !== 'same' && current?.decision.pacing ? `${current.decision.pacing} pacing` : 'balanced');
  $('current-decision').textContent=current ? `${running ? 'Playing' : 'Last played'} scene ${current.index+1} · decision ${current.decisionId || 'opening'} · ${currentChange}` : 'No clip playing';
  $('queue-state').textContent=`${playerMode} · ${s.clips.filter(c=>c.status==='ready').length} media ready · future queue limit 1`;
  $('latency').textContent=s.latency?.samples ? `Readiness ${s.latency.readinessS.map(t=>t.toFixed(1)+'s').join(', ')} (n=${s.latency.samples}); adaptation freezes at ${Math.min(3.5,current?.duration ?? s.duration).toFixed(1)}s of playback. Full-clip tracking runs independently.` : 'Waiting for measured readiness.';
  $('latency').textContent += s.playbackMode === 'stream' ? ` Delivery: ${current?.playbackDelivery === 'stream' ? 'progressive MP4' : 'validated download'}; local copy ${current?.localMediaStatus || 'pending'}.` : ' Delivery: fully validated download.';
  $('mapping-state').textContent=screenMapping.valid(windowKey()) ? `Screen mapping measured (scale ${screenMapping.transform.scale.toFixed(3)}) · eye calibration remains a separate check.` : 'Move the pointer diagonally across the player to measure screen mapping. Eye calibration is checked separately.';
  if (latest) {
    const d = latest.decision;
    $('decision').replaceChildren(el('span', `chip${d.focus ? ' on' : ''}`, `focus: ${d.focus || 'balanced'}`), el('span', `chip${d.tension !== 'same' ? ' on' : ''}`, `tension: ${d.tension}`),
      el('span', `chip${d.dialogue !== 'same' ? ' on' : ''}`, `dialogue: ${d.dialogue}`), el('span', `chip${d.pacing !== 'same' ? ' on' : ''}`, `pacing: ${d.pacing}`),
      el('span', `chip${d.tone ? ' on' : ''}`, `tone: ${d.tone || 'same'}`), el('span', `chip${d.event ? ' on' : ''}`, `new event: ${d.event ? 'yes' : 'no'}`));
    $('reasons').replaceChildren(...d.reasons.map(r => el('li', null, r)));
    $('change-note').textContent = latest.fallbackReason || latest.plan.change_note; $('writer').textContent = latest.fallback ? 'HELD FRAME · FAILED' : latest.writer;
    $('next-title').textContent = `Scene ${latest.index + 1} · decision ${latest.decisionId?.slice(0,8)}: ${latest.plan.scene_title}`;
    $('prompt-changes').textContent=JSON.stringify(latest.plan.prompt_changes || [],null,2);
    $('request-payload').textContent=JSON.stringify(latest.decisionTrace ? {submission:latest.decisionTrace.submission, promptExact:latest.decisionTrace.promptExact, changed:latest.decisionTrace.changed} : {status:'No recorded submission'},null,2); $('next-prompt').textContent = latest.plan.video_prompt;
  }
  $('scenes').replaceChildren(...s.clips.map(c => {
    const d = el('div', `scene${c.index === playingIndex ? ' current' : ''}`);
    d.append(el('b', null, `${c.index + 1}. ${c.plan.scene_title}`), el('span', null, `${c.status}${c.decision.focus ? ` · focus ${c.decision.focus}` : ''}${c.generatedS ? ` · ${c.generatedS}s to make` : ''}`));
    return d;
  }));
  managePlayer(s);
}
async function poll() {
  try {
    const before = Date.now(), value = await request('/api/adaptive/state');
    clockOffset = value.serverNow * 1000 - (before + Date.now()) / 2;
    if (value.session?.id === sessionId) for (const index of [...ending]) {
      await post('/api/adaptive/ended', {session_id:sessionId,clip:index}); ending.delete(index);
    }
    render(value); $('backend-status').textContent = 'Python backend · connected';
  }
  catch (e) { $('backend-status').textContent = `Backend: ${e.message}`; }
  setTimeout(poll, 200);
}
async function init() {
  addCharacter('SpongeBob', 'yellow rectangular sponge wearing a white shirt and brown square pants');
  addCharacter('Patrick', 'pink starfish wearing green shorts with purple flowers');
  $('premise').value = 'In Bikini Bottom, SpongeBob wants to discover what Patrick keeps inside his secret box.';
  try {
    const config = await request('/api/config');
    providerConfigured = config.configured;
    $('duration').replaceChildren(...config.durations.map(n => new Option(String(n), String(n)))); $('duration').value = '10';
    syncSequenceSettings();
    $('cost-note').textContent = config.demo ? 'DEMO mode: synthetic clips, no Fal calls.' : config.configured
      ? 'Each scene is a paid Fal generation. Decisions run locally. Scene frames and character references go to the configured visual services; webcam images stay local.' : 'Fal key missing: set FAL_KEY in .env and restart.';
  } catch (e) { error(e.message); }
  $('start').disabled = !providerConfigured || !state?.setup?.generationReady;
  void poll();
}
void init();
