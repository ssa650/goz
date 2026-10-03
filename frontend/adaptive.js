const $ = id => document.getElementById(id);
const COLORS = ['#e07a5f', '#3d85c6', '#81b29a', '#f2cc8f'];
const video = $('video'), canvas = $('overlay-canvas'), ctx = canvas.getContext('2d');
let state = null, sessionId = null, playingIndex = null, waitingFor = 0, sound = false, lastChanges = '', running = false;

async function request(path, options) {
  const r = await fetch(path, options); const value = await r.json();
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
  const characters = [...$('characters').children].map(r => ({ name: r.children[0].value.trim(), description: r.children[1].value.trim() })).filter(c => c.name);
  const body = new FormData(), file = $('opening').files[0];
  if (file) body.append(file.type.startsWith('video/') ? 'opening' : 'start', file);
  body.append('use_saved_sequence', $('use-sequence').checked ? '1' : '0');
  body.append('premise', $('premise').value);
  body.append('timeline', $('timeline').value);
  body.append('characters', JSON.stringify(characters)); body.append('duration', $('duration').value); body.append('resolution', $('resolution').value);
  $('start').disabled = true;
  try { await request('/api/adaptive/sessions', { method: 'POST', body }); }
  catch (e) { error(e.message); }
  finally { $('start').disabled = false; }
});
$('stop').addEventListener('click', async () => { try { await post('/api/adaptive/stop', {}); } catch (e) { error(e.message); } });
$('sound').addEventListener('click', () => { sound = !sound; video.muted = !sound; $('sound').textContent = sound ? 'Mute sound' : 'Enable sound'; });

// -- player -------------------------------------------------------------------
function contentRect() {
  const r = video.getBoundingClientRect();
  if (!video.videoWidth || !r.width) return null;
  const scale = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
  const w = video.videoWidth * scale, h = video.videoHeight * scale;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, w, h, el: r };
}
function screenRect() {
  const c = contentRect(); if (!c) return null;
  const chrome = Math.max(0, window.outerHeight - window.innerHeight);
  return { x: window.screenX + c.left, y: window.screenY + chrome + c.top, w: c.w, h: c.h };
}
function playClip(index) {
  const clip = state.session.clips[index];
  playingIndex = index; video.src = clip.url; video.muted = !sound;
  video.play().catch(() => { $('overlay').hidden = false; $('overlay').textContent = 'Click the video to start playback.'; });
}
video.addEventListener('playing', () => { $('overlay').hidden = true; });
video.addEventListener('ended', () => {
  if (playingIndex == null) return;
  void post('/api/adaptive/ended', { clip: playingIndex }).catch(() => {});
  waitingFor = playingIndex + 1; playingIndex = null;
});
$('screen').addEventListener('click', () => { if (video.src && video.paused && playingIndex != null) video.play(); });
function managePlayer(s) {
  if (playingIndex != null) return;
  const next = s.clips[waitingFor];
  if (next && next.status !== 'generating' && next.url) { playClip(waitingFor); return; }
  $('overlay').hidden = false;
  $('overlay').textContent = s.status === 'running'
    ? (waitingFor === 0 ? `Generating the opening scene… (${s.stage})` : `Generating scene ${waitingFor + 1}, adapted to you… (${s.stage})`)
    : s.status === 'failed' ? `Stopped: ${s.error}` : 'Story finished.';
}
setInterval(() => {
  if (playingIndex == null || !state?.session) return;
  void post('/api/adaptive/tick', { clip: playingIndex, video_t: video.currentTime, playing: !video.paused && !video.ended,
    rect: screenRect(), wall: Date.now() }).catch(() => {});
}, 100);

// -- overlay ------------------------------------------------------------------
function boxesAt(track, t) {
  if (!track?.length) return {};
  let best = track[0];
  for (const f of track) if (Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
  return Math.abs(best.t - t) <= 0.27 ? best.boxes : {};
}
function draw() {
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * devicePixelRatio)) { canvas.width = Math.round(r.width * devicePixelRatio); canvas.height = Math.round(r.height * devicePixelRatio); }
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
  const c = contentRect(), s = state?.session;
  if (c && s && playingIndex != null) {
    const ox = c.left - r.left, oy = c.top - r.top;
    const target = state.gaze.point?.target;
    if ($('show-boxes').checked) {
      for (const [name, b] of Object.entries(boxesAt(s.clips[playingIndex]?.track, video.currentTime))) {
        ctx.strokeStyle = color(name); ctx.lineWidth = name === target ? 4 : 2;
        ctx.strokeRect(ox + b[0] * c.w, oy + b[1] * c.h, (b[2] - b[0]) * c.w, (b[3] - b[1]) * c.h);
        ctx.fillStyle = color(name); ctx.font = '600 13px system-ui'; ctx.fillText(name, ox + b[0] * c.w + 6, oy + b[1] * c.h + 17);
      }
    }
    const g = state.gaze.point;
    if ($('show-gaze').checked && g && g.on_video) {
      ctx.beginPath(); ctx.arc(ox + g.nx * c.w, oy + g.ny * c.h, 11, 0, 2 * Math.PI);
      ctx.strokeStyle = g.valid ? '#ffffff' : '#777'; ctx.lineWidth = 3; ctx.stroke();
    }
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
  const gz = st.gaze, ee = st.eeg;
  pill('gaze-source', gz.error ? 'port busy' : gz.source === 'sim' ? 'SIM' : gz.live ? 'gazekit live' : 'no gaze', gz.source === 'sim' ? 'sim' : gz.live ? 'live' : 'off');
  pill('eeg-source', ee.source === 'sim' ? 'SIM' : ee.source === 'muse' ? (ee.live ? 'Muse 2 live' : 'Muse 2 …') : 'off', ee.source === 'sim' ? 'sim' : ee.live ? 'live' : 'off');
  $('eeg-state').textContent = `${ee.state} · β/(α+θ), z vs. rolling 60 s baseline`;
  $('gaze-hz').textContent = gz.hz ?? '—'; $('blinks').textContent = gz.blinks_per_min ?? '—'; $('yaw').textContent = gz.yaw != null ? Math.round(gz.yaw) : '—';
  const target = gz.point?.target, tg = $('gaze-target'); tg.replaceChildren();
  if (target) { const sw = el('span', 'swatch'); sw.style.background = color(target); tg.append(sw, target); }
  else tg.textContent = gz.point ? (gz.point.on_video ? 'background' : 'off screen') : '—';
  const z = ee.series.at(-1)?.z; $('eeg-z').textContent = z != null ? `${z >= 0 ? '+' : ''}${z.toFixed(2)}σ` : '—'; drawEeg(ee.series);
  if (!s) { $('stage').textContent = 'Idle.'; return; }
  if (s.id !== sessionId) { sessionId = s.id; playingIndex = null; waitingFor = 0; lastChanges = ''; video.removeAttribute('src'); video.load(); }
  running = s.status === 'running'; $('stop').disabled = !running; $('setup').hidden = running;
  $('stage').textContent = `${s.demo ? 'DEMO · ' : ''}${s.status} · ${s.stage}${s.error ? ` · ${s.error}` : ''}`;
  $('profile-clips').textContent = `${s.profile.clips} scene${s.profile.clips === 1 ? '' : 's'} measured`;
  const lastChange = [...s.clips].reverse().find(c => c.changes?.length)?.changes || [];
  const changedKeys = new Set(lastChange.map(c => c.key));
  $('affinity').replaceChildren(el('h4', null, 'Character response'), ...Object.entries(s.profile.characters).map(([n, v]) => bar(n, v, { swatch: color(n), flash: changedKeys.has(`character:${n}`) })));
  $('prefs').replaceChildren(el('h4', null, 'Preferences'), bar('pacing', s.profile.pacing, { signed: true, flash: changedKeys.has('pacing') }),
    bar('dialogue', s.profile.dialogue, { signed: true, flash: changedKeys.has('dialogue') }),
    ...Object.entries(s.profile.genres).filter(([, v]) => v !== 0).map(([g, v]) => bar(g, v, { signed: true, flash: changedKeys.has(`genre:${g}`) })));
  const key = JSON.stringify(lastChange);
  if (key !== lastChanges) {
    lastChanges = key;
    $('changes').replaceChildren(...(lastChange.length ? lastChange.map(c => {
      const li = el('li', c.strong ? 'strong' : null);
      li.append(el('span', null, `${c.key.replace('character:', '')} `), el('span', 'delta', `${c.before.toFixed(2)} → ${c.after.toFixed(2)}`), el('div', 'hint', c.why + (c.strong ? ' · STRONG RESPONSE' : '')));
      return li;
    }) : [el('li', 'hint', 'Nothing yet: watch the first scene.')]));
  }
  const latest = s.clips.at(-1);
  if (latest) {
    const d = latest.decision;
    $('decision').replaceChildren(el('span', `chip${d.focus ? ' on' : ''}`, `focus: ${d.focus || 'balanced'}`), el('span', `chip${d.tension !== 'same' ? ' on' : ''}`, `tension: ${d.tension}`),
      el('span', `chip${d.dialogue !== 'same' ? ' on' : ''}`, `dialogue: ${d.dialogue}`), el('span', `chip${d.pacing !== 'same' ? ' on' : ''}`, `pacing: ${d.pacing}`),
      el('span', `chip${d.tone ? ' on' : ''}`, `tone: ${d.tone || 'same'}`), el('span', `chip${d.event ? ' on' : ''}`, `new event: ${d.event ? 'yes' : 'no'}`));
    $('reasons').replaceChildren(...d.reasons.map(r => el('li', null, r)));
    $('change-note').textContent = latest.plan.change_note; $('writer').textContent = latest.writer;
    $('next-title').textContent = `Scene ${latest.index + 1}: ${latest.plan.scene_title}`; $('next-prompt').textContent = latest.plan.video_prompt;
  }
  $('scenes').replaceChildren(...s.clips.map(c => {
    const d = el('div', `scene${c.index === playingIndex ? ' current' : ''}`);
    d.append(el('b', null, `${c.index + 1}. ${c.plan.scene_title}`), el('span', null, `${c.status}${c.decision.focus ? ` · focus ${c.decision.focus}` : ''}${c.generatedS ? ` · ${c.generatedS}s to make` : ''}`));
    return d;
  }));
  managePlayer(s);
}
async function poll() {
  try { render(await request('/api/adaptive/state')); $('backend-status').textContent = 'Python backend · connected'; }
  catch (e) { $('backend-status').textContent = `Backend: ${e.message}`; }
  setTimeout(poll, 200);
}
async function init() {
  addCharacter(); addCharacter();
  try {
    const config = await request('/api/config');
    $('duration').replaceChildren(...config.durations.map(n => new Option(String(n), String(n)))); $('duration').value = '10';
    syncSequenceSettings();
    $('cost-note').textContent = config.demo ? 'DEMO mode: synthetic clips, no Fal calls.' : config.configured
      ? 'Each scene is a paid Fal generation (and an OpenAI call if OPENAI_API_KEY is set).' : 'Fal key missing: set FAL_KEY in .env and restart.';
  } catch (e) { error(e.message); }
  void poll();
}
void init();
