import { PlaybackQueue } from './queue.js';
const $ = id => document.getElementById(id);
let configured = false, busy = false, posting = false, sequence = null, pendingBody = null, pollTimer, pollEpoch = 0;
let sound = false, viewing = false, demo = false, countEpoch = 0, countTimer, previewURLs = [];
const active = s => s && !['completed','cancelled','failed','interrupted'].includes(s.status);
const error = message => { $('error').textContent = message || ''; $('error').hidden = !message; };
const player = new PlaybackQueue([$('video-a'), $('video-b')], ({ state, index }) => {
  viewing = ['playing','loading','buffering','gesture'].includes(state);
  const n = index + 1;
  const messages = { playing: `Playing clip ${n}.`, loading: `Loading clip ${n}…`, buffering: `Buffering — waiting for clip ${n}.`,
    finished: 'Sequence finished.', cancelled: 'Viewing cancelled.', stopped: 'No more clips. Generation stopped.',
    gesture: 'Your browser requires a click to start viewing.', error: `Clip ${n} could not play. The sequence will not skip it.` };
  $('playback-status').textContent = messages[state];
  $('overlay').hidden = state === 'playing'; $('overlay').textContent = messages[state];
  $('view').hidden = state !== 'gesture';
  if (state === 'error') error(messages[state]);
  controls();
});
function controls() {
  $('inputs').disabled = posting || Boolean(active(sequence)) || Boolean(pendingBody);
  $('run').disabled = !configured || busy || posting || Boolean(active(sequence));
  $('run').textContent = pendingBody ? 'Reconnect to run' : 'Run sequence';
  $('cancel').disabled = posting || !(active(sequence) || viewing);
}
async function request(path, options) {
  const r = await fetch(path, options); const value = await r.json();
  if (!r.ok) throw Object.assign(new Error(value.error || 'Request failed.'), { status: r.status });
  return value;
}
function render(s) {
  sequence = s; busy = s.generationBusy;
  $('debug-state').textContent = JSON.stringify(s, null, 2);
  const count = s.prompts.length;
  const labels = { preparing: 'Preparing', generating: s.generationStatus || 'Generating', extracting: 'Extracting actual last frame',
    completed: 'All clips generated', failed: 'Generation stopped', cancelled: 'Cancelled', interrupted: 'Interrupted' };
  const runTime = s.finishedAt && s.startedAt ? ` Total run time ${seconds(s.finishedAt - s.startedAt)}.` : '';
  $('status').textContent = `${labels[s.status]} · clip ${s.index + 1}/${count} · ${s.clips.length} ready.${runTime}${s.warning ? ` ${s.warning}` : ''}`;
  if (s.error) error(s.error);
  if (s.status === 'cancelled') player.stop(); else player.update(s.clips, s.status);
  $('downloads').replaceChildren(...s.clips.map(c => {
    const row = document.createElement('div'); row.className = 'clip';
    const a = document.createElement('a'); a.textContent = `Save clip ${c.index + 1}`; a.href = `/api/jobs/${c.jobId}/download`;
    const info = document.createElement('span'); info.textContent = clipTiming(c);
    row.append(a, info); return row;
  })); controls();
}
function seconds(ms) { return `${(ms / 1000).toFixed(1)}s`; }
function clipTiming(c) {
  const parts = [];
  if (c.totalElapsedMs != null) parts.push(`generated in ${seconds(c.totalElapsedMs)}`);
  if (c.uploadElapsedMs != null) parts.push(`upload ${seconds(c.uploadElapsedMs)}`);
  if (c.apiReadyMs != null) parts.push(`${demo ? 'demo ready' : 'Fal ready'} ${seconds(c.apiReadyMs)}`);
  if (c.inferenceMs != null) parts.push(`inference ${seconds(c.inferenceMs)}`);
  if (c.extractMs != null) parts.push(`last frame ${seconds(c.extractMs)}`);
  return parts.length ? `${c.duration}s clip · ${parts.join(' · ')}` : '';
}
function polling(id) {
  clearTimeout(pollTimer); const epoch = ++pollEpoch;
  async function tick() {
    try {
      const s = await request(`/api/sequences/${id}`);
      if (epoch !== pollEpoch) return;
      render(s);
      if (active(s) || s.generationBusy) pollTimer = setTimeout(tick, 500);
    } catch (e) {
      if (epoch !== pollEpoch) return;
      error(`Status connection interrupted: ${e.message}. Reconnecting; no generation will be resubmitted.`);
      pollTimer = setTimeout(tick, 1500);
    }
  }
  void tick();
}
const keyframeMode = () => $('frame-mode').value === 'keyframes';
const keyframeFiles = () => [...$('keyframes').files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
async function clipPrompts() {
  return request('/api/plan', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: $('prompts').value, mode: $('frame-mode').value, duration: Number($('duration').value) }) });
}
function frameCount(plan) {
  const files = keyframeFiles();
  $('keyframe-count').textContent = `${plan ? `Needs ${plan.frameCount} keyframes` : 'Add prompts to see how many keyframes are needed'} · ${files.length} selected${files.length ? `: ${files.map(f => f.name).join(', ')}` : ''}`;
}
function previewFrames() {
  previewURLs.forEach(URL.revokeObjectURL); previewURLs = [];
  const files = keyframeMode() ? keyframeFiles() : [...$('image').files];
  $('frame-previews').replaceChildren(...files.map((file, i) => {
    const figure = document.createElement('figure'), image = document.createElement('img'), caption = document.createElement('figcaption');
    const url = URL.createObjectURL(file); previewURLs.push(url); image.src = url; image.alt = `Frame ${i + 1}: ${file.name}`;
    caption.textContent = `${i + 1}. ${file.name}`; figure.append(image, caption); return figure;
  }));
}
function updateCounts() {
  $('keyframes-field').hidden = !keyframeMode(); $('keyframe-count').hidden = !keyframeMode(); $('image-field').hidden = keyframeMode();
  const runNote = demo ? 'Demo uses synthetic local clips. ' : 'Run submits paid generations to your Fal account. ';
  $('mode-note').textContent = runNote + (keyframeMode()
    ? 'Timed scenes are split into clips of the chosen length. Clip N runs from keyframe N to keyframe N+1. Frames are ordered by file name.'
    : "Each next clip uses the previous video's actual last frame. Generation may be slower than playback.");
  previewFrames(); frameCount(null);
  clearTimeout(countTimer); const epoch = ++countEpoch;
  if (!$('prompts').value.trim()) { $('batch-count').textContent = '0 clips · maximum 12'; return; }
  $('batch-count').textContent = 'Planning clips…';
  countTimer = setTimeout(async () => {
    try {
      const plan = await clipPrompts(); if (epoch !== countEpoch) return;
      $('batch-count').textContent = `${plan.scenes.length} scenes → ${plan.clips.length} clips of ${$('duration').value}s · ${plan.totalDuration}s total · maximum 12`;
      frameCount(plan);
    } catch (e) { if (epoch === countEpoch) $('batch-count').textContent = e.message; }
  }, 200);
}
$('prompts').addEventListener('input', updateCounts);
$('keyframes').addEventListener('change', updateCounts);
$('image').addEventListener('change', previewFrames);
$('duration').addEventListener('change', updateCounts);
$('frame-mode').addEventListener('change', () => { $('duration').value = keyframeMode() ? '5' : '15'; updateCounts(); });
$('prompt-file').addEventListener('change', async () => {
  const file = $('prompt-file').files[0]; if (!file) return;
  try {
    if (file.size > 100000) throw new Error('Prompt file must be at most 100 KB.');
    const text = await file.text(); $('prompts').value = text.replace(/^\uFEFF/, '');
    $('prompts').dispatchEvent(new Event('input')); error('');
  } catch (e) { error(e.message); }
});
$('run-form').addEventListener('submit', async event => {
  event.preventDefault(); if (posting || busy || active(sequence)) return;
  posting = true; controls(); error('');
  try {
    if (!pendingBody) {
      const { clips } = await clipPrompts(), id = crypto.randomUUID();
      const body = new FormData(); body.append('id', id);
      if (keyframeMode()) {
        const frames = keyframeFiles();
        if (frames.length !== clips.length + 1) throw new Error(`Select exactly ${clips.length + 1} keyframes for ${clips.length} clips (${frames.length} selected).`);
        if (frames.some(f => f.size > 10 * 1024 * 1024)) throw new Error('Each keyframe must be at most 10 MB.');
        body.append('mode', 'keyframes'); frames.forEach(f => body.append('frames', f));
      } else {
        const image = $('image').files[0];
        if (!image || image.size > 10 * 1024 * 1024) throw new Error('Supply an initial PNG, JPEG or WebP, up to 10 MB.');
        body.append('start', image);
      }
      body.append('prompts', $('prompts').value); body.append('duration', $('duration').value); body.append('resolution', $('resolution').value);
      pendingBody = body;
      sessionStorage.setItem('sequence-id', id);
      pollEpoch++; clearTimeout(pollTimer); sequence = null; player.reset();
    }
    posting = true; controls(); $('status').textContent = 'Starting sequence…';
    const s = await request('/api/sequences', { method: 'POST', body: pendingBody });
    pendingBody = null; render(s); polling(s.id);
  } catch (e) {
    if (e.status) pendingBody = null;
    error(e.message + (!e.status && pendingBody ? ' Use Reconnect to run to recover the same run without duplicating it.' : ''));
  } finally { posting = false; controls(); }
});
$('cancel').addEventListener('click', async () => {
  if (!sequence || (!active(sequence) && !viewing)) return;
  $('cancel').disabled = true;
  try { render(await request(`/api/sequences/${sequence.id}/cancel`, { method: 'POST' })); player.stop(); polling(sequence.id); }
  catch (e) { error(e.message); controls(); }
});
$('key-form').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await request('/api/key', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: $('key').value }) });
    $('key').value = ''; configured = true; $('key-status').textContent = 'Fal key configured'; $('key-panel').open = false; controls(); error('');
  } catch (e) { $('key').value = ''; error(e.message); }
});
$('view').addEventListener('click', () => player.tryPlay());
$('sound').addEventListener('click', () => { sound = !sound; player.setSound(sound); $('sound').textContent = sound ? 'Mute sound' : 'Enable sound'; });
async function init() {
  try {
    const config = await request('/api/config'); configured = config.configured;
    demo = Boolean(config.demo);
    $('backend-status').textContent = demo ? 'Python backend · DEMO' : 'Python backend · connected';
    $('key-panel').hidden = demo;
    $('key-status').textContent = configured ? 'Fal key configured' : 'Add Fal key'; $('key-panel').open = !configured;
    $('duration').replaceChildren(...config.durations.map(n => new Option(String(n), String(n))));
    $('duration').value = keyframeMode() ? '5' : String(config.duration); updateCounts();
    const runs = await request('/api/sequences');
    const savedId = sessionStorage.getItem('sequence-id');
    const saved = runs.find(s => active(s)) || runs.find(s => s.id === savedId);
    if (saved) {
      sessionStorage.setItem('sequence-id', saved.id);
      $('frame-mode').value = saved.mode; $('duration').value = String(saved.duration); $('resolution').value = saved.resolution;
      $('prompts').value = JSON.stringify(saved.scenes || saved.prompts, null, 2); updateCounts();
      render(saved); polling(saved.id);
    }
    else {
      const jobs = await request('/api/jobs'); busy = jobs.some(j => !['completed','failed','cancelled'].includes(j.status) || j.requestUncertain);
      if (busy) $('status').textContent = 'An existing Fal request is still active or uncertain. Check it before running again.';
    }
    controls();
  } catch (e) { error(`Cannot connect to the local server: ${e.message}`); }
}
void init();
