import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Execute the current request helper and submit handler, including real FormData.
const source = readFileSync(new URL('../frontend/adaptive.js', import.meta.url), 'utf8');
const requestSource = source.slice(source.indexOf('async function request('), source.indexOf('const post ='));
const submitSource = source.slice(source.indexOf("$('setup').addEventListener('submit'"), source.indexOf('for (const [id,recenter]'));

async function submit(response = new Response('{}', { status: 202 }), opening = 'start') {
  const nodes = {
    setup: { addEventListener: (_, fn) => { handler = fn; } },
    characters: { children: ['Ana', 'Bea'].map(value => ({ children: [{ value }, { value: `${value} look` }] })) },
    opening: { files: [new Blob([opening === 'start' ? Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64') : 'mock-video'], { type: opening === 'start' ? 'image/png' : 'video/mp4' })] },
    'use-sequence': { checked: false }, premise: { value: 'Two explorers find a box.' },
    timeline: { value: '0-5 Ana: Look at the box.' }, tracker: { value: 'color' },
    'stream-playback': { checked: true }, 'eeg-run-mode': { value: 'cumulative_prior_clips' },
    duration: { value: '15' }, resolution: { value: '480P' }, objects: { value: 'box' }, start: {},
  };
  let handler, sent, message;
  vm.runInNewContext(`${requestSource}\n${submitSource}`, {
    $: id => nodes[id], FormData, AbortController, setTimeout, clearTimeout,
    state: { setup: { generationReady: true } }, providerConfigured: true, submitting: false,
    trackerValue: value => value, error: value => { message = value; },
    fetch: async (path, options) => { sent = { path, ...options }; return response; },
  });
  await handler({ preventDefault() {} });
  return { sent, message, nodes };
}

if (process.argv.includes('--multipart-fixture')) {
  const { sent } = await submit(undefined, process.argv.includes('--opening-video') ? 'opening' : 'start');
  const wire = new Request('http://testserver' + sent.path, { method: sent.method, body: sent.body });
  process.stdout.write(JSON.stringify({ contentType: wire.headers.get('content-type'), body: Buffer.from(await wire.arrayBuffer()).toString('base64') }));
} else {
  test('current setup sends every field with streaming and sustained EEG together', async () => {
    const { sent, message, nodes } = await submit();
    assert.equal(sent.path, '/api/adaptive/sessions');
    assert.deepEqual([...sent.body.keys()], ['start', 'use_saved_sequence', 'premise', 'timeline', 'tracker', 'playback_mode', 'eegRunMode', 'characters', 'duration', 'resolution', 'objects']);
    assert.equal(sent.body.get('playback_mode'), 'stream');
    assert.equal(sent.body.get('eegRunMode'), 'cumulative_prior_clips');
    assert.equal(message, '');
    assert.equal(nodes.start.disabled, false);
  });
  for (const [body, expected] of [
    [JSON.stringify({ detail: 'Too many fields. Maximum number of fields is 16.' }), 'Too many fields. Maximum number of fields is 16.'],
    [JSON.stringify({ error: 'Choose download or stream playback.' }), 'Choose download or stream playback.'],
    ['Malformed multipart request.', 'Malformed multipart request.'],
    ['', 'Request failed (400).'],
    ['{}', 'Request failed (400).'],
  ]) {
    test(`setup displays server failure: ${expected}`, async () => {
      const { message, nodes } = await submit(new Response(body, { status: 400 }));
      assert.equal(message, expected);
      assert.equal(nodes.start.disabled, false);
    });
  }
  test('successful JSON response remains available to callers', async () => {
    const context = { fetch: async () => new Response('{"id":"session"}'), AbortController, setTimeout, clearTimeout };
    vm.runInNewContext(requestSource, context);
    assert.equal((await context.request('/test')).id, 'session');
  });
}
