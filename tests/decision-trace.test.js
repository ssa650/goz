import test from 'node:test';
import assert from 'node:assert/strict';
import { filteredTraces, renderTraces, withTrackingDiagnostics } from '../frontend/decision-trace.js';

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.textContent = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  set innerHTML(value) { throw new Error('HTML injection sink used'); }
}

test('debug records and diff render as text, preserving XSS-looking strings', () => {
  const payload = '<img src=x onerror=alert(1)><script>alert(1)</script>';
  const rows = [{sceneIndex:0, sessionId:payload, action:'keep', submission:'confirmed', diff:payload}];
  const container = new Element('div');
  renderTraces(container, rows, {createElement: tag => new Element(tag)});
  assert.equal(container.children[0].tag, 'details');
  assert.equal(container.children[0].children[0].tag, 'summary');
  assert.ok(container.children[0].children[0].textContent.includes(payload));
  assert.ok(container.children[0].children[1].textContent.includes(payload));
  assert.equal(container.children[0].children[1].children.length, 0);
});

test('session filter separates reloaded runs and allows all-runs export', () => {
  const records = [{sessionId:'old'}, {sessionId:'new'}];
  assert.deepEqual(filteredTraces(records, 'old'), [records[0]]);
  assert.deepEqual(filteredTraces(records, ''), records);
  assert.deepEqual(filteredTraces(records, 'missing'), []);
});


test('saved detector history attaches only to the decision clip and its source in the same session', () => {
  const decisions=[{sessionId:'new',clipId:'next',references:{sourceClipId:'source'}}];
  const events=[{sessionId:'new',clipId:'source'}, {sessionId:'new',clipId:'next'},
    {sessionId:'old',clipId:'source'}, {sessionId:'new',clipId:'unrelated'}];
  const result=withTrackingDiagnostics(decisions,events);
  assert.deepEqual(result[0].trackingDiagnostics.records,events.slice(0,2));
  assert.match(result[0].trackingDiagnostics.note,/after the decision freeze/);
  assert.equal(decisions[0].trackingDiagnostics,undefined);
});
