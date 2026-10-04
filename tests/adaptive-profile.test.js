import test from 'node:test';
import assert from 'node:assert/strict';
import {profilePresentation} from '../frontend/adaptive-profile.js';

const session=()=>({profile:{clips:0,characters:{A:.5,B:.5}},clips:[]});
test('equal seeded weights are never labeled as measured attention',()=>{
  const s=session(), view=profilePresentation(s);
  assert.equal(view.scenes,'0 scenes evaluated');
  assert.equal(view.characterHeading,'Starting profile weights');
  assert.match(view.characterNote,/do not mean equal attention/);
  assert.equal(view.deliveryHeading,'Default delivery settings');
  assert.match(view.noChanges,/No viewing evidence/);
  assert.deepEqual(s.profile.characters,{A:.5,B:.5});
});
test('evaluated scenes with no usable evidence remain starting weights',()=>{
  const s=session();s.profile.clips=4;s.clips=Array.from({length:4},()=>({profileChanges:[]}));
  const view=profilePresentation(s);
  assert.equal(view.scenes,'4 scenes evaluated');assert.equal(view.characterUpdated,false);
  assert.match(view.characterNote,/No supported comparative attention update/);
  assert.match(view.noChanges,/No evidence-supported preference change/);
});
test('recorded updates change only the applicable presentation and stay policy weights',()=>{
  const s=session();s.profile.clips=2;s.profile.characters={A:.75,B:.25};
  s.clips=[{profileChanges:[{key:'character:A',before:.5,after:.75},{key:'character:B',before:.5,after:.25}]},{profileChanges:[]}];
  const view=profilePresentation(s);
  assert.equal(view.characterHeading,'Tentative profile weights');
  assert.match(view.characterNote,/not measured attention percentages/);
  assert.equal(view.deliveryHeading,'Default delivery settings');
  s.clips.push({profileChanges:[{key:'pacing',before:0,after:-.25}]});
  assert.equal(profilePresentation(s).deliveryHeading,'Tentative delivery adjustments');
});
test('missing changes and nonfinite changes do not establish measured preferences',()=>{
  const s=session();s.profile.characters={A:.8,B:.2};s.clips=[{changes:[{key:'character:A',before:NaN,after:.8}]}];
  assert.equal(profilePresentation(s).characterUpdated,false);
});
