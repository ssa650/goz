import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseSeed, randomSeed, settingsFor, sequenceFor, missingFrameMessage} from '../frontend/clip-state.js';

test('manual seeds reject fractions, empty values, and unsafe JSON integers',()=>{
  assert.equal(parseSeed('483729'),483729);
  assert.equal(parseSeed('0'),0);
  assert.equal(parseSeed('-42'),-42);
  for(const invalid of ['', ' ', '1.5', 'Infinity', '1e6', '9007199254740992']) assert.throws(()=>parseSeed(invalid));
  for(let i=0;i<20;i++) {const seed=randomSeed();assert.ok(Number.isInteger(seed)&&seed>=0&&seed<=0x7fffffff);}
});
test('imported filename requirements block missing images without changing prompt or seed',()=>{
  const clip={id:'clip-1',order:0,prompt:'Scene one',seed:741203,firstFrame:null,endFrame:null,initialFrameFile:'00-00.jpg',endFrameFile:'00-15.jpg',promptExpansionMode:'disabled',duration:15,resolution:'480P'};
  assert.match(missingFrameMessage(clip),/00-00.jpg/);
  assert.match(missingFrameMessage(clip),/00-15.jpg/);
  clip.firstFrame={id:'asset-1',name:'00-00.jpg'};
  assert.doesNotMatch(missingFrameMessage(clip),/00-00.jpg/);
  clip.endFrame={id:'asset-2',name:'00-15.jpg'};
  assert.equal(missingFrameMessage(clip),null);
  const body=sequenceFor([clip],'batch');
  assert.equal(body.clips[0].firstFrame,'asset-1');assert.equal(body.clips[0].endFrame,'asset-2');
  assert.equal(body.clips[0].seed,741203);assert.equal(body.clips[0].prompt,'Scene one');
});
test('sequence serialization follows UI order and keeps settings and both frames in each bundle',()=>{
  const clips=Array.from({length:4},(_,index)=>({id:`clip-${index+1}`,order:index,prompt:`prompt-${index+1}`,seed:100+index,
    firstFrame:{id:`initial-${index+1}`},endFrame:{id:`end-${index+1}`},promptExpansionMode:'disabled',duration:15,resolution:'480P'}));
  const reordered=[clips[0],clips[2],clips[1],clips[3]];
  const body=sequenceFor(reordered,'run-id');
  assert.deepEqual(body.clips.map(c=>c.id),['clip-1','clip-3','clip-2','clip-4']);
  assert.deepEqual(body.clips.map(c=>c.order),[0,1,2,3]);
  for(const clip of body.clips) {
    const n=Number(clip.id.split('-')[1]);
    assert.equal(clip.prompt,`prompt-${n}`);assert.equal(clip.seed,99+n);
    assert.equal(clip.firstFrame,`initial-${n}`);assert.equal(clip.endFrame,`end-${n}`);
  }
  const shortened=sequenceFor(reordered.filter(c=>c.id!=='clip-2'),'run-2');
  assert.deepEqual(shortened.clips.map(c=>c.id),['clip-1','clip-3','clip-4']);
  assert.deepEqual(shortened.clips.map(c=>c.order),[0,1,2]);
  assert.equal(clips[1].order,1,'Serialization never mutates original bundles');
  assert.equal('prompts' in body,false);assert.equal('initialFrames' in body,false);
});
test('serialization preserves each clip seed and immutable frame IDs across renders',()=>{
  const frame={id:'first',name:'frame.png',previewUrl:'/api/images/first',providerUrl:null,contentType:'image/png',size:123};
  const clip1={prompt:'one',seed:483729,firstFrame:frame,endFrame:null,promptExpansionMode:'disabled',duration:5,resolution:'480P'};
  const clip2={prompt:'two',seed:99,firstFrame:null,endFrame:null,promptExpansionMode:'quality',duration:10,resolution:'1080P'};
  assert.deepEqual(settingsFor(clip1),settingsFor(clip1));
  const one=settingsFor(clip1);const two=settingsFor(clip2);
  assert.equal(one.seed,483729);assert.equal(one.firstFrame,'first');assert.equal(two.seed,99);assert.equal(two.firstFrame,null);
  two.seed=3;assert.equal(clip1.seed,483729);assert.equal(clip2.seed,99);
});
