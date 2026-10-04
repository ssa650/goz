import assert from 'node:assert/strict';
import test from 'node:test';
import {trackerValue,trackingSummary} from '../frontend/adaptive-tracker.js';

test('provider choice defaults safely and retains local rollback option',()=>{
  assert.equal(trackerValue('opencv'),'opencv');
  assert.equal(trackerValue('fal'),'fal');
  assert.equal(trackerValue('made-up'),'fal');
});
test('tracking status uses past fresh evidence and explicitly reports unknown',()=>{
  const clip={detectionStatus:'processing',track:[
    {t:0,boxes:{Patrick:[.1,.1,.4,.9]},valid_until:.125,source:'opencv_reference_flow'},
    {t:1,boxes:{SpongeBob:[.6,.1,.9,.9]},valid_until:1.125,source:'opencv_reference_flow'}]};
  assert.match(trackingSummary({tracker:'opencv'},clip,.1),/Local OpenCV.*Patrick.*opencv_reference_flow/);
  assert.match(trackingSummary({tracker:'opencv'},clip,.2),/identity unknown.*no fresh frame/);
  assert.doesNotMatch(trackingSummary({tracker:'opencv'},clip,.5),/SpongeBob/);
  assert.match(trackingSummary({tracker:'fal'},{...clip,detectionLifecycle:'cancelled'},.5),/Florence.*cancelled.*identity unknown/);
  assert.match(trackingSummary({tracker:'opencv',status:'stopped'},clip,.1),/stopped.*identity unknown.*no fresh frame/);
});
