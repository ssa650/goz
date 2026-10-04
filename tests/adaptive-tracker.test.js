import assert from 'node:assert/strict';
import test from 'node:test';
import {trackerValue,trackingSummary,colorMatchSummary} from '../frontend/adaptive-tracker.js';

test('provider choice defaults safely and retains local rollback option',()=>{
  assert.equal(trackerValue('opencv'),'opencv');
  assert.equal(trackerValue('fal'),'fal');
  assert.equal(trackerValue('made-up'),'color');
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

test('color scores describe accepted and rejected hypotheses without a probability claim',()=>{
  const frame={t:0,valid_until:.25,source:'opencv_color_shape_demo',boxes:{SpongeBob:[.1,.1,.4,.8]},unknown:['Patrick'],regions:[
    {identity:'SpongeBob',candidate_identity:'SpongeBob',identity_status:'demo_color_shape',verification:{score_kind:'uncalibrated_heuristic_score',score:.853}},
    {identity:null,candidate_identity:'Patrick',identity_status:'unknown_insufficient_support',verification:{score_kind:'uncalibrated_heuristic_score',score:.72}}]};
  const text=colorMatchSummary(frame);
  assert.match(text,/SpongeBob: color match score \(uncalibrated\) 85%/);
  assert.match(text,/Patrick: color match score \(uncalibrated\) 72% · not confidently detected \(insufficient visual support\)/);
  assert.doesNotMatch(text,/probability|absent|confidence 72/);
  assert.doesNotMatch(trackingSummary({tracker:'color'},{track:[frame]},.5),/match score/);
});
test('no color candidate has unavailable score and a rejection reason, never fabricated zero',()=>{
  const text=colorMatchSummary({source:'opencv_color_shape_demo',boxes:{},unknown:['Patrick'],regions:[],abstention_reason:'blank_frame'});
  assert.match(text,/score \(uncalibrated\) unavailable · not confidently detected \(blank frame\)/);
  assert.equal(colorMatchSummary({source:'opencv_reference_flow'}),'');
});
