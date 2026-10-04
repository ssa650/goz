import test from 'node:test';
import assert from 'node:assert/strict';
import {overlayDiagnostic} from '../frontend/tracking-diagnostics.js';

const params={clip:{track:[{t:.75,valid_until:1,boxes:{SpongeBob:[.2,.3,.4,.7]}}]},
  enabled:true,mediaTime:.8,content:{left:100,top:150,w:800,h:450},
  canvas:{left:80,top:100,width:840,height:550},video:{left:100,top:100,width:800,height:550},dpr:2};

test('overlay diagnostics use actual canvas CSS coordinates without retina scaling',()=>{
  const d=overlayDiagnostic(params);
  assert.equal(d.status,'rendered');
  assert.deepEqual(d.boxes[0].normalized,[.2,.3,.4,.7]);
  assert.equal(d.boxes[0].canvas[0],180);
  assert.equal(d.boxes[0].canvas[1],185);
  assert.equal(d.boxes[0].canvas[2],160);
  assert.ok(Math.abs(d.boxes[0].canvas[3]-180)<1e-6);
  assert.equal(d.devicePixelRatio,2);
});

test('late received boxes report expiry, while disabled and missing bounds stay distinct',()=>{
  assert.equal(overlayDiagnostic({...params,mediaTime:12}).status,'expired_frame');
  assert.deepEqual(overlayDiagnostic({...params,mediaTime:12}).boxes,[]);
  assert.equal(overlayDiagnostic({...params,enabled:false}).status,'disabled');
  assert.equal(overlayDiagnostic({...params,content:null}).status,'content_bounds_unavailable');
  assert.equal(overlayDiagnostic({...params,clip:{track:[]}}).status,'no_frame');
});

test('empty current frame and a future frame do not resurrect an old named box',()=>{
  const clip={track:[...params.clip.track,{t:1,valid_until:1.25,boxes:{}},{t:3,boxes:{Patrick:[0,0,1,1]}}]};
  assert.equal(overlayDiagnostic({...params,clip,mediaTime:1.1}).status,'no_named_boxes');
  assert.equal(overlayDiagnostic({...params,clip,mediaTime:.5}).status,'no_frame');
});
