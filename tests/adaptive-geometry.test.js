import test from 'node:test';
import assert from 'node:assert/strict';
import { videoContentRect, screenContentRect, boxesAt } from '../frontend/adaptive-geometry.js';

test('video mapping excludes letterbox/pillarbox and includes layout offsets', () => {
  assert.deepEqual(videoContentRect({left:100,top:200,width:800,height:600}, 1920, 1080), {left:100,top:275,w:800,h:450});
  assert.deepEqual(videoContentRect({left:100,top:200,width:800,height:450}, 600, 600), {left:275,top:200,w:450,h:450});
  assert.equal(videoContentRect({left:0,top:0,width:800,height:450}, 0, 0), null);
});

test('sparse Florence overlay uses the same bounded past-only tolerance as fusion', () => {
  const boxes = {Patrick:[.1,.2,.3,.4]};
  const track = [{t:0,boxes}, {t:2,boxes:{}}];
  assert.deepEqual(boxesAt(track,.8), boxes);
  assert.deepEqual(boxesAt(track,1.2), {});
  assert.deepEqual(boxesAt(track,8), {});
});


test('cover cropping, resized fullscreen, screen origin and retina use logical pixels', () => {
  const c=videoContentRect({left:0,top:0,width:800,height:600},1920,1080,'cover');
  assert.ok(c.left<0); assert.equal(c.h,600);
  const w={screenX:90,screenY:30,innerWidth:800,outerWidth:800,innerHeight:600,outerHeight:700,fullscreen:false};
  assert.deepEqual(screenContentRect({left:10,top:20,w:800,h:450},w),{x:100,y:150,w:800,h:450});
  assert.deepEqual(screenContentRect({left:10,top:20,w:800,h:450},{...w,fullscreen:true}),{x:100,y:50,w:800,h:450});
  assert.deepEqual(videoContentRect({left:0,top:0,width:1440,height:900},1920,1080),{left:0,top:45,w:1440,h:810});
});

test('expired detection and scene cut never reuse old boxes or choose future boxes',()=>{
  const track=[{t:0,valid_until:.3,boxes:{Patrick:[0,0,1,1]}},{t:.4,valid_until:1,boxes:{}}];
  assert.deepEqual(boxesAt(track,-.1),{});assert.deepEqual(boxesAt(track,.31),{});assert.deepEqual(boxesAt(track,.4),{});
});
