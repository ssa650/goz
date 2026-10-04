import test from 'node:test';
import assert from 'node:assert/strict';
import { videoContentRect, visibleContentRect, ScreenPointMapping, boxesAt } from '../frontend/adaptive-geometry.js';

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
  const mapping=new ScreenPointMapping();
  mapping.observe({clientX:0,clientY:0,screenX:90,screenY:130},'window');
  mapping.observe({clientX:200,clientY:100,screenX:290,screenY:230},'window');
  assert.deepEqual(mapping.rect({left:10,top:20,w:800,h:450},'window'),{x:100,y:150,w:800,h:450});
  assert.equal(mapping.rect({left:10,top:20,w:800,h:450},'fullscreen'),null);
  assert.deepEqual(videoContentRect({left:0,top:0,width:1440,height:900},1920,1080),{left:0,top:45,w:1440,h:810});
});

test('object fit/position retain full content and exclude cropped or scrolled pixels',()=>{
  const element={left:-60,top:-40,width:300,height:300};
  const content=videoContentRect(element,400,200,'cover','100% 0%');
  assert.deepEqual(content,{left:-360,top:-40,w:600,h:300});
  assert.deepEqual(visibleContentRect(content,[{left:-60,top:-40,w:300,h:300},{left:0,top:0,w:800,h:600},
    {left:25,top:10,w:120,h:80}]),{left:25,top:10,w:120,h:80});
  assert.deepEqual(videoContentRect({left:0,top:0,width:100,height:100},200,50,'scale-down'),{left:0,top:37.5,w:100,h:25});
  assert.deepEqual(videoContentRect({left:0,top:0,width:100,height:100},40,20,'none','10px 15px'),{left:10,top:15,w:40,h:20});
  assert.equal(videoContentRect(element,400,200,'contain','calc(50% + 1px) center'),null);
  assert.equal(videoContentRect({...element,left:NaN},400,200),null);
});

test('fully hidden video yields zero visible area, never a clamped gaze rectangle',()=>{
  const full={left:10,top:650,w:800,h:450};
  const visible=visibleContentRect(full,[{left:0,top:0,w:900,h:600}]);
  assert.equal(visible.h,0);
  const mapping=new ScreenPointMapping();mapping.observe({clientX:0,clientY:0,screenX:0,screenY:0},'s');
  mapping.observe({clientX:100,clientY:100,screenX:100,screenY:100},'s');
  assert.equal(mapping.rect(visible,'s'),null);
  assert.deepEqual(mapping.rect(full,'s'),{x:10,y:650,w:800,h:450});
});

test('expired detection and scene cut never reuse old boxes or choose future boxes',()=>{
  const track=[{t:0,valid_until:.3,boxes:{Patrick:[0,0,1,1]}},{t:.4,valid_until:1,boxes:{}}];
  assert.deepEqual(boxesAt(track,-.1),{});assert.deepEqual(boxesAt(track,.31),{});assert.deepEqual(boxesAt(track,.4),{});
});
