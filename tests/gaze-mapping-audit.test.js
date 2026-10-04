import test from 'node:test';
import assert from 'node:assert/strict';
import {ScreenPointMapping, videoContentRect} from '../frontend/adaptive-geometry.js';

// Mock MouseEvent observations follow browser CSS units and OS screen points.
const pointer=(x,y,scale=1,ox=90,oy=130)=>({clientX:x,clientY:y,screenX:ox+x*scale,screenY:oy+y*scale});
for(const scale of [1,1.25,.8,2]) test(`scripted gaze hits ground-truth video boxes at scale ${scale}`,()=>{
  const mapping=new ScreenPointMapping();
  mapping.observe(pointer(100,80,scale),'display');
  assert.equal(mapping.valid('display'),false); // One pointer does not validate scale.
  mapping.observe(pointer(350,240,scale),'display');
  const content=videoContentRect({left:100,top:200,width:800,height:600},1920,1080);
  const screen=mapping.rect(content,'display');
  assert.ok(mapping.valid('display'));
  assert.ok(Math.abs(screen.w-800*scale)<1e-7);
  const boxes={left:[.1,.1,.4,.9],right:[.6,.1,.9,.9]};
  for(const [name,box] of Object.entries(boxes)) {
    const point={x:screen.x+(box[0]+box[2])/2*screen.w,y:screen.y+(box[1]+box[3])/2*screen.h};
    const nx=(point.x-screen.x)/screen.w,ny=(point.y-screen.y)/screen.h;
    assert.ok(nx>=box[0]&&nx<=box[2]&&ny>=box[1]&&ny<=box[3],name);
  }
  const letterbox=pointer(500,220,scale);
  assert.ok((letterbox.screenY-screen.y)/screen.h<0);
  assert.equal(mapping.audit('display').eyeCalibrationVerified,false);
});
test('retina/zoom/window/fullscreen changes require new fit; mirror/scale mismatch rejects',()=>{
 const m=new ScreenPointMapping();m.observe(pointer(0,0),'dpr2');m.observe(pointer(200,100),'dpr2');
 assert.ok(m.valid('dpr2'));assert.equal(m.rect({left:0,top:0,w:1,h:1},'dpr1'),null);
 m.observe(pointer(0,0),'fullscreen');assert.equal(m.valid('fullscreen'),false);
 m.observe({...pointer(200,100),screenX:-110},'fullscreen');assert.equal(m.valid('fullscreen'),false);
 const a=new ScreenPointMapping();a.observe(pointer(0,0),'s');a.observe({...pointer(200,100),screenY:330},'s');
 assert.equal(a.valid('s'),false); // anisotropic or wrong screen transform
});
test('unit-scale origin-only implementation demonstrably misses at non-unit scale',()=>{
 const p=pointer(100,80,1.25),rect={left:100,top:275,w:800,h:450};
 const old={x:p.screenX-p.clientX+rect.left,y:p.screenY-p.clientY+rect.top,w:rect.w,h:rect.h};
 const m=new ScreenPointMapping();m.observe(p,'s');m.observe(pointer(350,240,1.25),'s');
 const actual=m.rect(rect,'s');
 const x=actual.x+actual.w*.85;
 assert.ok((x-old.x)/old.w>1); // pointer appearance alone cannot establish valid mapping
 assert.ok(Math.abs((x-actual.x)/actual.w-.85)<1e-7);
});
