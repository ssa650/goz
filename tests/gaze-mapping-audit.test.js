import test from 'node:test';
import assert from 'node:assert/strict';
import {ScreenPointMapping, mappingIdentity, videoContentRect, visibleContentRect} from '../frontend/adaptive-geometry.js';

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

const windowFixture=()=>({screenX:139,screenY:0,innerWidth:1420,innerHeight:1021,outerWidth:1420,outerHeight:1107,
  devicePixelRatio:2,screen:{width:1710,height:1107,availLeft:0,availTop:25,availWidth:1710,availHeight:1082},fullscreen:false,
  visualViewport:{width:1420,height:1021,offsetLeft:0,offsetTop:0,scale:1}});
const recover=(m,key,scale=1,ox=139,oy=86)=>{
  m.beginRecovery(key);
  assert.equal(m.acceptAnchor(pointer(100,100,scale,ox,oy),key),false);
  assert.equal(m.acceptAnchor(pointer(600,400,scale,ox,oy),key),false);
  assert.equal(m.valid(key),false);assert.equal(m.rect({left:0,top:0,w:200,h:100},key),null);
  assert.equal(m.acceptAnchor(pointer(350,250,scale,ox,oy),key),true);
  assert.ok(m.valid(key));
};
const closeRect=(actual,expected)=>Object.entries(expected).forEach(([axis,value])=>assert.ok(Math.abs(actual[axis]-value)<1e-7,axis));

test('entering and exiting fullscreen never reuses an old origin and offers measured recovery',()=>{
  const m=new ScreenPointMapping(), win=windowFixture(), key=mappingIdentity(win);
  recover(m,key);
  const windowed=videoContentRect({left:22,top:633,width:920,height:516.625},1280,800);
  closeRect(m.rect(windowed,key),{x:207.7,y:719,w:826.6,h:516.625});
  win.fullscreen=true;win.innerWidth=1710;win.innerHeight=1073;win.visualViewport.width=1710;win.visualViewport.height=1073;
  const fullscreenKey=mappingIdentity(win);
  assert.equal(m.rect(windowed,fullscreenKey),null);
  assert.equal(m.audit(fullscreenKey).reason,'window-display-or-zoom-changed');
  recover(m,fullscreenKey,1,0,0);
  const full=videoContentRect({left:0,top:0,width:1710,height:1073},1280,800);
  assert.deepEqual(m.rect(full,fullscreenKey),{x:0,y:2.125,w:1710,h:1068.75});
  assert.equal(m.rect(windowed,key),null); // Exit requires a fresh origin too.
  recover(m,key);closeRect(m.rect(windowed,key),{x:207.7,y:719,w:826.6,h:516.625});
});

for (const change of ['move','resize','dpr','screen','same-size-display','pinch','visual-scroll']) test(`${change} invalidates immediately without pointer input`,()=>{
  const win=windowFixture(), m=new ScreenPointMapping();recover(m,mappingIdentity(win));
  if(change==='move')win.screenY=35;
  if(change==='resize')win.innerWidth=win.outerWidth=900;
  if(change==='dpr')win.devicePixelRatio=1.25;
  if(change==='screen')win.screen.width=1920;
  if(change==='same-size-display')win.screen.availLeft=-1710;
  if(change==='pinch')win.visualViewport.scale=1.2;
  if(change==='visual-scroll')win.visualViewport.offsetTop=20;
  const key=mappingIdentity(win);m.sync(key);
  assert.equal(m.valid(key),false);assert.equal(m.audit(key).samples,0);
  assert.equal(m.audit(key).reason,'window-display-or-zoom-changed');
});

test('missing/repeated anchors, mismatched verification, reflected and anisotropic transforms abstain',()=>{
  for (const invalid of [pointer(100,100),{...pointer(350,250),screenX:999},pointer(350,250,-1),{...pointer(350,250),screenY:500}]) {
    const m=new ScreenPointMapping();m.beginRecovery('s');m.acceptAnchor(pointer(100,100),'s');
    assert.equal(m.valid('s'),false);m.acceptAnchor(pointer(600,400),'s');
    if(invalid.clientX===100) {m.reset('s','retry');m.beginRecovery('s');m.acceptAnchor(invalid,'s');m.acceptAnchor(invalid,'s');}
    assert.equal(m.acceptAnchor(invalid,'s'),false);assert.equal(m.valid('s'),false);
  }
  const m=new ScreenPointMapping();m.beginRecovery('s');m.acceptAnchor(pointer(100,100),'s');
  assert.equal(m.valid('s'),false);assert.equal(m.audit('s').recoverySamples,1);
  m.sync('other-display');assert.equal(m.audit('other-display').recoveryActive,false);
  assert.equal(m.acceptAnchor(pointer(600,400),'other-display'),false);
});

test('trusted mapping never bootstraps a new scale from pointer movement or guessed DPR',()=>{
  const m=new ScreenPointMapping();recover(m,'s',1.25);
  assert.ok(m.validatePointer(pointer(800,500,1.25,139,86),'s'));
  assert.equal(m.validatePointer(pointer(800,500,2,139,86),'s'),false);
  assert.equal(m.valid('s'),false);assert.equal(m.audit('s').reason,'pointer-transform-changed');
  assert.equal(m.validatePointer(pointer(900,600,2,139,86),'s'),false);
  assert.equal(m.valid('s'),false);
});

test('scroll and ancestor clipping keep full normalized geometry and separate visible bounds',()=>{
  const m=new ScreenPointMapping();recover(m,'s',.8,-1200,30);
  const content=videoContentRect({left:20,top:-100,width:800,height:500},1280,800);
  const visible=visibleContentRect(content,[{left:20,top:-100,w:800,h:500},{left:0,top:0,w:900,h:600},{left:100,top:20,w:600,h:200}]);
  assert.deepEqual(m.rect(content,'s'),{x:-1184,y:-50,w:640,h:400});
  assert.deepEqual(m.rect(visible,'s'),{x:-1120,y:46,w:480,h:160});
  const actualGaze={x:-1150,y:0};
  const full=m.rect(content,'s'), crop=m.rect(visible,'s');
  assert.ok((actualGaze.x-full.x)/full.w>=0);
  assert.ok(actualGaze.x<crop.x); // Inside full image but clipped out of view.
  assert.deepEqual(actualGaze,{x:-1150,y:0});
  assert.equal(m.audit('s').eyeCalibrationVerified,false);
});
