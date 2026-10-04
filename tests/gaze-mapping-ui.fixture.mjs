// Isolated headless fixture; no app backend, real media, sensors or user tabs.
// Run with installed Playwright: node tests/gaze-mapping-ui.fixture.mjs
// Optional GOZ_PLAYWRIGHT_MODULE and GOZ_HEADLESS_BROWSER select local runtimes.
import assert from 'node:assert/strict';
import {readFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const {chromium}=await import(process.env.GOZ_PLAYWRIGHT_MODULE || 'playwright');
const executablePath=process.env.GOZ_HEADLESS_BROWSER || (process.platform==='darwin' ?
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : undefined);
const browser=await chromium.launch({headless:true,executablePath});
try {
  const context=await browser.newContext({viewport:{width:1440,height:1107},deviceScaleFactor:2});
  const page=await context.newPage(), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const state={gaze:{live:true,source:'camera',point:null},eeg:{series:[],confidence:0},response:{},session:null};
  await page.route('**/*',async route=>{
    const url=new URL(route.request().url());
    if(url.origin!=='http://mapping.fixture')return route.abort();
    if(url.pathname==='/api/config')return route.fulfill({json:{configured:false,demo:true,durations:[10]}});
    if(url.pathname==='/api/adaptive/state')return route.fulfill({json:{...state,serverNow:Date.now()/1000}});
    if(url.pathname.startsWith('/api/'))return route.fulfill({json:{}});
    const filename=path.basename(url.pathname);
    if(!/\.(html|js|css)$/.test(filename))return route.abort();
    let body=await readFile(path.join(root,'frontend',filename),'utf8');
    if(filename==='adaptive.js')body+='\nwindow.mappingFixture={screenRect,mappingAudit,mappingAllowsObservation,renderMappingStatus,render};';
    return route.fulfill({body,contentType:filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.css') ? 'text/css' : 'text/html'});
  });
  await page.goto('http://mapping.fixture/adaptive.html');
  await page.waitForFunction(()=>!!window.mappingFixture);
  await page.evaluate(()=>{
    for(const id of ['video','video-next'])Object.defineProperties(document.getElementById(id),{
      videoWidth:{get:()=>1280,configurable:true},videoHeight:{get:()=>800,configurable:true}});
    document.getElementById('video').hidden=false; // Present an isolated fixture frame.
    window.mappingFixture.renderMappingStatus();
  });
  const audit=()=>page.evaluate(()=>window.mappingFixture.mappingAudit());
  const map=async()=>{
    await page.locator('#mapping-start').click();
    for(let i=0;i<3;i++) {
      assert.equal((await audit()).valid,false);
      await page.locator('#mapping-target').click();
    }
    const mapped=await audit();assert.equal(mapped.valid,true,JSON.stringify(mapped));
    assert.equal((await audit()).eyeCalibrationVerified,false);
    const rectangles=await page.evaluate(()=>window.mappingFixture.screenRect());
    assert.ok(rectangles.rect.w>0 && rectangles.visible_rect.h>0);
  };
  assert.equal((await audit()).valid,false);
  assert.match(await page.locator('#mapping-message').textContent(),/Click Map gaze to video/);
  await page.locator('#mapping-start').click();await page.locator('#mapping-target').click();
  assert.equal((await audit()).valid,false); // Missing anchors never map.
  await page.locator('#mapping-cancel').click();await map();
  await page.locator('#fullscreen').click();
  await page.waitForFunction(()=>!!document.fullscreenElement);
  await page.locator('#mapping-exit-fullscreen').waitFor({state:'visible'});
  assert.equal((await audit()).valid,false);
  assert.equal((await audit()).fullscreen,true);
  assert.equal(await page.locator('#mapping-panel').isVisible(),true);
  assert.equal(await page.locator('#mapping-exit-fullscreen').isVisible(),true);
  assert.equal(await page.evaluate(()=>document.getElementById('screen').contains(document.getElementById('mapping-panel'))),true);
  assert.equal(await page.evaluate(()=>window.mappingFixture.mappingAllowsObservation({t:Date.now()/1000-1})),false);
  await page.locator('#mapping-start').click();
  await mkdir(path.join(root,'output/playwright'),{recursive:true});
  await page.screenshot({path:path.join(root,'output/playwright/gaze-mapping-fullscreen-recovery.png')});
  for(let i=0;i<3;i++)await page.locator('#mapping-target').click();
  assert.equal((await audit()).valid,true);
  assert.equal(await page.evaluate(()=>window.mappingFixture.mappingAllowsObservation({t:Date.now()/1000-1})),false);
  assert.equal(await page.evaluate(()=>window.mappingFixture.mappingAllowsObservation({t:Date.now()/1000+.1})),true);
  await page.locator('#mapping-exit-fullscreen').click();
  await page.waitForFunction(()=>!document.fullscreenElement);
  assert.equal((await audit()).valid,false);await map();
  await page.setViewportSize({width:1280,height:900});
  assert.equal((await audit()).valid,false);await map();
  const cdp=await context.newCDPSession(page);
  await cdp.send('Emulation.setPageScaleFactor',{pageScaleFactor:1.25});
  assert.equal((await audit()).valid,false);assert.equal((await audit()).reason,'visual-viewport-scaled');
  await page.evaluate(()=>window.mappingFixture.renderMappingStatus());
  assert.equal(await page.locator('#mapping-start').isDisabled(),true);
  await cdp.send('Emulation.setPageScaleFactor',{pageScaleFactor:1});
  assert.equal((await audit()).valid,false);await map();
  await page.evaluate(()=>{Object.defineProperty(window,'screenX',{value:-1710,configurable:true});window.mappingFixture.renderMappingStatus();});
  assert.equal((await audit()).valid,false);assert.equal((await audit()).windowX,-1710);
  await map();
  await page.evaluate(()=>{Object.defineProperty(window,'devicePixelRatio',{value:1.25,configurable:true});window.mappingFixture.renderMappingStatus();});
  assert.equal((await audit()).valid,false);await map();
  assert.equal((await audit()).scale,1); // Canvas DPR never supplies pointer scale.
  await page.evaluate(()=>{
    document.getElementById('screen').style.marginTop='180px';window.mappingFixture.renderMappingStatus();
  });
  const clipped=await page.evaluate(()=>window.mappingFixture.screenRect());
  assert.ok(clipped.visible_rect.h<clipped.rect.h);
  assert.equal(await page.evaluate(()=>window.mappingFixture.mappingAllowsObservation({t:Date.now()/1000-1})),false);
  await page.evaluate(()=>window.scrollBy(0,100));
  const scrolled=await page.evaluate(()=>window.mappingFixture.screenRect());
  assert.equal((await audit()).valid,true);assert.ok(scrolled.rect.y<clipped.rect.y);
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.evaluate(()=>{document.getElementById('screen').style.marginTop='2000px';window.mappingFixture.renderMappingStatus();});
  assert.equal((await audit()).valid,false);assert.equal((await audit()).reason,'video-not-visible');
  assert.equal(await page.evaluate(()=>window.mappingFixture.screenRect()),null);
  await page.evaluate(()=>{
    document.getElementById('screen').style.marginTop='0';
    window.mappingFixture.render({gaze:{live:true,point:null},eeg:{series:[],confidence:0},response:{},session:{
      id:'fixture',status:'stopped',stage:'Fixture',names:['A','B'],clips:[],story:{premise:'Fixture',scenes:[]},
      profile:{clips:4,characters:{A:.5,B:.5},pacing:0,dialogue:0,genres:{}},duration:10}});
  });
  assert.equal(await page.locator('#profile-clips').textContent(),'4 scenes evaluated');
  assert.match(await page.locator('#affinity').textContent(),/Starting profile weights/);
  assert.match(await page.locator('#affinity').textContent(),/do not mean equal attention/);
  assert.equal(await page.locator('#response-strength').textContent(),'—');
  assert.deepEqual(errors,[]);
  console.log('PASS isolated headless fullscreen entry/exit, three-click recovery, missing anchors, resize, window move, DPR, pinch zoom, scroll/clipping, stale gaze and no-evidence profile');
  await context.close();
} finally {await browser.close();}
