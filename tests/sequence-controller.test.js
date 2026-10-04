import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SequenceController, regenerateRequest, runError, formatTime } from '../frontend/sequence-controller.js';

const storage=()=>{
  const data=new Map();return {getItem:key=>data.get(key)||null,setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key)};
};
const clips=()=>Array.from({length:12},(_,order)=>({id:`clip-${order}`,order,prompt:`Prompt ${order}`,seed:100+order,
  firstFrame:{id:`first-${order}`},endFrame:{id:`end-${order}`},duration:15,resolution:'480P',promptExpansionMode:'disabled'}));
const runFor=(request,status='generating')=>({...structuredClone(request),mode:'bundles',status,generationBusy:status==='generating',finalVideoUrl:null,
  clips:request.clips.map(c=>({...c,jobId:`job-${c.order}`,status:'queued',generatedVideoUrl:null,result:null}))});
const config={configured:true,demo:true,pollMs:500};

test('Generate submits all 12 complete bundles once; polling starts live playback without starting another run',async()=>{
  const requests=[],updates=[];let run;
  const request=async(path,options)=>{
    if(path==='/api/config')return config;
    if(path==='/api/clips')return clips();
    if(path==='/api/sequences'&&!options)return [];
    if(options){const body=JSON.parse(options.body);requests.push(body);run=runFor(body);return run;}
    return run;
  };
  const controller=new SequenceController(()=>{},r=>updates.push(structuredClone(r)),request,storage());
  await controller.init();await Promise.all([controller.generate(),controller.generate()]);
  assert.equal(requests.length,1);assert.equal(requests[0].clips.length,12);
  assert.deepEqual(requests[0].clips.map(c=>[c.order,c.prompt,c.seed,c.firstFrame,c.endFrame]),
    clips().map(c=>[c.order,c.prompt,c.seed,c.firstFrame.id,c.endFrame.id]));
  assert.equal('prompts' in requests[0],false);
  run.clips[2].status='completed';run.clips[2].generatedVideoUrl='/clip-3';
  await controller.refresh();assert.equal(updates.at(-1).clips[0].generatedVideoUrl,null);
  run.clips[0].status='completed';run.clips[0].generatedVideoUrl='/clip-1';
  await controller.refresh();assert.equal(updates.at(-1).clips[0].generatedVideoUrl,'/clip-1');
  assert.equal(updates.at(-1).status,'generating');assert.equal(requests.length,1);
});

test('Regenerate uses the previous frozen settings and a new unique run ID',async()=>{
  const previous=runFor({id:'previous',clips:clips().map(c=>({...c,firstFrame:c.firstFrame.id,endFrame:c.endFrame.id}))},'completed');
  previous.clips.reverse();const requests=[];
  const request=async(path,options)=>{
    if(path==='/api/config')return config;if(path==='/api/clips')return clips().map(c=>({...c,seed:999}));
    if(!options)return [previous];const body=JSON.parse(options.body);requests.push(body);return runFor(body);
  };
  const controller=new SequenceController(()=>{},()=>{},request,storage());await controller.init();await controller.generate(true);
  assert.notEqual(requests[0].id,previous.id);
  assert.deepEqual(requests[0],regenerateRequest(previous,requests[0].id));
  assert.equal(requests[0].clips[0].seed,100);assert.equal(previous.clips[0].order,11);
});

test('Lost submission acknowledgement reuses the saved UUID across retry and page reload',async()=>{
  const saved=storage(),bodies=[];let accepted;
  const request=async(path,options)=>{
    if(path==='/api/config')return config;if(path==='/api/clips')return clips();
    if(!options)return [];const body=JSON.parse(options.body);bodies.push(body);
    if(!accepted){accepted=runFor(body);throw new Error('Connection lost');}return accepted;
  };
  const controller=new SequenceController(()=>{},()=>{},request,saved);await controller.init();await controller.generate();
  assert.ok(controller.pending);assert.match(controller.error,/same run/);
  const restored=new SequenceController(()=>{},()=>{},request,saved);await restored.init();await restored.generate();
  assert.deepEqual(bodies[1],bodies[0]);assert.equal(restored.pending,null);assert.equal(saved.getItem('bundle-request'),null);
});

test('Reload reconnects an already accepted request without any POST',async()=>{
  const saved=storage(),body={id:'accepted',clips:[]};saved.setItem('bundle-request',JSON.stringify(body));
  const run=runFor(body),watched=[];
  const request=async(path,options)=>{
    assert.equal(options,undefined);
    return path==='/api/config'?config:path==='/api/clips'?clips():[run];
  };
  const controller=new SequenceController(()=>{},r=>watched.push(r),request,saved);await controller.init();
  assert.equal(controller.run.id,'accepted');assert.equal(controller.pending,null);assert.equal(watched.length,1);
});

test('Existing generic failure shows the actual billing error and timing values',()=>{
  const run=runFor({id:'failed',clips:clips()},'failed');run.error='Some clips did not complete.';
  for(const clip of run.clips)clip.error='User is locked. Reason: Exhausted balance.';
  assert.equal(runError(run),'User is locked. Reason: Exhausted balance.');
  run.clips[2].error='Invalid image';assert.match(runError(run),/Clip 3: Invalid image/);
  assert.equal(formatTime(null),'—');assert.equal(formatTime(0),'0.0s');assert.equal(formatTime(12300),'12.3s');assert.equal(formatTime(92000),'1m 32s');
});

test('Sensor calibration blocks Generate/Regenerate and polling unlocks without an existing run',async()=>{
  let ready=false;const submitted=[];
  const request=async(path,options)=>{
    if(path==='/api/config')return {...config,sensorSetup:{required:true,generationReady:ready,phase:ready?'ready':'calibrating_muse',message:'Collecting clean EEG',error:null,canRetry:false,muse:{cleanSeconds:20,targetSeconds:60,qualityError:''}}};
    if(path==='/api/clips')return clips();
    if(!options)return [];
    const body=JSON.parse(options.body);submitted.push(body);return runFor(body);
  };
  const controller=new SequenceController(()=>{},()=>{},request,storage());await controller.init();
  assert.equal(controller.generationReady,false);
  await controller.generate();assert.equal(submitted.length,0);assert.match(controller.error,/EEG/);
  ready=true;await controller.refresh();assert.equal(controller.generationReady,true);
  await controller.generate();assert.equal(submitted.length,1);
});

test('Sensor retry uses only the setup route and never starts a generation',async()=>{
  const requests=[];
  const setup={required:true,generationReady:false,phase:'connecting_muse',message:'Connecting',error:null,canRetry:false,muse:{cleanSeconds:0,targetSeconds:60,qualityError:''}};
  const request=async(path,options)=>{
    if(path==='/api/config')return {...config,sensorSetup:setup};
    if(path==='/api/clips')return clips();if(!options)return [];
    requests.push([path,options]);return setup;
  };
  const controller=new SequenceController(()=>{},()=>{},request,storage());await controller.init();
  await controller.retrySensorSetup();assert.deepEqual(requests,[['/api/sensors/setup',{method:'POST'}]]);
  assert.equal(controller.pending,null);assert.equal(controller.config.sensorSetup.phase,'connecting_muse');
});

test('Removing eye calibration updates sensor state and is blocked during generation',async()=>{
  const calls=[];
  const cleared={required:true,generationReady:false,phase:'select_camera',gaze:{savedCalibration:false,reusingCalibration:false}};
  const request=async(path,options)=>{
    calls.push([path,options]);return cleared;
  };
  const controller=new SequenceController(()=>{},()=>{},request,storage());
  controller.config={...config,sensorSetup:{...cleared,gaze:{savedCalibration:true,reusingCalibration:true}}};
  await controller.removeGazeCalibration();
  assert.deepEqual(calls,[['/api/sensors/gaze-calibration',{method:'DELETE'}]]);
  assert.equal(controller.generationReady,false);
  assert.equal(controller.config.sensorSetup.gaze.savedCalibration,false);
  controller.starting=true;
  await controller.removeGazeCalibration();
  assert.equal(calls.length,1);
});
