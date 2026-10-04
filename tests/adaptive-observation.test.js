import test from 'node:test';
import assert from 'node:assert/strict';
import {currentObservation} from '../frontend/adaptive-observation.js';
test('overlay and dashboard reject previous clip/session, future and stale observations',()=>{
 const p={session_id:'s',clip_id:'a',t:20,valid:true,target:'Patrick'};
 assert.equal(currentObservation(p,'s','a',20.1),p);
 for(const [s,c,t] of [['s','b',20.1],['new','a',20.1],['s','a',21],['s','a',19]])assert.equal(currentObservation(p,s,c,t),null);
 assert.equal(currentObservation(null,'s','a',20.1),null);
 const invalid={...p,valid:false};assert.equal(currentObservation(invalid,'s','a',20.1),invalid);
});
