"""Explicit calibration actions, with offline HTTP and no device operations."""
from copy import deepcopy
from types import SimpleNamespace
import time

import pytest
from backend.sensor_setup import SensorSetup
from test_backend import harness
from test_cumulative_eeg_policy import quality
from test_sensor_setup import sensors


@pytest.mark.asyncio
async def test_explicit_calibration_start_repeated_click_and_gaze_only_do_not_restart_hardware(tmp_path,monkeypatch):
    async with harness(tmp_path) as (_,_,client):
        sn=client._transport.app.state.sensors
        calls=[];q=quality(calibrated=False,cleanSeconds=22.)
        sn.eeg.calibration=None
        monkeypatch.setattr(sn.eeg,'status',lambda:deepcopy(q))
        def begin(seconds):
            calls.append(seconds);sn.eeg.calibration_started=123.;q['cleanSeconds']=0.
        monkeypatch.setattr(sn.eeg,'begin_calibration',begin)
        response=await client.post('/api/adaptive/muse/calibration',json={'action':'start_or_recalibrate'})
        assert response.status_code==200 and calls==[60.]
        response=await client.post('/api/adaptive/muse/calibration',json={'action':'start_or_recalibrate'})
        assert response.json()['alreadyInProgress'] and calls==[60.]
        before=deepcopy(q)
        response=await client.post('/api/adaptive/muse/calibration',json={'action':'gaze_only'})
        assert response.json()['gazeOnly'] and sn.eeg_gaze_only and q==before and calls==[60.]
        # Re-entering the same in-progress calibration enables EEG without reset.
        response=await client.post('/api/adaptive/muse/calibration',json={'action':'start_or_recalibrate'})
        assert not response.json()['gazeOnly'] and calls==[60.]
        status=(await client.get('/api/adaptive/muse/calibration')).json()
        assert status['targetSeconds']==60. and status['protocol'].startswith('eyes-open')


@pytest.mark.asyncio
@pytest.mark.parametrize('action',['start_or_recalibrate','gaze_only'])
async def test_active_run_and_busy_generation_guard_calibration_actions(tmp_path,monkeypatch,action):
    async with harness(tmp_path) as (engine,_,client):
        sn=client._transport.app.state.sensors;calls=[]
        monkeypatch.setattr(sn.eeg,'begin_calibration',lambda *a:calls.append(a))
        sn.session=SimpleNamespace(status='running')
        response=await client.post('/api/adaptive/muse/calibration',json={'action':action})
        assert response.status_code==409 and not calls
        sn.session=None;engine.new_job(dict(mode='text',prompt='Pending',duration=5,resolution='480P'))
        response=await client.post('/api/adaptive/muse/calibration',json={'action':action})
        assert response.status_code==409 and not calls


@pytest.mark.asyncio
async def test_missing_live_muse_and_unsupported_sources_cannot_start_timer(tmp_path,monkeypatch):
    async with harness(tmp_path) as (_,_,client):
        sn=client._transport.app.state.sensors;calls=[]
        monkeypatch.setattr(sn.eeg,'begin_calibration',lambda *a:calls.append(a))
        for q in (quality(source='mindmonitor'),quality(live=False)):
            monkeypatch.setattr(sn.eeg,'status',lambda:deepcopy(q))
            response=await client.post('/api/adaptive/muse/calibration',json={'action':'start_or_recalibrate'})
            assert response.status_code==409
        assert not calls
        assert (await client.post('/api/adaptive/muse/calibration',json={'action':'invalid'})).status_code==400


def test_gaze_only_readiness_keeps_every_existing_gaze_gate(tmp_path):
    sn=sensors();sn.eeg_gaze_only=True
    setup=SensorSetup(sn,tmp_path);setup.required=True;setup.phase='ready';setup.gaze_calibrated=True
    setup.children['gaze']=SimpleNamespace(returncode=None)
    sn.gaze.add(dict(t=time.time(),valid=True,face=True));sn.gaze.last_rx=time.time()
    before=sn.eeg.calibration
    assert setup.snapshot()['generationReady'] and setup.snapshot()['gazeOnly']
    assert sn.eeg.calibration==before
    sn.eeg_gaze_only=False
    assert not setup.snapshot()['generationReady']
    sn.eeg_gaze_only=True;setup.gaze_calibrated=False
    assert not setup.snapshot()['generationReady']
    setup.gaze_calibrated=True;setup.children['gaze'].returncode=1
    assert not setup.snapshot()['generationReady']
