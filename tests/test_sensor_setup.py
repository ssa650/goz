import asyncio
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from uuid import uuid4

import numpy as np
import pytest

from backend.adaptive.sensors import EegFeed, GazeFeed, EEG_RATE, _consume_muse
from backend.clips import ClipService
from backend.fal_adapter import FalError
from backend.sensor_setup import SensorSetup
from test_backend import harness, png


def wave(feed, start, seconds, kind='clean'):
    for block in range(int(seconds * 4)):
        stamps = start + block / 4 + np.arange(64) / EEG_RATE
        x = 8 * np.sin(2 * math.pi * 10 * stamps) + 3 * np.sin(2 * math.pi * 20 * stamps)
        if kind == 'flat': x *= 0
        if kind == 'clipped': x += 990
        if kind == 'mains': x += 40 * np.sin(2 * math.pi * 60 * stamps)
        if kind == 'movement': x += 120 * np.sin(2 * math.pi * 2 * stamps)
        feed.push(np.stack([x] * 4, axis=1).tolist(), stamps.tolist())


def sensors():
    return SimpleNamespace(gaze=GazeFeed(), eeg=EegFeed(), gaze_mode='gazekit', eeg_mode='muse', gaze_error=None)


def test_clip_list_annotations_resolve_without_shadowing_builtin_list():
    assert get_type_hints(ClipService.list)['return'] == list[dict]
    assert get_type_hints(ClipService.reorder)['clip_ids'] == list[str]
    assert get_type_hints(ClipService.import_prompts)['return'] == list[dict]


@pytest.mark.parametrize('kind', ['flat', 'clipped', 'mains', 'movement'])
def test_muse_calibration_rejects_bad_signals(kind):
    feed = EegFeed()
    feed.connected('Muse-device')
    feed.begin_calibration(seconds=1)
    start = feed.calibration_started
    wave(feed, start, 5, kind)
    assert feed.calibration is None and not feed.calibration_samples
    assert feed.status()['qualityError']


def test_muse_baseline_requires_clean_duration_freezes_and_invalidates_on_gap():
    feed = EegFeed()
    feed.connected('Muse-device')
    feed.begin_calibration(seconds=3)
    start = feed.calibration_started
    wave(feed, start, 4)
    assert feed.calibration is None
    wave(feed, start + 4, 3)
    assert feed.calibration['deviceId'] == 'Muse-device' and feed.calibration['cleanSeconds'] >= 3
    baseline = dict(feed.calibration)
    wave(feed, start + 7, 4)
    assert feed.calibration == baseline
    wave(feed, start + 15, 3)
    assert feed.calibration is None
    feed.connected('Muse-other-device')
    assert feed.device_id == 'Muse-other-device' and not feed.calibration


def test_gaze_ignores_other_calibration_and_invalid_udp_packets():
    gaze = GazeFeed()
    gaze.expected_setup_id = 'new-calibration'
    sample = dict(t=time.time(), x=30, y=40, valid=True, face=True)
    assert not gaze.packet(sample)
    assert not gaze.packet(dict(sample, setupId='old-calibration'))
    assert not gaze.packet(dict(sample, setupId='new-calibration', x=float('nan')))
    assert not gaze.packet(dict(sample, setupId='new-calibration', t=None))
    assert gaze.packet(dict(sample, setupId='new-calibration'))
    assert len(gaze.samples) == 1


@pytest.mark.parametrize('timestamp', [123.0, 1000.0])
def test_muse_reader_selects_muse_and_handles_lsl_and_unix_clocks(monkeypatch, timestamp):
    import backend.adaptive.sensors as module
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: 1000.0))
    stop, captured = threading.Event(), []
    class Info:
        def __init__(self, name): self.label = name
        def name(self): return self.label
        def channel_count(self): return 5
        def nominal_srate(self): return 256
        def source_id(self): return 'Muse-device'
    class Inlet:
        def __init__(self, info, **kwargs): assert info.name() == 'Muse'
        def pull_chunk(self, timeout):
            stop.set()
            return [[1, 2, 3, 4, 999]], [timestamp]
        def time_correction(self): return 0.0
        def close_stream(self): captured.append('closed')
    pylsl = SimpleNamespace(resolve_byprop=lambda *a, **k: [Info('Unrelated EEG'), Info('Muse')],
                            StreamInlet=Inlet, local_clock=lambda: 123.0)
    feed = SimpleNamespace(connected=lambda value: captured.append(value), disconnected=lambda: None,
                           push=lambda samples, stamps: captured.append((samples, stamps)), state='')
    _consume_muse(feed, stop, pylsl)
    assert captured == ['Muse-device', ([[1, 2, 3, 4]], [1000.0]), 'closed']


class Child:
    next_pid = 900000
    def __init__(self, code=None):
        Child.next_pid += 1
        self.pid, self.returncode = Child.next_pid, code
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.ended = asyncio.Event()
        if code is not None: self.ended.set()
    async def wait(self):
        await self.ended.wait()
        return self.returncode


@pytest.mark.asyncio
@pytest.mark.parametrize('passed', [True, False])
async def test_owned_startup_calibrates_both_then_unblocks_and_cleans_up(tmp_path, monkeypatch, passed):
    monkeypatch.setenv('GOZ_REQUIRE_SENSORS', '1')
    repo = tmp_path / 'gazekit'
    (repo / 'gazekit').mkdir(parents=True)
    (repo / 'gazekit' / 'calibrate.py').touch()
    monkeypatch.setenv('GOZ_GAZEKIT_DIR', str(repo))
    interpreter = tmp_path / 'venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv('GOZ_GAZEKIT_PYTHON', str(interpreter))
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-reach-child')
    monkeypatch.setenv('FAL_KEY', 'must-not-reach-child')
    sn, calls, children, signals = sensors(), [], {}, []
    async def spawn(*argv, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'FAL_KEY' not in kwargs['env']
        assert kwargs['start_new_session'] is True
        calls.append(argv)
        if 'backend.gaze_worker' in argv:
            assert argv[0] == str(interpreter)
        if 'calibrate' in argv:
            child = Child(0)
            Path(argv[argv.index('--model') + 1]).write_bytes(b'validated local model')
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps(dict(verdict='STABLE' if passed else 'POOR')))
        else: child = Child()
        children[child.pid] = child
        return child
    def killpg(pid, sig):
        signals.append(pid)
        children[pid].returncode = -15
        children[pid].ended.set()
    monkeypatch.setattr('backend.sensor_setup.os.killpg', killpg)
    setup = SensorSetup(sn, tmp_path, spawn=spawn)
    async def wait_for(predicate, timeout, child=None):
        if setup.phase == 'connecting_muse':
            if not calls: raise ValueError('No pre-existing bridge')
            sn.eeg.connected('Muse-owned-device')
        elif setup.phase == 'calibrating_muse':
            sn.eeg.calibration_started -= 70
            wave(sn.eeg, time.time() - 65, 65)
        elif setup.phase == 'starting_gaze':
            assert '--setup-id' in calls[-1]
            sn.gaze.packet(dict(t=time.time(), x=100, y=100, valid=True, face=True, setupId=setup.attempt_dir.name))
        assert predicate()
    setup.wait_for = wait_for
    await setup.start()
    await setup.task
    if passed:
        assert setup.snapshot()['generationReady'] and setup.phase == 'ready'
        assert '--backend' in calls[0] and 'bleak' in calls[0] and '--lsltime' in calls[0]
        assert [argv[4] for argv in calls[1:]] == ['calibrate', 'stream']
        assert (setup.attempt_dir / 'muse.json').exists()
        setup.require_ready()
        sn.gaze.last_rx = 0
        assert not setup.snapshot()['generationReady'] and setup.snapshot()['canRetry']
        with pytest.raises(FalError): setup.require_ready()
    else:
        assert setup.phase == 'failed' and 'poor' in setup.error.lower()
        assert len(calls) == 2 and not setup.snapshot()['generationReady']
    # Recalibration must reuse the live Muse bridge and replace only the gaze worker.
    bridge = setup.children['muse']
    await setup.retry()
    await setup.task
    assert setup.children['muse'] is bridge
    assert sum('muselsl' in argv for argv in calls) == 1
    await setup.close()
    assert not setup.children and signals


@pytest.mark.asyncio
async def test_generation_routes_cannot_bypass_sensor_gate(tmp_path):
    async with harness(tmp_path) as (engine, adapter, client):
        card = (await client.post('/api/clips', json={'prompt':'A scene'})).json()
        setup = client._transport.app.state.sensors.setup
        setup.required = True
        setup.set_phase('calibrating_muse', 'Finish Muse and gaze calibration first.')
        response = await client.post('/api/sequences', json=dict(id=str(uuid4()), clips=[dict(id=card['id'], order=0,
            prompt=card['prompt'], seed=card['seed'], firstFrame=None, endFrame=None,
            promptExpansionMode='disabled', duration=15, resolution='480P')]))
        assert response.status_code == 409
        response = await client.post(f"/api/clips/{card['id']}/generate", json={'token':str(uuid4())})
        assert response.status_code == 409
        response = await client.post('/api/jobs', data=dict(mode='frames', prompt='Legacy', duration='5', resolution='480P'),
                                     files=[('start',('start.png',png(),'image/png')),('end',('end.png',png(),'image/png'))])
        assert response.status_code == 409
        response = await client.post('/api/sequences', data=dict(id=str(uuid4()), mode='chain', prompts='Legacy', duration='5', resolution='480P'),
                                     files=[('start',('start.png',png(),'image/png'))])
        assert response.status_code == 409
        response = await client.post('/api/adaptive/sessions', data={}, files=[('start',('start.png',png(),'image/png'))])
        assert response.status_code == 409
        assert not adapter.submissions and not engine.jobs


@pytest.mark.asyncio
async def test_server_sensor_status_and_retry_busy_guard(tmp_path):
    async with harness(tmp_path) as (engine, adapter, client):
        state = (await client.get('/api/sensors')).json()
        assert state['phase'] == 'disabled' and state['generationReady']
        assert (await client.get('/api/config')).json()['sensorSetup'] == state
        engine.new_job(dict(mode='text', prompt='Existing', duration=5, resolution='480P'))
        response = await client.post('/api/sensors/setup')
        assert response.status_code == 409
