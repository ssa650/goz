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
    return SimpleNamespace(gaze=GazeFeed(), eeg=EegFeed(), gaze_mode='gazekit', eeg_mode='muse', gaze_error=None, start_muse_reader=lambda: None)


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


def test_gaze_rejects_truthy_strings_in_validity_flags():
    gaze = GazeFeed()
    sample = dict(t=time.time(), x=-22, y=400, valid="false", face=True)
    assert not gaze.packet(sample)
    assert gaze.packet(dict(sample, valid=True))  # Outside-screen values remain outside.
    assert gaze.latest()["x"] == -22


def test_saved_calibration_paths_are_cwd_independent_and_profile_separated(tmp_path, monkeypatch):
    from backend.sensor_setup import ROOT
    monkeypatch.chdir(tmp_path)
    setup = SensorSetup(sensors(), "data")
    assert setup.directory == ROOT / "data"
    monkeypatch.setenv("GOZ_VIEWER_PROFILE", "another viewer")
    other = SensorSetup(sensors(), "data")
    assert other.saved_gaze_path != setup.saved_gaze_path
    assert other.snapshot()["gaze"]["profileId"] == "another viewer"


@pytest.mark.asyncio
async def test_muse_connection_is_user_initiated_and_duplicate_safe(monkeypatch):
    from backend.adaptive.routes import Sensors
    monkeypatch.setenv('GOZ_EEG', 'muse')
    monkeypatch.setenv('GOZ_GAZE', 'sim')
    sn = Sensors(demo=True)
    async def no_setup(): pass
    sn.setup.start = no_setup
    await sn.start()
    assert sn.muse_thread is None and sn.eeg.source == 'off' and sn.sim_task is not None
    await sn.close()

    setup = SensorSetup(sensors(), 'data')
    gate = asyncio.Event()
    attempts = []
    async def pending(generation):
        attempts.append(generation)
        await gate.wait()
    setup._connect_muse = pending
    await setup.connect_muse()
    await asyncio.sleep(0)
    await setup.connect_muse()
    assert len(attempts) == 1 and setup.manual_muse_state == 'connecting'
    await setup.disconnect_muse()
    assert setup.manual_muse_state == 'disconnected'
    gate.set()
    setup.manual_muse_state = 'error'
    gate.clear()
    await setup.connect_muse()
    await asyncio.sleep(0)
    assert len(attempts) == 2 and setup.manual_muse_state == 'connecting'
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_manual_disconnect_does_not_kill_external_bridge(monkeypatch):
    setup = SensorSetup(sensors(), 'data')
    killed = []
    monkeypatch.setattr('backend.sensor_setup.os.killpg', lambda *args: killed.append(args))
    setup.sensors.eeg.device_id = 'externally-published-muse'
    await setup.disconnect_muse()
    assert not killed
    assert setup.sensors.eeg.device_id is None


def test_gaze_metadata_rejects_changed_viewer_display_schema_and_missing_camera(monkeypatch):
    import backend.gaze_worker as worker
    monkeypatch.setattr(worker, "model_schema", lambda repo: "features-v5")
    report = dict(schemaVersion=3, profileId="viewer", modelSchema="features-v5", screen=[1710, 1107], cameraName="FaceTime",
                  cameraDeviceId="mac", cameraIdentityVerified=True, cameraBackend="avfoundation-uid")
    worker.validate_compatibility(report, "unused", "viewer", (1710, 1107))
    for fields, message in [({"profileId":"other"}, "viewer profile"), ({"screen":[1000, 800]}, "geometry changed"),
                            ({"modelSchema":"old"}, "schema changed"), ({"schemaVersion":2}, "predates verified"),
                            ({"cameraDeviceId":None}, "camera identity"), ({"cameraIdentityVerified":False}, "camera identity")]:
        with pytest.raises(ValueError, match=message):
            worker.validate_compatibility(dict(report, **fields), "unused", "viewer", (1710, 1107))


@pytest.mark.asyncio
async def test_calibration_registration_does_not_wait_for_results_screen(tmp_path):
    model = tmp_path / "calibration" / "attempt" / "gaze_model.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"local-model")
    child = Child()
    child.stdout = asyncio.StreamReader()
    child.stdout.feed_data(("GOZ_CALIBRATION_SAVED " + json.dumps(dict(verdict="STABLE", camera="0")) + "\n").encode())
    child.stdout.feed_eof()
    async def spawn(*args, **kwargs): return child
    setup = SensorSetup(sensors(), tmp_path, spawn=spawn)
    setup.attempt_dir = model.parent
    await setup.launch("gaze_calibration", ["test"], tmp_path)
    await setup.readers["gaze_calibration"]
    assert child.returncode is None  # Native results screen still open.
    assert setup.saved_gaze()[0] == model
    assert setup.snapshot()["gaze"]["calibrationState"] == "saved"


@pytest.mark.parametrize('timestamp', [123.0, 1000.0])
def test_muse_reader_selects_muse_and_handles_lsl_and_unix_clocks(monkeypatch, timestamp):
    import backend.adaptive.sensors as module
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: 1000.0, monotonic=lambda: 100.0))
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
            return [[1, 2, 3, 4, 999]], [timestamp]
        def time_correction(self, timeout): return 0.0
        def close_stream(self): captured.append('closed')
    pylsl = SimpleNamespace(resolve_byprop=lambda *a, **k: [Info('Unrelated EEG'), Info('Muse')],
                            StreamInlet=Inlet, local_clock=lambda: 123.0)
    def push(samples, stamps):
        captured.append((samples, stamps))
        stop.set()
    feed = SimpleNamespace(connected=lambda value: captured.append(value), disconnected=lambda: None,
                           push=push, state='')
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
@pytest.mark.parametrize('eeg_mode', ['muse', 'mindmonitor'])
async def test_owned_startup_calibrates_both_then_unblocks_and_cleans_up(tmp_path, monkeypatch, passed, eeg_mode):
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
    if eeg_mode == 'mindmonitor':
        from backend.adaptive.mindmonitor import MindMonitorFeed
        sn.eeg, sn.eeg_mode = MindMonitorFeed(), eeg_mode
    readers = []
    sn.start_muse_reader = lambda: readers.append('started')
    async def spawn(*argv, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'FAL_KEY' not in kwargs['env']
        assert kwargs['start_new_session'] is True
        calls.append(argv)
        if 'backend.gaze_worker' in argv:
            assert argv[0] == str(interpreter)
        if 'calibrate' in argv:
            child = Child(0)
            Path(argv[argv.index('--model') + 1]).write_bytes(b'validated local model')
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps(dict(verdict='STABLE' if passed else 'POOR', camera='2')))
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
            assert [argv[4] for argv in calls[:2]] == ['calibrate', 'stream']
            if eeg_mode == 'mindmonitor':
                from unittest.mock import patch
                now = time.time() - 1
                with patch('backend.adaptive.mindmonitor.time.time', return_value=now):
                    sn.eeg.receive('phone', '/muse/elements/horseshoe', 4, 1, 4, 4)
                for i in range(10):
                    with patch('backend.adaptive.mindmonitor.time.time', return_value=now + (i + 1) * .1):
                        sn.eeg.receive('phone', '/muse/elements/alpha_absolute', 0.)
                        sn.eeg.receive('phone', '/muse/elements/beta_absolute', 0.)
            else:
                if 'muse' not in setup.children: raise ValueError('No pre-existing bridge')
                sn.eeg.connected('Muse-owned-device')
        elif setup.phase == 'calibrating_muse':
            sn.eeg.calibration_started -= 70
            sn.eeg.last_eval = 0  # Test capture uses backdated samples.
            wave(sn.eeg, time.time() - 65, 65)
        elif setup.phase == 'starting_gaze':
            assert '--setup-id' in calls[-1]
            sn.gaze.packet(dict(t=time.time(), x=100, y=100, valid=True, face=True, setupId=setup.attempt_dir.name))
        assert predicate(), setup.phase
    setup.wait_for = wait_for
    await setup.start()
    await setup.task
    if passed:
        if eeg_mode == 'muse':
            # Gaze setup completes without probing/launching EEG; Muse is an explicit action.
            assert setup.phase == 'ready' and not setup.snapshot()['generationReady']
            assert readers == [] and not any('muselsl' in argv for argv in calls)
            return
        assert setup.snapshot()['generationReady'] and setup.phase == 'ready'
        if eeg_mode == 'muse':
            assert '--backend' in calls[2] and 'bleak' in calls[2] and '--lsltime' in calls[2]
        else:
            assert len(calls) == 2 and 'muse' not in setup.children
        assert [argv[4] for argv in calls[:2]] == ['calibrate', 'stream']
        assert calls[0][calls[0].index('--camera') + 1] == 'select'
        assert calls[1][calls[1].index('--camera') + 1] == '2'
        assert readers == ['started']
        assert (setup.attempt_dir / 'muse.json').exists()
        assert setup.snapshot()['gaze']['savedCalibration']
        setup.require_ready()
        sn.gaze.last_rx = 0
        assert not setup.snapshot()['generationReady'] and setup.snapshot()['canRetry']
        with pytest.raises(FalError): setup.require_ready()
    else:
        assert setup.phase == 'failed' and 'poor' in setup.error.lower()
        assert len(calls) == 1 and not setup.snapshot()['generationReady']
        assert not readers and 'muse' not in setup.children
    # Recalibration must reuse the live Muse bridge and replace only the gaze worker.
    bridge = setup.children.get('muse')
    await setup.retry()
    await setup.task
    if passed:
        assert sum('calibrate' in argv for argv in calls) == 1
        assert '--reuse-calibration' in next(argv for argv in reversed(calls) if 'stream' in argv)
        assert setup.snapshot()['gaze']['reusingCalibration']
    assert setup.children.get('muse') is bridge
    assert sum('muselsl' in argv for argv in calls) == (1 if passed and eeg_mode == 'muse' else 0)
    await setup.close()
    assert not setup.children
    assert bool(signals) is passed
    if passed:
        # A new setup object represents a backend restart: reuse the exact model.
        first_model = setup.saved_gaze()[0]
        sn = sensors()
        if eeg_mode == 'mindmonitor':
            sn.eeg, sn.eeg_mode = MindMonitorFeed(), eeg_mode
        sn.start_muse_reader = lambda: readers.append('started')
        setup = SensorSetup(sn, tmp_path, spawn=spawn)
        setup.wait_for = wait_for
        await setup.start()
        await setup.task
        assert setup.phase == 'ready' and setup.reusing_gaze, setup.error
        assert setup.saved_gaze()[0] == first_model
        assert sum('calibrate' in argv for argv in calls) == 1
        assert setup.snapshot()['generationReady']
        # Removal immediately invalidates gaze and starts a fresh calibration.
        await setup.remove_gaze_calibration()
        assert not setup.gaze_calibrated and not setup.saved_gaze_path.exists()
        await setup.task
        assert setup.phase == 'ready' and not setup.reusing_gaze
        assert sum('calibrate' in argv for argv in calls) == 2
        assert setup.saved_gaze()[0] != first_model
        await setup.close()


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
        response = await client.delete('/api/sensors/gaze-calibration')
        assert response.status_code == 409


@pytest.mark.asyncio
async def test_remove_eye_calibration_api_clears_saved_state(tmp_path):
    async with harness(tmp_path) as (_, _, client):
        setup = client._transport.app.state.sensors.setup
        model = setup.directory / 'calibration' / 'validated' / 'gaze_model.pkl'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'validated model')
        setup.save_gaze(model, dict(verdict='STABLE', camera='2'))
        assert (await client.get('/api/sensors')).json()['gaze']['savedCalibration']
        response = await client.delete('/api/sensors/gaze-calibration')
        assert response.status_code == 202
        assert not response.json()['gaze']['savedCalibration']
        assert SensorSetup(sensors(), setup.directory).saved_gaze() is None


@pytest.mark.parametrize('damage', ['model', 'metadata', 'outside'])
def test_damaged_saved_calibration_requires_explicit_removal(tmp_path, damage):
    setup = SensorSetup(sensors(), tmp_path)
    model = tmp_path / 'calibration' / 'validated' / 'gaze_model.pkl'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'validated model')
    setup.save_gaze(model, dict(verdict='USABLE', camera='1'))
    if damage == 'model':
        model.write_bytes(b'changed')
    elif damage == 'metadata':
        setup.saved_gaze_path.write_text('{broken')
    else:
        data = json.loads(setup.saved_gaze_path.read_text())
        data['model'] = '../outside.pkl'
        setup.saved_gaze_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='full eye recalibration'):
        SensorSetup(sensors(), tmp_path).saved_gaze()
    assert setup.saved_gaze_path.exists()  # Never silently replace saved state.


def test_saved_camera_follows_device_when_indexes_change():
    from backend.gaze_worker import saved_camera
    cameras = [dict(index=0, name='Other', deviceId='other'), dict(index=3, name='iPhone Camera', deviceId='phone')]
    assert saved_camera(cameras, 'phone', 'iPhone Camera') is cameras[1]
    assert saved_camera(cameras, None, 'iPhone Camera') is cameras[1]
    with pytest.raises(ValueError, match='Reconnect'):
        saved_camera(cameras, 'missing', 'iPhone Camera')


def test_loaded_predictor_preserves_outside_screen_coordinates(tmp_path, monkeypatch):
    """Regression: both Gazekit.predict and stream previously clipped to edges."""
    from backend.gaze_worker import load_predictor
    ridge = SimpleNamespace(screen_size=(100, 80), bias=np.zeros(2),
                            pipe=SimpleNamespace(predict=lambda features: np.array([[-20., 115.]])))
    module = SimpleNamespace(GazeModel=SimpleNamespace(load=lambda path: ridge), transform=lambda features: features)
    monkeypatch.setitem(sys.modules, 'gazekit.model', module)
    predict, loaded = load_predictor(tmp_path / 'model.pkl', (100, 80))
    assert loaded is ridge
    assert predict(SimpleNamespace(ok=True, features=np.zeros(14))).tolist() == [-20., 115.]
    assert predict(SimpleNamespace(ok=False)) is None
    with pytest.raises(ValueError, match='geometry'):
        load_predictor(tmp_path / 'model.pkl', (200, 80))


def test_alignment_survives_restart_without_opening_targets(tmp_path):
    from backend.gaze_worker import configure_alignment
    model = tmp_path / 'gaze_model.pkl'
    initial = SimpleNamespace(_quick_align=lambda *a: (1.2, 10., .8, -5.))
    assert configure_alignment(initial, model, False)
    initial._quick_align(None)
    resumed = SimpleNamespace(build_predictor=lambda *a: (lambda obs: None if obs is None else (100., 50.), 'ridge', 'active'))
    assert not configure_alignment(resumed, model, True)
    predict, ridge, active = resumed.build_predictor()
    assert predict('face') == (130., 35.) and predict(None) is None
    assert ridge == 'ridge' and active == 'active'


def test_external_camera_reference_is_shared_with_dataset(tmp_path, monkeypatch):
    from backend.gaze_worker import configure_camera
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    model = attempt / 'gaze.pkl'
    dataset = SimpleNamespace()
    monkeypatch.chdir(tmp_path)
    configure_camera(2, model, dataset)
    assert dataset.CONFIG_PATH == attempt / 'camera.json'
    assert json.loads(dataset.CONFIG_PATH.read_text()) == {'camera':'2'}
    assert Path.cwd() == attempt


@pytest.mark.parametrize('keys,expected,iphone', [([255,13], '2', True), ([255,ord('1')], '2', True), ([255,ord('2')], '0', True),
                                               ([255,27], None, True), ([255,13,27], None, False)])
def test_gazekit_native_camera_selection(keys, expected, iphone, monkeypatch):
    from backend.gaze_worker import choose_camera
    import backend.gaze_worker as worker
    presses = iter(keys)
    closed = []
    class Window:
        name, w, h = 'camera-test', 1200, 800
        def canvas(self): return None
        def show(self, img): return next(presses)
        def close(self): closed.append(True)
    ui = SimpleNamespace(FullscreenWindow=lambda *a: Window(), center_text=lambda *a: None, ACCENT=1, WHITE=2)
    cv2 = SimpleNamespace(EVENT_LBUTTONDOWN=1, setMouseCallback=lambda *a: None, waitKey=lambda *a: None)
    cameras = [dict(index=0, name="FaceTime HD Camera")]
    if iphone: cameras.append(dict(index=2, name="iPhone Camera"))
    camera = SimpleNamespace(list_cameras=lambda **kw: cameras)
    monkeypatch.setattr(worker, "connected_cameras", lambda module: module.list_cameras())
    if expected is None:
        with pytest.raises(ValueError, match='cancelled'):
            choose_camera(ui, cv2, camera, lambda: (1200,800))
    else:
        assert choose_camera(ui, cv2, camera, lambda: (1200,800)) is next(c for c in cameras if str(c['index']) == expected)
    assert closed == [True]


@pytest.mark.asyncio
async def test_gazekit_screen_coordinates_arrive_in_goz_over_udp():
    import socket
    feed = GazeFeed()
    feed.expected_setup_id = 'validated-attempt'
    await feed.listen(0)
    sample = dict(t=time.time(), x=452.5, y=218.25, sw=1512, sh=982,
                  valid=True, face=True, setupId='validated-attempt')
    async def receive():
        while feed.latest() is None:
            await asyncio.sleep(0)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(json.dumps(sample).encode(), feed.transport.get_extra_info('sockname'))
        await asyncio.wait_for(receive(), 1)
        received = feed.latest()
        assert {key: received[key] for key in sample} == sample
        assert set(received) == set(sample) | {'receivedAt', 'receiptDiagnostics'}
        assert math.isfinite(received['receivedAt']) and sample['t'] <= received['receivedAt'] <= time.time()
        receipt = received['receiptDiagnostics']
        assert isinstance(receipt, dict) and set(receipt) == set(feed.status()['inputDiagnostics'])
        assert all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value >= 0 for value in receipt.values())
        assert (receipt['packets'], receipt['accepted'], receipt['rejected']) == (1, 1, 0)
        assert receipt['captureToReceiptMsMax'] == pytest.approx((received['receivedAt']-sample['t'])*1000)
        assert feed.status()['live'] and feed.status()['source'] == 'gazekit'
    finally:
        feed.close()



def test_continuity_enumeration_keeps_native_identity_without_probing(monkeypatch):
    import backend.gaze_worker as worker
    # Native order deliberately differs from sorted UID order; identity is retained.
    devices = [dict(index=0, name='iPhone Camera', deviceId='Z'),
               dict(index=1, name='FaceTime HD Camera', deviceId='A')]
    monkeypatch.setattr('backend.native_camera.list_cameras', lambda: devices)
    camera = SimpleNamespace(list_cameras=lambda **kw: pytest.fail('Do not hide cameras while awaiting their first frame'))
    if sys.platform != 'darwin': pytest.skip('macOS enumeration')
    found = worker.connected_cameras(camera)
    assert found is devices
