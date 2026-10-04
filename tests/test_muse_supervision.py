"""Intent-bound Muse supervision using fake children/LSL; no device access."""
import asyncio
import json
import signal
import threading
from types import SimpleNamespace

import pytest

from backend.adaptive.sensors import EegFeed, GazeFeed
from backend.sensor_setup import SensorSetup
from backend.muse_diagnostics import MuseDiagnostics
from backend.muse_diagnostics import producer_observation


class Child:
    next_pid = 80000

    def __init__(self):
        Child.next_pid += 1
        self.pid = Child.next_pid
        self.returncode = None
        self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
        self.ended = asyncio.Event()

    def exit(self, code):
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.ended.set()

    async def wait(self):
        await self.ended.wait()
        return self.returncode


@pytest.fixture
def rig(tmp_path, monkeypatch):
    children, signals, reads = [], [], []
    async def spawn(*args, **kwargs):
        assert kwargs['stderr'] == asyncio.subprocess.PIPE
        child = Child()
        children.append(child)
        return child
    sn = SimpleNamespace(eeg=EegFeed(), gaze=GazeFeed(), eeg_mode='muse', gaze_mode='off',
                         start_muse_reader=lambda: reads.append('start'))
    async def stop_reader():
        reads.append('stop')
        return True
    sn.stop_muse_reader = stop_reader
    setup = SensorSetup(sn, tmp_path, spawn=spawn)
    def terminate(pid, sig):
        signals.append((pid, sig))
        next(c for c in children if c.pid == pid).exit(-15)
    monkeypatch.setattr('backend.sensor_setup.os.killpg', terminate)
    return setup, children, signals, reads


@pytest.fixture
def fast_time(monkeypatch):
    import backend.sensor_setup as module
    elapsed, sleeps = [0.], []
    original_sleep = asyncio.sleep
    async def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds
        await original_sleep(0)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: elapsed[0], time=lambda: 1700000000 + elapsed[0]))
    monkeypatch.setattr(module, 'asyncio', SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith('__')}))
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    return elapsed, sleeps


async def until(predicate):
    for _ in range(3000):
        if predicate(): return
        await asyncio.sleep(0)
    pytest.fail('fake supervision did not reach expected state')


def fresh(setup):
    import time
    feed = setup.sensors.eeg
    feed.source = 'muse'
    feed.connected('Muse-fake')
    feed.push([[1, 2, 3, 4]], [time.time()])


@pytest.mark.asyncio
async def test_durable_stdout_stderr_exit_and_redaction(rig, monkeypatch):
    setup, children, signals, reads = rig
    setup.muse_intent = True
    child = await setup.launch('muse', ['fake'], '.')
    child.stdout.feed_data(b'Bluetooth connected AA:BB:CC:DD:EE:FF\n[1.0, 2.0, 3.0]\n')
    child.stderr.feed_data(b'backend failed token=hidden\n')
    child.exit(7)
    await setup.muse_exit_task
    rows = [json.loads(line) for line in setup.muse_diagnostics.path.read_text().splitlines()]
    outputs = [row for row in rows if row['event'] == 'producer_output']
    assert {row['stream'] for row in outputs} == {'stdout', 'stderr'}
    assert all(row['timestamp'] and row['at'] for row in rows)
    text = setup.muse_diagnostics.path.read_text()
    assert 'AA:BB' not in text and 'hidden' not in text and '1.0, 2.0' not in text
    assert setup.muse_exit['code'] == 7 and setup.muse_exit['reason'] == 'producer_exited'
    assert setup.muse_connection()['state'] != 'connected'
    assert setup.muse_diagnostics.path.stat().st_mode & 0o777 == 0o600
    await setup.disconnect_muse()


def test_journal_and_memory_are_bounded(tmp_path):
    journal = MuseDiagnostics(tmp_path)
    for i in range(1800): journal.record('producer_output', text='x' * 400)
    assert len(journal.events) == 80
    assert journal.path.stat().st_size <= journal.MAX_BYTES
    assert journal.path.with_suffix('.jsonl.1').stat().st_size <= journal.MAX_BYTES
    assert len(list(journal.path.parent.iterdir())) == 2


@pytest.mark.asyncio
async def test_repeated_connect_disconnect_and_cancelled_backoff(rig, fast_time):
    setup, children, signals, reads = rig
    await setup.connect_muse()
    await setup.connect_muse()
    await until(lambda: bool(children))
    assert len(children) == 1 and setup.muse_intent
    children[0].exit(9)
    await until(lambda: setup.muse_retry_count == 1)
    await setup.disconnect_muse()
    count = len(children)
    for _ in range(20): await asyncio.sleep(0)
    assert len(children) == count
    assert setup.muse_connection()['state'] == 'disconnected' and not setup.muse_intent
    assert reads[-1] == 'stop'
    await setup.connect_muse()
    await until(lambda: len(children) == count + 1)
    await setup.disconnect_muse()
    assert all(sig == signal.SIGTERM for _, sig in signals)


@pytest.mark.asyncio
async def test_external_outlet_never_spawns_or_is_killed_even_after_loss(rig, fast_time):
    setup, children, signals, reads = rig
    fresh(setup)
    await setup.connect_muse()
    await until(lambda: setup.manual_muse_state == 'connected')
    assert setup.muse_owner == 'external' and setup.muse_connection()['state'] == 'connected'
    setup.sensors.eeg.disconnected()
    await setup.muse_task
    assert setup.manual_muse_state == 'error' and setup.muse_retry_count == 3
    assert not children and not signals and setup.muse_intent
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_no_outlet_timeout_has_finite_restarts_and_backoff(rig, fast_time):
    setup, children, signals, reads = rig
    await setup.connect_muse()
    await setup.muse_task
    assert len(children) == 4 and setup.muse_retry_count == 3
    assert setup.muse_reason == 'no_outlet' and setup.manual_muse_state == 'error'
    assert [row['seconds'] for row in setup.muse_diagnostics.events if row['event'] == 'recovery_backoff'] == [2, 4, 8]
    assert setup.muse_intent  # Failure does not invent a Disconnect intent.
    await setup.disconnect_muse()
    assert len(signals) == 4 and all(sig == signal.SIGTERM for _, sig in signals)


@pytest.mark.asyncio
async def test_exit_while_connecting_and_after_streaming_recovers(rig, fast_time):
    setup, children, signals, reads = rig
    await setup.connect_muse()
    await until(lambda: len(children) == 1)
    children[0].exit(12)
    await until(lambda: len(children) == 2)
    assert setup.muse_retry_count == 1
    fresh(setup)
    await until(lambda: setup.manual_muse_state == 'connected')
    children[1].exit(13)
    # Sticky manual flag cannot override a producer exit, even before supervisor's tick.
    assert setup.muse_connection()['state'] == 'connecting'
    setup.sensors.eeg.disconnected()
    await until(lambda: len(children) == 3)
    assert setup.muse_retry_count == 2
    rows = [json.loads(line) for line in setup.muse_diagnostics.path.read_text().splitlines()]
    assert any(row['event'] == 'producer_exit' and row['code'] == 12 for row in rows)
    assert any(row['event'] == 'sample_progress' and row['lastSampleAt'] for row in rows)
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_silent_outlet_is_distinct_from_missing_outlet(rig, fast_time):
    setup, children, signals, reads = rig
    setup.sensors.eeg.connected('Muse-external')
    await setup.connect_muse()
    await setup.muse_task
    assert setup.muse_reason == 'silent_outlet' and not children and not signals
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_disconnect_during_spawn_registers_and_stops_owned_child(rig):
    setup, children, signals, reads = rig
    entered, release = asyncio.Event(), asyncio.Event()
    original_spawn = setup.spawn
    async def pending(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_spawn(*args, **kwargs)
    setup.spawn = pending
    setup.muse_intent = True
    setup.muse_task = asyncio.create_task(setup.launch('muse', ['fake'], '.'))
    await entered.wait()
    disconnect = asyncio.create_task(setup.disconnect_muse())
    await asyncio.sleep(0)
    release.set()
    await disconnect
    assert len(children) == 1 and signals == [(children[0].pid, signal.SIGTERM)]
    assert not setup.children and not setup.muse_intent


@pytest.mark.asyncio
async def test_native_reader_hang_retains_reference_and_blocks_duplicate(monkeypatch, tmp_path):
    from backend.adaptive.routes import Sensors
    monkeypatch.setenv('GOZ_EEG', 'muse')
    sn = Sensors(directory=tmp_path)
    class Hung:
        def is_alive(self): return True
        def join(self, timeout): pass
    thread = sn.muse_thread = Hung()
    sn.muse_reader_stop = threading.Event()
    await sn.setup.disconnect_muse()
    assert sn.muse_thread is thread and sn.muse_reader_stop.is_set()
    await sn.setup.connect_muse()
    assert sn.setup.manual_muse_state == 'error' and sn.setup.muse_reason == 'reader_stop_timeout'
    assert sn.muse_thread is thread and not sn.setup.children


def test_retained_last_sample_survives_true_disconnect_without_live_evidence():
    feed = EegFeed()
    import time
    feed.connected('Muse-fake')
    stamp = time.time()
    feed.push([[1, 2, 3, 4]], [stamp])
    feed.muse_event('interruption', reason='silent_outlet', detail='test')
    feed.disconnected()
    status = feed.status()
    assert status['lastSampleAt'] is None and not status['live'] and not status['calibrationRetained']
    assert status['acquisitionDiagnostics']['lastSampleAt'] == stamp
    assert status['acquisitionDiagnostics']['lastInterruption']['reason'] == 'silent_outlet'


@pytest.mark.asyncio
async def test_owned_producer_hang_never_force_killed_or_replaced(rig, monkeypatch):
    import backend.sensor_setup as module
    setup, children, signals, reads = rig
    setup.muse_intent = True
    child = await setup.launch('muse', ['fake'], '.')
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    original_wait = asyncio.wait_for
    async def bounded_wait(awaitable, timeout):
        if timeout == 3:
            awaitable.close()
            raise TimeoutError
        return await original_wait(awaitable, timeout)
    monkeypatch.setattr(module, 'asyncio', SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith('__')}))
    monkeypatch.setattr(module.asyncio, 'wait_for', bounded_wait)
    await setup.disconnect_muse()
    assert setup.children['muse'] is child and setup.muse_reason == 'producer_stop_timeout'
    assert signals == [(child.pid, signal.SIGTERM)] and not setup.muse_intent
    with pytest.raises(ValueError, match='intent was cancelled'):
        await setup.launch('muse', ['fake'], '.')
    child.exit(-15)
    await setup.muse_exit_task
    await setup._stop_owned_muse('test_cleanup')


@pytest.mark.asyncio
async def test_clock_unavailability_never_restarts_same_inlet_or_producer(rig, fast_time):
    setup, children, signals, reads = rig
    setup.sensors.eeg.connected('Muse-external')
    setup.sensors.eeg.acquisition_phase = 'waiting_for_clock'
    await setup.connect_muse()
    await setup.muse_task
    assert setup.muse_reason == 'clock_unavailable' and setup.muse_retry_count == 0
    assert reads == ['start'] and not children and not signals
    assert setup.sensors.eeg.device_id == 'Muse-external'
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_failed_ambiguous_external_discovery_cannot_launch_producer(rig, fast_time):
    setup, children, signals, reads = rig
    setup.sensors.eeg.connection_error = 'Multiple Muse streams found; select an address.'
    await setup.connect_muse()
    await setup.muse_task
    assert not children and not signals and setup.muse_reason == 'reader_error'
    assert 'Multiple Muse' in setup.manual_muse_error
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_setup_retry_preserves_live_baseline_and_connect_intent(rig):
    setup, children, signals, reads = rig
    fresh(setup)
    sentinel = setup.sensors.eeg.calibration = {'retained': True}
    # Avoid feature identity checks during this lifecycle-only test.
    setup.muse_intent = True
    cancelled = []
    async def supervise():
        try: await asyncio.Future()
        finally: cancelled.append(True)
    setup.muse_task = asyncio.create_task(supervise())
    async def start(): pass
    setup.start = start
    await asyncio.sleep(0)
    await setup.retry()
    assert setup.muse_intent and not setup.muse_task.done() and not cancelled
    assert setup.sensors.eeg.calibration is sentinel
    await setup.disconnect_muse()
    assert cancelled == [True]


def test_critical_exit_and_last_sample_survive_noisy_rotation_and_reload(tmp_path):
    journal = MuseDiagnostics(tmp_path)
    journal.record('sample_progress', lastSampleAt=1700000000., samplesReceived=256)
    journal.record('producer_exit', code=5, reason='producer_exited', pid=12)
    for _ in range(800): journal.record('producer_output', text='x' * 400)
    restored = MuseDiagnostics(tmp_path).snapshot()['latestEvents']
    assert restored['producer_exit']['code'] == 5
    assert restored['sample_progress']['lastSampleAt'] == 1700000000.
    assert journal.summary_path.stat().st_size < 32 * 1024
    assert journal.summary_path.stat().st_mode & 0o777 == 0o600


def test_producer_observations_do_not_guess_disconnect_cause():
    assert producer_observation('Disconnected.') == 'stream_disconnected'
    assert producer_observation('No Muses found.') == 'discovery_empty'
    assert producer_observation('[debug] BLE connected.') == 'ble_connected'
    assert producer_observation('Streaming EEG...') == 'stream_started'
    assert producer_observation('Failed to connect to Muse.') == 'connection_failed'
    assert producer_observation('Battery low? Bluetooth RF? OS error?') is None


def test_nested_muse_diagnostics_omit_arrays_credentials_urls_and_device_uuid(tmp_path):
    journal = MuseDiagnostics(tmp_path)
    journal.record('recovery_needed', nested={'detail': 'token=hidden https://private.example',
        'device': '1BF29CEA-7874-A428-FD99-8731FDF27DBE', 'samples': [1, 2, 3, 4]}, value=float('nan'))
    text = journal.path.read_text() + journal.summary_path.read_text()
    for value in ('hidden', 'private.example', '1BF29CEA', '[1, 2, 3, 4]', 'NaN'):
        assert value not in text
    row = json.loads(journal.path.read_text())
    assert row['value'] is None and row['nested']['samples'] == '[data omitted]'


def test_pathological_nested_diagnostics_cannot_overflow_file_or_summary_limits(tmp_path):
    journal = MuseDiagnostics(tmp_path)
    huge = {f'field{i}': {f'item{j}': '✓' * 400 for j in range(24)} for i in range(24)}
    for event in journal.SUMMARY_EVENTS:
        journal.record(event, lastSampleAt=1700000000., nested=huge)
    assert journal.path.stat().st_size < journal.MAX_BYTES
    assert journal.summary_path.stat().st_size <= 32 * 1024
    assert all(row['lastSampleAt'] == 1700000000. for row in journal.summary.values())
    assert all(row.get('detailOmitted') for row in journal.summary.values())


@pytest.mark.asyncio
async def test_zero_exit_and_empty_discovery_remain_observations_with_precise_status(rig):
    setup, children, signals, reads = rig
    setup.muse_intent = True
    child = await setup.launch('muse', ['fake'], '.')
    child.stdout.feed_data(b'No Muses found.')  # final partial pipe line is retained too
    child.exit(0)
    await setup.muse_exit_task
    state = setup.muse_connection()
    assert state['state'] == 'connecting' and state['reason'] == 'producer_exited'
    assert state['lastProducerObservation']['category'] == 'discovery_empty'
    assert state['exit']['code'] == 0 and state['bluetoothCause'] == 'unknown'
    assert 'no Muses found' in state['message']
    assert 'unknown' in state['message'] and len(state['message']) <= 500
    setup.manual_muse_state = 'error'
    setup.muse_retry_count = setup.muse_retry_limit
    state = setup.muse_connection()
    assert state['recoveryExhausted'] and state['retryRemaining'] == 0
    assert 'choose Connect' in state['message'] and state['nextRetryAt'] is None
    await setup.disconnect_muse()


@pytest.mark.asyncio
async def test_loss_recovery_retains_reader_evidence_and_records_restored_samples(rig, fast_time):
    setup, children, signals, reads = rig
    await setup.connect_muse()
    await until(lambda: bool(children))
    fresh(setup)
    await until(lambda: setup.manual_muse_state == 'connected')
    setup.sensors.eeg.muse_event('interruption', reason='stream_interrupted', detail='the stream has been lost.')
    setup.sensors.eeg.disconnected()
    children[0].stdout.feed_data(b'Disconnected.\n')
    children[0].exit(0)
    await until(lambda: len(children) == 2)
    loss = setup.muse_connection()['lastLoss']
    assert loss['reason'] == 'producer_exited'
    assert loss['readerInterruption']['reason'] == 'stream_interrupted'
    assert loss['lastSampleAt'] is not None and loss['ownership'] == 'owned'
    fresh(setup)
    await until(lambda: setup.manual_muse_state == 'connected')
    state = setup.muse_connection()
    assert state['message'] == 'Muse samples are live.' and state['lastLoss'] == loss
    assert state['retryCount'] == 1 and state['retryRemaining'] == 2
    assert state['nextRetryAt'] is None
    summary = setup.muse_diagnostics.snapshot()['latestEvents']
    assert summary['recovery_restored']['lastSampleAt'] is not None
    assert summary['recovery_backoff']['nextRetryAt'] is not None
    assert summary['recovery_needed']['readerInterruption']['reason'] == 'stream_interrupted'
    await setup.disconnect_muse()
