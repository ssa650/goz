"""Synthetic transport/model timing; never enumerate or open live hardware."""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from backend import gaze_worker as worker
from backend.gaze_pipeline import BlinkGate, StageDiagnostics, DurableDiagnostics
from backend.adaptive import sensors, gaze_audit
from test_camera_identity import PHONE, fake_capture, frame_packet
from backend import native_camera as native


def observation(score=.05, openness=.25, ok=True):
    return SimpleNamespace(ok=ok, blink=score, features=[0.,0.,0.,openness,0.,0.,0.,openness]+[0.]*6,
                           yaw=0., pitch=0., brightness=100., interocular_px=.2)


def test_actual_blink_and_continuous_reopen_hold(tmp_path):
    g=BlinkGate(tmp_path/'absent')
    assert not g.update(observation(), 0.)
    assert g.update(observation(.8, .08), .033)
    assert g.update(observation(), .066)
    # Ambiguous evidence resets the reopen hold, instead of retaining old time.
    assert g.update(observation(.22), .2)
    assert g.update(observation(), .3)
    assert g.update(observation(), .5)
    assert not g.update(observation(), .56)


@pytest.mark.parametrize('obs,reason', [(observation(.8),'blink-score'),
    (observation(.05,.10),'eyelid-collapse'), (observation(ok=False),'no-face'),
    (observation(float('nan')),'invalid-eye-features')])
def test_prolonged_gate_never_turns_invalid_landmarks_into_valid_gaze(tmp_path, obs, reason):
    g=BlinkGate(tmp_path/'absent')
    for i in range(200): assert g.update(obs, i/30)
    d=g.diagnostics(200/30)
    assert d['reason']==reason and d['gatedSeconds']>6
    assert d['thresholds']==dict(on=.28, off=.18, openMin=.16, reopenHoldS=.25)
    json.dumps(d, allow_nan=False)


def test_open_samples_after_delivery_gap_need_fresh_hold(tmp_path):
    g=BlinkGate(tmp_path/'absent')
    assert not g.update(observation(), 1)
    assert g.update(observation(), 13.286)
    assert g.update(observation(), 13.386)
    assert not g.update(observation(), 13.586)


def test_personal_blink_thresholds_preserved_and_bad_profile_fails(tmp_path):
    p=tmp_path/'profile.json'
    p.write_text(json.dumps(dict(blink_on=.4,blink_off=.2,open_min=.12)))
    g=BlinkGate(p)
    assert not g.update(observation(.3,.15), 0)
    assert g.profile=='personalized'
    p.write_text(json.dumps(dict(blink_on=.1,blink_off=.2,open_min=.12)))
    with pytest.raises(ValueError, match='Invalid blink profile'): BlinkGate(p)


def test_tracker_video_clock_follows_irregular_capture_not_inference_clock():
    class Tracker:
        _ts_ms=0
        def process(self, frame):
            self._ts_ms+=33
            return self._ts_ms
    t=Tracker()
    assert [worker.process_timed(t,None,stamp) for stamp in [40,40.233,40.921,53.207]]==[1,234,922,13208]
    # Millisecond rounding never causes a duplicate VIDEO timestamp.
    assert worker.process_timed(t,None,53.2071)==13209


def test_latest_native_slot_replaces_pixels_before_conversion(monkeypatch):
    packets=b''.join(frame_packet(bytes([i,22,33,255,44,55,66,255]), i+1) for i in range(64))
    process,_=fake_capture(monkeypatch, packets=packets)
    calls=[]
    monkeypatch.setattr(sys.modules['cv2'],'cvtColor',lambda p,c: calls.append(p[0,0,0]) or p[...,:3].copy())
    cap=native.NativeCapture(PHONE)
    try:
        # Synchronize with receiver without running a camera or a subprocess.
        deadline=time.monotonic()+1
        while cap.diagnostics()['received']<64 and time.monotonic()<deadline: time.sleep(.001)
        assert calls==[]
        ok, frame, timing=cap.read_timed()
        assert ok and frame[0,0,0]==63 and calls==[63]
        assert timing['sequence']==64 and timing['capturedAt']<timing['readAt']
        d=cap.diagnostics()
        assert d['replaced']==63 and d['queueCapacity']==1 and d['consumed']==1
    finally: cap.release()


@pytest.mark.parametrize('change,message', [(dict(sequence=0),'sequence'),
    (dict(capturedAt=float('nan')),'timestamps'),(dict(width=4097),'packet')])
def test_bad_native_metadata_fails_without_fallback(monkeypatch,change,message):
    process,calls=fake_capture(monkeypatch, packets=frame_packet(bytes(8), **change))
    cap=native.NativeCapture(PHONE)
    try:
        deadline=time.monotonic()+1
        while not cap.error and time.monotonic()<deadline: time.sleep(.001)
        with pytest.raises(ValueError,match=message):cap.read_timed()
        assert len(calls)==1
    finally:cap.release()


def test_stage_storage_and_durable_snapshot_are_bounded(tmp_path):
    stats=StageDiagnostics()
    cpu=time.process_time()
    for i in range(10000):
        stats.count('frames');stats.observe('inference',.01)
    assert time.process_time()-cpu<1 # synthetic counter overhead only
    snapshot=stats.snapshot()
    assert snapshot['counts']['frames']==10000 and snapshot['timings']['inference']['maxMs']==10
    assert len(snapshot['timings'])==10
    path=tmp_path/'diagnostics.json'
    report=DurableDiagnostics(path,stats.snapshot,worker.atomic_json)
    report.close()
    assert json.loads(path.read_text())['counts']['frames']==10000
    assert [p.name for p in tmp_path.iterdir()]==['diagnostics.json']
    assert path.stat().st_mode & 0o777==0o600


@pytest.mark.parametrize('cold',[True,False])
def test_stream_cold_warm_and_steady_stale_paths(monkeypatch,tmp_path,cold):
    class Clock:
        now=100.
        def monotonic(self):return self.now
        def time(self):return self.now+1000
    clock=Clock();monkeypatch.setattr(worker,'time',clock)
    order=[];packets=[];durations=[]
    class Tracker:
        _ts_ms=0
        def __init__(self,path):
            order.append('model')
            clock.now+=2 if cold else .01
        def process(self,frame):
            self._ts_ms+=33
            clock.now+=1.418 if len(packets)==0 else .01
            durations.append(self._ts_ms)
            return observation()
        def close(self):order.append('tracker-close')
    class Capture:
        identity={'deviceId':'verified'}
        i=0
        def read_timed(self):
            self.i+=1;clock.now+=1/30
            age=.8 if self.i==1 else .01
            return True,np.zeros((1,2,3),dtype=np.uint8),dict(capturedMonotonic=clock.now-age,
                capturedAt=clock.time()-age, sequence=self.i,deliveredAt=clock.time(),receivedAt=clock.time(),readAt=clock.time())
        def diagnostics(self):return {'queueCapacity':1}
        def release(self):order.append('capture-close')
    cap=Capture()
    def opened(args):order.append('camera');return cap
    class Sender:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def sendto(self,data,address):packets.append(json.loads(data))
    package=ModuleType('gazekit');package.__path__=[]
    tracker=ModuleType('gazekit.tracker');tracker.FaceTracker=Tracker
    filters=ModuleType('gazekit.filters')
    filters.GazeSmoother=lambda:SimpleNamespace(apply=lambda x,y,t:(x,y))
    for k,m in [('gazekit',package),('gazekit.tracker',tracker),('gazekit.filters',filters)]:monkeypatch.setitem(sys.modules,k,m)
    monkeypatch.setattr(worker,'load_predictor',lambda *a:(lambda obs:(70,50),None))
    monkeypatch.setattr(worker,'open_selected_camera',opened)
    monkeypatch.setattr(worker.socket,'socket',lambda *a:Sender())
    args=SimpleNamespace(seconds=2,setup_id='s',profile='default',port=5590)
    worker.stream_gaze(args,tmp_path/'model.pkl','asset',(100,80))
    assert order[:2]==['model','camera'] and order[-2:]==['tracker-close','capture-close']
    assert packets[0]['stale'] and not packets[0]['valid']
    assert all(p['captureTiming']=='native-host-clock-pts' for p in packets)
    assert any(p['valid'] for p in packets[1:])
    d=json.loads((tmp_path/'gaze-stream-diagnostics.json').read_text())
    assert d['finished'] and d['worker']['counts']['staleBefore']==1
    assert d['worker']['counts']['staleAfter']==1
    assert d['worker']['counts']['valid']>=5
    assert all(b>a for a,b in zip(durations,durations[1:]))


def test_receipt_gap_and_sender_delay_are_separate(monkeypatch):
    clock=[10.]
    monkeypatch.setattr(sensors,'time',SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0]))
    feed=sensors.GazeFeed();feed.expected_setup_id='s'
    def packet(seq):return dict(t=clock[0]-.04,x=1,y=2,sentAt=clock[0]-.02,frameSequence=seq,setupId='s',valid=True)
    assert feed.packet(packet(1));clock[0]+=12.286
    assert feed.packet(packet(5))
    d=feed.status()['inputDiagnostics']
    assert d['interarrivalMsMax']==pytest.approx(12286)
    assert d['sendToReceiptMsMax']==pytest.approx(20)
    assert d['eventLoopLagMsMax']==0 and d['frameSequenceGaps']==3 and d['udpSequenceGaps']==0
    assert not feed.packet(dict(packet(6),t=clock[0]-6))
    assert feed.diagnostics['timestampRejected']==1
    assert feed.samples[-1]['receivedAt']==clock[0]
    audit=gaze_audit.summarize(list(feed.samples),[],[])
    assert audit['receipt_diagnostics']['frameSequenceGaps']==3


@pytest.mark.asyncio
async def test_event_loop_monitor_with_fake_endpoint_records_lag_not_udp(monkeypatch):
    feed=sensors.GazeFeed()
    loop=asyncio.get_running_loop()
    mono=[0.]
    monkeypatch.setattr(sensors,'time',SimpleNamespace(monotonic=lambda:mono[0]))
    protocol=[]
    async def endpoint(factory,**kwargs):
        protocol.append(factory());return SimpleNamespace(close=lambda:None),protocol[0]
    monkeypatch.setattr(loop,'create_datagram_endpoint',endpoint)
    await feed.listen(0)
    await asyncio.sleep(0)
    mono[0]=1.5
    await asyncio.sleep(.27)
    try:
        assert feed.diagnostics['eventLoopLagMsMax']==pytest.approx(1250)
        assert feed.diagnostics['interarrivalMsMax']==0
        protocol[0].datagram_received(b'not-json',('127.0.0.1',1))
        assert feed.diagnostics['malformed']==1
    finally:
        feed.close()
        await asyncio.gather(feed.loop_monitor,return_exceptions=True)


def test_native_heartbeat_survives_without_any_pixel_packet(monkeypatch):
    process,_=fake_capture(monkeypatch)
    import io
    now=time.time()
    values=dict(nativeDelivered=50,nativeDropped=2,nativeReplaced=47,nativePending=1,
                nativeCallbackAgeS=.01,nativeWriteAgeS=12.2,previousWriteMs=10,nativeHeartbeatAt=now)
    process.stderr=io.BytesIO(('GOZ_NATIVE_DIAGNOSTICS '+json.dumps(values)+'\n').encode())
    cap=native.NativeCapture(PHONE)
    try:
        d=cap.diagnostics()
        assert d['received']==0 and d['nativeHeartbeat']['nativeDelivered']==50
        assert d['nativeHeartbeat']['nativeWriteAgeS']==12.2 and d['nativeHeartbeatAgeS']<1
    finally:cap.release()


def test_rejected_old_packets_do_not_count_as_udp_loss(monkeypatch):
    clock=[20.]
    monkeypatch.setattr(sensors,'time',SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0]))
    feed=sensors.GazeFeed();feed.expected_setup_id='s'
    def packet(seq,t):return dict(t=t,x=1,y=2,sentAt=t+.01,setupId='s',sentSequence=seq)
    assert feed.packet(packet(1,19.9))
    clock[0]=32
    assert not feed.packet(packet(2,20))
    assert feed.packet(packet(3,31.9))
    assert feed.diagnostics['udpSequenceGaps']==0
    assert feed.diagnostics['captureToReceiptMsMax']==12000
    assert feed.diagnostics['timestampRejected']==1
    clock[0]=32.1
    assert feed.packet(packet(5,32.05))
    assert feed.diagnostics['udpSequenceGaps']==1


def test_durable_heartbeat_updates_during_processing_silence(tmp_path):
    import threading
    stats=StageDiagnostics();stats.enter('inference')
    written=threading.Event();values=[]
    def save(path,value):
        values.append(value)
        if len(values)>=2:written.set()
    reporter=DurableDiagnostics(tmp_path/'snapshot.json',stats.snapshot,save)
    try:
        assert written.wait(3)
        assert values[-1]['stage']=='inference' and values[-1]['stageAgeS']>=2
        assert values[-1]['counts']['frames']==0
    finally:reporter.close()
