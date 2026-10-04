"""No downloaded models/network/sensors; actual labelled reference atlas + contract."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import numpy as np
import pytest

from backend.adaptive.yoloe_detector import DetectorConfig, YOLOECharacterDetector, build_reference_prompt
from backend.adaptive.yoloe_detector.detector import UltralyticsRuntime, parse_predictions
from backend.adaptive.yoloe_detector.evaluation import evaluate

CAST = ["SpongeBob", "Patrick"]
MAP = {0: "SpongeBob", 1: "Patrick"}
CONFIG = DetectorConfig()


def candidate(cls=0, score=.7, box=(.1,.1,.4,.8)):
    return dict(class_id=cls, score=score, box=list(box))


def test_actual_five_labelled_references_and_generic_sequential_map():
    from backend.adaptive.tracks import REFERENCE_CROPS as shared_crops
    from backend.adaptive.yoloe_detector.config import REFERENCE_CROPS
    assert REFERENCE_CROPS == shared_crops
    atlas, prompt, mapping, sources = build_reference_prompt(CAST + ["Squidward"], strategy='atlas')
    assert atlas.dtype == np.uint8 and atlas.shape == (416,624,3)
    assert mapping == MAP
    assert prompt['cls'].tolist() == [0,0,0,1,1]
    assert len(sources) == 5
    assert all((prompt['bboxes'][:,2:] > prompt['bboxes'][:,:2]).ravel())
    # Class order is attached to the references; generic model captions unused.
    assert build_reference_prompt(list(reversed(CAST)))[2] == {0:'Patrick',1:'SpongeBob'}


def test_default_prompt_preserves_real_scene_and_reference_boxes():
    reference,prompt,mapping,sources=build_reference_prompt(CAST)
    assert max(reference.shape[:2])<=640 and prompt['cls'].tolist()==[0,1]
    assert mapping==MAP and [s['file'] for s in sources]==['00-30.jpg','00-30.jpg']
    assert prompt['bboxes'][0,0]/reference.shape[1] == pytest.approx(.555)


def test_absent_or_unreferenced_names_are_not_forced():
    boxes, regions = parse_predictions([], MAP, CAST, CONFIG)
    assert boxes == {} and regions == []
    assert parse_predictions([candidate(1)], MAP, ['Squidward'], CONFIG)[0] == {}
    with pytest.raises(ValueError, match='No labelled'):
        build_reference_prompt(['Squidward'])


@pytest.mark.parametrize('class_id', [5, .5, float('nan'), float('inf')])
def test_unknown_generic_class_id_abstains(class_id):
    boxes, regions = parse_predictions([candidate(class_id)], MAP, CAST, CONFIG)
    assert boxes == {} and regions[0]['identity_status'] == 'unknown_class_id'


def test_low_score_is_visible_unknown_candidate():
    boxes, regions = parse_predictions([candidate(score=.3)], MAP, CAST, CONFIG)
    assert boxes == {} and regions[0]['candidate_identity'] == 'SpongeBob'
    assert regions[0]['identity'] is None
    assert regions[0]['verification']['score_kind'] == 'uncalibrated_model_score'


@pytest.mark.parametrize('bad', [candidate(score=float('nan')), candidate(score=1.1),
    candidate(box=[-.1,0,.5,1]), candidate(box=[.1,0,.1,1]), candidate(box=[0,0,float('inf'),1])])
def test_malformed_provider_geometry_cannot_escape_into_overlay(bad):
    assert parse_predictions([bad], MAP, CAST, CONFIG) == ({},[])


def test_overlapping_similar_identity_candidates_abstain_both():
    boxes, regions = parse_predictions([candidate(0,.7),candidate(1,.65)], MAP, CAST, CONFIG)
    assert boxes == {} and all(r['identity_status']=='ambiguous_identity' for r in regions)
    # This compares returned alternatives; no claim of a full posterior margin.
    assert parse_predictions([candidate(0,.9), candidate(1,.3)], MAP, CAST, CONFIG)[0].keys() == {'SpongeBob'}


def test_disjoint_duplicate_identity_locations_abstain():
    boxes, regions = parse_predictions([candidate(),candidate(box=[.6,.1,.9,.8])], MAP, CAST, CONFIG)
    assert boxes == {} and all(r['identity_status']=='ambiguous_reference_locations' for r in regions)


def test_one_box_per_identity_and_weaker_duplicate_retains_diagnostic():
    boxes, regions = parse_predictions([candidate(score=.9),candidate(score=.46, box=[.6,.1,.9,.8])], MAP, CAST, CONFIG)
    assert set(boxes) == {'SpongeBob'} and regions[1]['identity_status']=='suppressed_duplicate'


def test_unknown_blank_cut_and_new_pose_do_not_inherit_old_identity():
    class Runtime:
        def __init__(self): self.calls = 0
        def predict(self, pixels):
            self.calls += 1
            return [candidate()] if self.calls == 1 else []
    runtime = Runtime()
    d = YOLOECharacterDetector(CAST + ['Squidward'], runtime=runtime)
    rng = np.random.default_rng(8)
    frame = rng.integers(0,256,(90,160,3),dtype=np.uint8)
    first = d.step(frame, 0)
    assert set(first['boxes']) == {'SpongeBob'} and first['unknown'] == ['Patrick','Squidward']
    # Empty new-frame detections never get a tracked identity.
    assert d.step(frame, .5)['boxes'] == {}
    blank = d.step(np.zeros_like(frame), 1)
    assert blank['cut'] and blank['shot_id'] == 1 and blank['status']=='blank_frame'
    assert runtime.calls == 2
    after = d.step(frame, 1.5)
    assert after['cut'] and after['boxes']=={} and after['valid_until']==2
    assert after['experimental'] and after['coordinate_space']=='video-normalized'
    with pytest.raises(ValueError, match='monotonically'):
        d.step(frame, 1.5)


def test_package_import_and_license_gate_do_not_import_vendor_runtime():
    assert 'ultralytics' not in sys.modules
    with pytest.raises(RuntimeError, match='license decision'):
        CONFIG.validated_weights()
    assert 'ultralytics' not in sys.modules


def test_checkpoint_validated_before_any_pickle_capable_loader(tmp_path, monkeypatch):
    path = tmp_path/'yoloe-11s-seg.pt'
    path.write_bytes(b'not a checkpoint; never loaded')
    config = replace(CONFIG, weights=str(path), license_reviewed=True, weights_sha256='0'*64)
    with pytest.raises(ValueError, match='pinned official'):
        config.validated_weights()
    from backend.adaptive.yoloe_detector import config as config_module
    monkeypatch.setattr(config_module,'OFFICIAL_WEIGHT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    valid = replace(config, weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert valid.validated_weights() == path
    assert 'ultralytics' not in sys.modules


@pytest.mark.parametrize('fields', [dict(fps=5),dict(queue_size=9),dict(device='cuda'),dict(max_seconds=121),
    dict(identity_threshold=.1,candidate_threshold=.3),dict(fps=float('nan'))])
def test_configuration_enforces_negotiated_local_budget(fields):
    with pytest.raises(ValueError): DetectorConfig(**fields)


def test_mps_operator_failure_has_exactly_one_cpu_retry():
    r = object.__new__(UltralyticsRuntime)
    r.config, r.device, r.initialized = CONFIG, 'mps', True
    r.model = SimpleNamespace(predictor='cached', to=lambda device:None)
    devices=[]
    def predict(pixels):
        devices.append(r.device)
        if r.device == 'mps': raise NotImplementedError('MPS operator missing')
        return []
    r._predict=predict
    assert r.predict(None) == [] and devices == ['mps','cpu']
    r.predict(None)
    assert devices == ['mps','cpu','cpu']
    assert r.fallback_reason == 'NotImplementedError'


def test_class_map_failure_is_not_masked_as_cpu_fallback():
    r=object.__new__(UltralyticsRuntime)
    r.config,r.device=CONFIG,'mps'
    r._predict=lambda pixels: (_ for _ in ()).throw(RuntimeError('class map changed'))
    with pytest.raises(RuntimeError, match='class map'): r.predict(None)


def test_metric_wrong_identity_unknown_cut_and_latency():
    annotations=[dict(clip='a',t=t,shot_id=shot,detections=[dict(identity='SpongeBob',instance_id='s',box=[.1,.1,.4,.8])])
                 for t,shot in [(0,0),(.5,0),(1,0),(1.5,1)]]
    records=[dict(clip='a',t=t,boxes=boxes,inference_seconds=latency) for t,boxes,latency in
             [(0,{'SpongeBob':[.1,.1,.4,.8]},.1),(.5,{'Patrick':[.1,.1,.4,.8]},.2),
              (1,{},.3),(1.5,{'Patrick':[.1,.1,.4,.8]},.4)]]
    metrics=evaluate(annotations,records)
    assert metrics['true_positive']==1 and metrics['false_positive']==2 and metrics['false_negative']==3
    assert metrics['precision']==1/3 and metrics['recall']==.25
    assert metrics['unknown_rate']==.25 and metrics['identity_errors']==2 and metrics['identity_switches']==1
    assert metrics['first_valid_media_seconds']=={'a':0} and metrics['inference_p50_seconds']==.25
    with pytest.raises(ValueError, match='Duplicate prediction'): evaluate(annotations,records+records[:1])


@pytest.mark.asyncio
async def test_isolated_adapter_protocol_tags_progress_and_cooperative_close(monkeypatch):
    import asyncio,io,threading
    from backend.adaptive.yoloe_detector import worker
    stopped=threading.Event()
    payloads=[]
    class Input:
        def write(self,value): payloads.append(value)
        def flush(self): pass
        def close(self): stopped.set()
    class Process:
        stdin=Input()
        stdout=io.StringIO('\n'.join([__import__('json').dumps(dict(t=0,boxes={},clip_id='c',session_id='s')),
                                     __import__('json').dumps(dict(worker_done=True))])+'\n')
        def wait(self): stopped.wait(1)
    processes=[]
    def popen(args,**kwargs):
        processes.append((args,kwargs));return Process()
    monkeypatch.setattr(worker.subprocess,'Popen',popen)
    monkeypatch.setattr(DetectorConfig,'validated_weights',lambda self:Path('/tmp/yoloe-11s-seg.pt'))
    updates=[]
    config=replace(CONFIG,runtime_python=sys.executable,max_wall_seconds=1,queue_size=1)
    result=await worker.detect_yoloe('local.mp4',CAST,clip_id='c',session_id='s',config=config,on_progress=updates.append)
    assert result[0]['clip_id']=='c' and updates[-1]==result and stopped.is_set()
    assert processes[0][0]==[sys.executable,'-m','backend.adaptive.yoloe_detector.worker']
    assert processes[0][1]['env']['OMP_NUM_THREADS']=='1'
    assert __import__('json').loads(payloads[0])['config']['queue_size']==1
    for _ in range(100):
        if not worker._slot.locked():break
        await asyncio.sleep(.01)
    assert not worker._slot.locked()


@pytest.mark.asyncio
async def test_cancel_during_isolated_startup_keeps_slot_until_exit_and_drops_late_output(monkeypatch):
    import asyncio,threading
    from backend.adaptive.yoloe_detector import worker
    stop=threading.Event();exited=threading.Event();released=threading.Event()
    class Input:
        def write(self,value): pass
        def flush(self): pass
        def close(self): stop.set()
    class Output:
        def __iter__(self):
            stop.wait(1)
            yield '{"t":0,"boxes":{"SpongeBob":[0,0,1,1]}}\n'
        def close(self): pass
    class Process:
        stdin=Input();stdout=Output()
        def wait(self): released.wait(1);exited.set()
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(DetectorConfig,'validated_weights',lambda self:Path('/tmp/yoloe-11s-seg.pt'))
    updates=[]
    config=replace(CONFIG,runtime_python=sys.executable,max_wall_seconds=1)
    task=asyncio.create_task(worker.detect_yoloe('local.mp4',CAST,clip_id='c',session_id='s',config=config,on_progress=updates.append))
    await asyncio.sleep(.03);task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert stop.is_set() and worker._slot.locked()
    with pytest.raises(RuntimeError,match='worker busy'):
        await worker.detect_yoloe('local.mp4',CAST,clip_id='c',session_id='s',config=config)
    released.set()
    for _ in range(100):
        if not worker._slot.locked():break
        await asyncio.sleep(.01)
    assert exited.is_set() and updates==[] and not worker._slot.locked()
