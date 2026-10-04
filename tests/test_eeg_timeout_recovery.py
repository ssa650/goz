"""Installed pylsl exception-layout regressions; no hardware or LSL connection."""
import threading
from types import SimpleNamespace

import pylsl.util as lsl_errors
import pytest

from backend.adaptive.sensors import EegFeed, _consume_muse
from test_eeg_acquisition import Info, clock


@pytest.mark.parametrize('error_type',[lsl_errors.TimeoutError,lsl_errors.LostError])
@pytest.mark.parametrize('phase',['open','pull','clock'])
def test_installed_util_exception_recovers_and_accepts_subsequent_samples(clock,error_type,phase):
    feed,stop=EegFeed(),threading.Event()
    feed.source='muse'
    opened,closed,attempts,captured=[],[],[],[]
    original_push=feed.push
    def push(samples,stamps):
        captured.append((samples,stamps))
        original_push(samples,stamps)
    feed.push=push
    class Inlet:
        def __init__(self,info,**kwargs):
            attempts.append(info.source_id())
            self.number=len(attempts)
            assert kwargs['recover'] is False
            if phase=='open' and self.number==1:
                raise error_type('installed liblsl transient error')
            opened.append(self.number)
        def pull_chunk(self,timeout):
            if phase=='pull' and self.number==1:
                raise error_type('installed liblsl transient error')
            if phase=='clock' and self.number==1:
                return [[1,2,3,4,99]],[122.8]
            stop.set()
            return [[1,2,3,4,99]],[1000.0]
        def time_correction(self,timeout):
            assert phase=='clock' and self.number==1 and timeout==.5
            raise error_type('the operation failed due to a timeout.')
        def close_stream(self):closed.append(self.number)
    # Reproduce the actual installed package: util classes, NO root aliases.
    api=SimpleNamespace(util=lsl_errors,resolve_byprop=lambda *a,**k:[Info()],
                        StreamInlet=Inlet,local_clock=lambda:123.0)
    assert not hasattr(api,'TimeoutError') and not hasattr(api,'LostError')
    _consume_muse(feed,stop,api)
    assert attempts==['Muse-device','Muse-device']
    assert opened==closed
    assert feed.samples_received==1 and captured==[([[1,2,3,4]],[1000.0])]
    assert feed.last_sample_at is None, 'disconnect cleanup removes stale sample freshness'
    assert feed.status()['confidence']==0, 'one raw sample is not a clean EEG baseline'


@pytest.mark.parametrize('error_type',[lsl_errors.TimeoutError,lsl_errors.LostError])
def test_recoverable_discovery_error_retries_without_opening_duplicate_inlets(clock,error_type):
    feed,stop=EegFeed(),threading.Event()
    attempts,opened=[],[]
    def resolve(*args,**kwargs):
        attempts.append(True)
        if len(attempts)==1:raise error_type('temporary discovery interruption')
        return [Info()]
    class Inlet:
        def __init__(self,info,**kwargs):opened.append(info.source_id())
        def pull_chunk(self,timeout):
            stop.set();return [[1,2,3,4]],[1000.0]
        def close_stream(self):pass
    api=SimpleNamespace(util=lsl_errors,resolve_byprop=resolve,StreamInlet=Inlet,local_clock=lambda:123.0)
    _consume_muse(feed,stop,api)
    assert len(attempts)==2 and opened==['Muse-device'] and feed.samples_received==1


@pytest.mark.parametrize('phase',['discovery','inlet'])
def test_unrelated_runtime_error_remains_visible_instead_of_retry_loop(clock,phase):
    feed,stop=EegFeed(),threading.Event()
    closed=[]
    def resolve(*args,**kwargs):
        if phase=='discovery':raise RuntimeError('unexpected resolver bug')
        return [Info()]
    class Inlet:
        def __init__(self,*args,**kwargs):pass
        def pull_chunk(self,timeout):raise RuntimeError('unexpected inlet bug')
        def close_stream(self):closed.append(True)
    api=SimpleNamespace(util=lsl_errors,resolve_byprop=resolve,StreamInlet=Inlet,local_clock=lambda:123.0)
    with pytest.raises(RuntimeError,match='unexpected'):
        _consume_muse(feed,stop,api)
    assert closed==([] if phase=='discovery' else[True])
    assert feed.samples_received==0
