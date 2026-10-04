import pytest
from backend.adaptive import fusion

RECT = dict(x=100, y=200, w=800, h=400)
BOXES = {'SpongeBob': [.1,.1,.4,.9], 'Patrick': [.6,.1,.9,.9]}


def series(points=(.1,.2,.3,.4), *, names=BOXES, clip='a'):
    ticks = [dict(wall=100+t, video_t=t, playing=True, rect=RECT, sessionId='s', clipId=clip, epoch=0) for t in points]
    track = [dict(t=0, valid_until=10, boxes={n: list(b) for n,b in names.items()}, clip_id=clip)]
    samples = [dict(t=100+t, x=700, y=400, valid=True, confidence=.8) for t in points]
    return ticks, track, samples


def test_elapsed_time_integration_has_no_sample_count_or_tail_inflation():
    ticks, track, samples = series((.1,.13,.2,.25,.4))
    analysis = fusion.analyze(fusion.label(samples, ticks, track), [], track, list(BOXES))
    assert analysis['valid_gaze_s'] == .3
    assert analysis['comparison_s'] == .3
    assert analysis['characters']['Patrick']['dwell_s'] == .3
    assert analysis['characters']['Patrick']['comparison_dwell_s'] == .3
    assert analysis['gaze_confidence'] == .8
    one = fusion.analyze(fusion.label(samples[:1], ticks, track), [], track, list(BOXES))
    assert one['valid_gaze_s'] == 0


def test_only_character_visibility_is_not_comparative_preference():
    ticks, track, samples = series(names={'Patrick': BOXES['Patrick']})
    analysis = fusion.analyze(fusion.label(samples, ticks, track), [], track, list(BOXES))
    assert analysis['characters']['Patrick']['dwell_s'] == .3
    assert analysis['comparison_s'] == 0
    assert analysis['characters']['Patrick']['comparison_dwell_s'] == 0


@pytest.mark.parametrize('kind', ['invalid', 'stale', 'pause', 'seek', 'clip', 'gap'])
def test_bad_intervals_never_create_dwell(kind):
    ticks, track, samples = series((.1,.2))
    if kind == 'invalid': samples[1]['valid'] = False
    if kind == 'stale': samples[1]['stale'] = True
    if kind == 'pause': ticks[1]['playing'] = False
    if kind == 'seek': ticks[1].update(video_t=5, epoch=1)
    if kind == 'clip': ticks[1]['clipId'] = 'b'
    if kind == 'gap': samples[1]['t'] += 1
    analysis = fusion.analyze(fusion.label(samples, ticks, track), [], track, list(BOXES))
    assert analysis['valid_gaze_s'] == 0
    assert analysis['characters']['Patrick']['dwell_s'] == 0


def test_missing_tracking_is_not_look_away_and_missing_eeg_not_zero_response():
    ticks, track, samples = series()
    for sample in samples: sample['valid'] = False
    analysis = fusion.analyze(fusion.label(samples, ticks, track), [], track, list(BOXES))
    assert analysis['look_away_frac'] is None
    assert analysis['characters']['Patrick']['eeg_response'] is None
    assert not analysis['eeg_available']


def test_shared_screen_css_content_rect_agrees_with_normalized_coordinates_and_clip_bounds():
    ticks, track, samples = series((.1,))
    normalized = dict(samples[0], coordinate_space='video-normalized', nx=.75, ny=.5)
    a = fusion.label(samples, ticks, track)[0]
    b = fusion.label([normalized], ticks, track)[0]
    assert (a['nx'], a['ny'], a['target']) == (b['nx'], b['ny'], b['target'])
    ticks[0]['visible_rect'] = dict(x=100, y=200, w=400, h=400)
    c = fusion.label(samples, ticks, track)[0]
    assert not c['on_video'] and c['target'] is None and c['state'] == 'outside-video'


def test_overlap_unknown_expired_and_wrong_clip_are_not_named_attention():
    ticks, track, samples = series((.1,))
    track[0]['boxes']['SpongeBob'] = [.6,.1,.9,.9]
    assert fusion.label(samples, ticks, track)[0]['state'] == 'ambiguous'
    track[0]['clip_id'] = 'old'
    e = fusion.label(samples, ticks, track)[0]
    assert e['target'] is None and e['state'] == 'unavailable'


def test_no_future_playback_tick_is_used_as_capture_evidence():
    ticks, track, samples = series((.1,))
    samples[0]['t'] = 100.05
    assert fusion.label(samples, ticks, track) == []


def test_alternating_character_plus_object_is_not_two_character_comparison():
    ticks,track,samples=series((.1,.2,.3,.4))
    track=[dict(t=0,valid_until=.25,boxes={'Patrick':BOXES['Patrick'],'box':[.4,.4,.6,.6]},clip_id='a'),
           dict(t=.25,valid_until=1,boxes={'SpongeBob':BOXES['Patrick'],'box':[.4,.4,.6,.6]},clip_id='a')]
    analysis=fusion.analyze(fusion.label(samples,ticks,track),[],track,['SpongeBob','Patrick'])
    assert analysis['comparison_s']==0
    assert all(c['comparison_dwell_s']==0 for c in analysis['characters'].values())
