"""Color-demo contracts and independent synthetic confounds; no live devices."""
import json
import cv2
import numpy as np
import pytest

from backend.adaptive.color_detector import ColorCharacterDetector, ColorDetectorConfig


def background():
    image = np.full((240,400,3),(25,100,185),np.uint8)
    image[210:] = (190,180,140)
    return image


def sponge(image, x=110):
    cv2.rectangle(image,(x,100),(x+65,165),(240,210,40),-1)
    cv2.ellipse(image,(x+22,127),(9,13),0,0,360,(240,240,240),-1)
    cv2.ellipse(image,(x+44,127),(9,13),0,0,360,(240,240,240),-1)
    cv2.rectangle(image,(x+3,166),(x+62,184),(175,130,80),-1)
    return image


def patrick(image):
    cv2.fillPoly(image,[np.array([[285,38],[258,155],[255,190],[335,190],[314,155]])],(240,145,130))
    cv2.ellipse(image,(284,116),(9,13),0,0,360,(240,240,240),-1)
    cv2.ellipse(image,(306,116),(9,13),0,0,360,(240,240,240),-1)
    cv2.rectangle(image,(257,191),(333,207),(140,205,45),-1)
    return image


def test_detects_two_supported_characters_and_absent_patrick():
    d=ColorCharacterDetector(clip_id='c',session_id='s',generation_id='g')
    one=d.step(sponge(background()),0)
    assert set(one['boxes'])=={'SpongeBob'} and one['unknown']==['Patrick']
    both=d.step(patrick(sponge(background())),.25)
    assert set(both['boxes'])=={'SpongeBob','Patrick'}
    assert both['source']=='opencv_color_shape_demo'
    assert (both['clip_id'],both['session_id'],both['generation_id'])==('c','s','g')
    assert both['coordinate_space']=='video-normalized' and both['valid_until']==.5
    for b in both['boxes'].values():
        assert len(b)==4 and all(0<=v<=1 for v in b) and b[0]<b[2] and b[1]<b[3]
    assert both['provider_confidence_available'] is False
    json.dumps(both,allow_nan=False)


def test_thin_clouds_and_flower_outlines_do_not_become_filled_bodies():
    image=background()
    for color,center in [((240,210,40),(110,100)),((240,145,130),(290,100))]:
        cv2.ellipse(image,center,(65,32),0,0,360,color,2)
        cv2.ellipse(image,(center[0]-30,center[1]-18),(25,15),0,0,360,color,2)
        cv2.ellipse(image,(center[0]+30,center[1]-18),(25,15),0,0,360,color,2)
        cv2.circle(image,(center[0],center[1]),9,(245,245,245),-1)
        cv2.rectangle(image,(center[0]-40,145),(center[0]+40,165),(140,205,45),-1)
    result=ColorCharacterDetector().step(image,0)
    assert result['boxes']=={} and set(result['unknown'])=={'SpongeBob','Patrick'}


def test_thick_colored_blobs_without_eyes_and_clothes_abstain():
    image=background()
    cv2.rectangle(image,(70,80),(170,160),(240,210,40),-1)
    cv2.ellipse(image,(290,100),(40,70),0,0,360,(240,145,130),-1)
    result=ColorCharacterDetector().step(image,0)
    assert not result['boxes']
    assert any(r['identity_status']=='unknown_insufficient_support' for r in result['regions'])


def test_duplicate_full_bodies_are_ambiguous():
    image=sponge(sponge(background(),x=40),x=230)
    result=ColorCharacterDetector(['SpongeBob']).step(image,0)
    assert not result['boxes']
    assert sum(r['identity_status']=='ambiguous_color_locations' for r in result['regions'])==2


def test_blank_cut_gap_and_reset_do_not_inherit_boxes_or_provenance():
    d=ColorCharacterDetector(clip_id='old',session_id='old-s')
    assert d.step(patrick(sponge(background())),0)['boxes']
    r=d.step(np.zeros((240,400,3),np.uint8),.25)
    assert r['boxes']=={} and r['status']=='blank_frame' and r['cut']
    assert not d.previous_boxes
    r=d.step(background(),1.5)
    assert not r['boxes'] and not d.previous_boxes
    d.reset(clip_id='new',session_id='new-s')
    r=d.step(background(),0)
    assert r['shot_id']==0 and not r['cut'] and r['clip_id']=='new' and r['generation_id'] is None


@pytest.mark.parametrize('t',[-1,float('nan'),float('inf'),True,'1'])
def test_rejects_bad_times(t):
    with pytest.raises(ValueError): ColorCharacterDetector().step(background(),t)


def test_requires_monotonic_times_and_uint8_rgb():
    d=ColorCharacterDetector();d.step(background(),0)
    with pytest.raises(ValueError):d.step(background(),0)
    for image in [np.zeros((5,5),np.uint8),np.zeros((5,5,3),float),np.zeros((1,5,3),np.uint8)]:
        with pytest.raises(ValueError):ColorCharacterDetector().step(image,0)


@pytest.mark.parametrize('kwargs',[{'fps':5},{'fps':float('nan')},{'fps':True},{'image_size':1920},{'max_candidates_per_identity':9},{'max_candidates_per_identity':True}])
def test_bounded_configuration(kwargs):
    with pytest.raises(ValueError):ColorDetectorConfig(**kwargs)


def test_supported_names_only_and_no_opencv_global_thread_change(monkeypatch):
    with pytest.raises(ValueError):ColorCharacterDetector(['Squidward'])
    def fail(*args):raise AssertionError('process-global OpenCV thread mutation')
    monkeypatch.setattr(cv2,'setNumThreads',fail)
    result=ColorCharacterDetector(['SpongeBob']).step(sponge(background()),0)
    assert set(result['boxes'])=={'SpongeBob'}


def test_long_edge_and_candidate_work_are_bounded(monkeypatch):
    d=ColorCharacterDetector();seen=[]
    original=d._candidates
    def candidates(name,mask,*rest):
        seen.append(mask.shape)
        result=original(name,mask,*rest)
        assert len(result)<=8
        return result
    monkeypatch.setattr(d,'_candidates',candidates)
    large=cv2.resize(patrick(sponge(background())),(2400,1440))
    d.step(large,0)
    assert seen and all(max(shape)<=640 for shape in seen)
