"""Bounded visual identity checks on generated/story frames only, never webcam.

One cached Responses image-input request per clip (max eight scene frames).
References establish appearance; expected cast is optional, not presumed present.
Returned boxes are model observations, not calibrated probabilities.
"""
import base64
import hashlib
import json
import math
import os
import time
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ENDPOINT = 'https://api.openai.com/v1/responses'
SCHEMA_VERSION = 2


def _image(jpeg):
    return dict(type='input_image', image_url='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(), detail='high')


def reference_inputs(names):
    # Same visually inspected reference crops as local geometry verification.
    from .tracks import REFERENCE_CROPS
    root = Path(__file__).resolve().parents[2] / 'presets/secret-box/frames'
    content = []
    for name in names:
        if name not in REFERENCE_CROPS:
            continue
        filename, rect = REFERENCE_CROPS[name][0]
        with Image.open(root / filename) as image:
            w,h = image.size
            crop = image.crop(tuple(int(v * (w if i % 2 == 0 else h)) for i,v in enumerate(rect)))
            crop.thumbnail((384,384))
            buffer = BytesIO();crop.save(buffer, format='JPEG')
        content.extend([dict(type='input_text', text=f'Appearance reference only: {name}. This is not a scene frame and does not prove presence.'), _image(buffer.getvalue())])
    return content


def schema(names):
    box = dict(type='object', properties=dict(identity=dict(type='string', enum=names + ['unknown']),
        box=dict(type='array', items=dict(type='number'), minItems=4, maxItems=4),
        evidence=dict(type='string')), required=['identity','box','evidence'], additionalProperties=False)
    frame = dict(type='object', properties=dict(frame_index=dict(type='integer'), detections=dict(type='array',items=box)),
        required=['frame_index','detections'], additionalProperties=False)
    return dict(type='object',properties=dict(frames=dict(type='array',items=frame)),required=['frames'],additionalProperties=False)


def validate(result, count, names):
    if not isinstance(result,dict) or set(result) != {'frames'} or not isinstance(result['frames'],list):
        raise ValueError('Malformed identity result.')
    out = {}
    for frame in result['frames']:
        if not isinstance(frame,dict) or not isinstance(frame.get('frame_index'),int) or not 0 <= frame['frame_index'] < count:
            raise ValueError('Wrong identity frame index.')
        if frame['frame_index'] in out or not isinstance(frame.get('detections'),list) or len(frame['detections']) > 12:
            raise ValueError('Duplicate or unbounded identity frame.')
        detections = []
        for detection in frame['detections']:
            b = detection.get('box')
            if detection.get('identity') not in names + ['unknown'] or not isinstance(b,list) or len(b) != 4:
                raise ValueError('Malformed identity detection.')
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0 <= v <= 1 for v in b):
                raise ValueError('Identity box is not normalized finite video coordinates.')
            if b[0] >= b[2] or b[1] >= b[3]:
                raise ValueError('Identity box has invalid geometry.')
            if not isinstance(detection.get('evidence'),str) or not detection['evidence'].strip():
                raise ValueError('Identity must have visual evidence.')
            detections.append(dict(identity=detection['identity'],box=b,evidence=detection['evidence'][:300]))
        out[frame['frame_index']] = detections
    if set(out) != set(range(count)):
        raise ValueError('Identity response omitted scene frames.')
    return out


async def verify_frames(frames, names, directory=None, transport=None):
    key = os.getenv('OPENAI_API_KEY','')
    model = os.getenv('GOZ_IDENTITY_MODEL') or os.getenv('OPENAI_MODEL')
    diagnostic = dict(endpoint=ENDPOINT, model=model or None)
    if os.getenv('GOZ_IDENTITY', 'auto') == 'off':
        return None, dict(diagnostic,status='disabled',reason='Scene identity verification is disabled by GOZ_IDENTITY=off.')
    if not key:
        return None, dict(diagnostic,status='unavailable',reason='OPENAI_API_KEY is not configured.')
    if not model:
        return None, dict(diagnostic,status='unavailable',reason='Configure GOZ_IDENTITY_MODEL or OPENAI_MODEL.')
    sampled = [f for f in frames if f.get('jpeg')][:8]
    selected, indexes = [], {}
    for frame in sampled:
        digest = hashlib.sha256(frame['jpeg']).hexdigest()
        if digest not in indexes:
            indexes[digest] = len(selected)
            selected.append(frame)
    def restore_times(result):
        return {f['t']: [dict(d) for d in result[indexes[hashlib.sha256(f['jpeg']).hexdigest()]]] for f in sampled}

    if not selected:
        return {}, dict(status='empty', elapsed_seconds=0)
    names = list(dict.fromkeys(names))
    references = reference_inputs(names)
    effort = os.getenv('GOZ_IDENTITY_EFFORT', 'medium' if model == 'gpt-6-luna' else '')
    digest = hashlib.sha256(json.dumps([model,effort,names,SCHEMA_VERSION,references],sort_keys=True).encode() + b''.join(f['jpeg'] for f in selected)).hexdigest()
    cache = Path(directory)/'detections'/f'identity-{digest}.json' if directory else None
    started = time.perf_counter()
    try:
        if cache and cache.exists():
            value = json.loads(cache.read_text())
            result = validate(value['result'],len(selected),names)
            return restore_times(result),dict(value['provenance'],cache_hit=True,elapsed_seconds=round(time.perf_counter()-started,4))
        content = [dict(type='input_text',text=(
            'Inspect ONLY the actual pixels of each SCENE frame independently. Detect the visible character bodies and faces with tight bounding boxes. '
            'The allowed identity names are '+json.dumps(names)+'. These are possibilities, not a required cast. '
            'Use the supplied appearance references to distinguish them. Do not infer identity from color alone, the show title, a box, background flowers, or prior/next frames. '
            'If a named character is absent, do not output them. Empty detections are valid. If identity is ambiguous output unknown. '
            'Include partially visible characters only if their identity is visually supported; box only the visible portion. '
            'Boxes are [left,top,right,bottom] normalized 0..1 relative to the complete SCENE image including any black bars. '
            'Return every frame_index exactly once and a short concrete visual evidence phrase per detection. Reference images do not have frame_index.'))] + references
        for i,frame in enumerate(selected):
            # Visible index prevents accidental association of adjacent images.
            image = Image.open(BytesIO(frame['jpeg'])).convert('RGB')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,155,24), fill='black')
            draw.text((4,4),f'SCENE FRAME {i}',fill='white',font_size=17)
            labelled = BytesIO();image.save(labelled,format='JPEG',quality=92)
            content.extend([dict(type='input_text',text=f'SCENE frame_index={i}, media timestamp={frame["t"]:.6f} seconds. Read the visible SCENE FRAME number; do not shift or borrow from adjacent frames.'),_image(labelled.getvalue())])
        payload = dict(model=model,input=[dict(role='user',content=content)],max_output_tokens=4500,
                       text=dict(format=dict(type='json_schema',name='scene_identity',strict=True,schema=schema(names))),store=False)
        if effort:
            payload['reasoning'] = dict(effort=effort)
        async with httpx.AsyncClient(timeout=40,transport=transport) as client:
            response=await client.post(ENDPOINT,headers={'Authorization':'Bearer '+key},json=payload)
        response.raise_for_status()
        body=response.json()
        output=''.join(c.get('text','') for item in body.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text')
        raw=json.loads(output)
        try:
            result=validate(raw,len(selected),names)
        except (ValueError,KeyError,TypeError) as validation_error:
            if cache:
                cache.parent.mkdir(parents=True,exist_ok=True)
                rejected = cache.with_name(cache.stem + '-rejected.json')
                rejected.write_text(json.dumps(dict(result=raw,reason=str(validation_error),model=model,
                    reasoning_effort=effort,response_id=body.get('id'))))
            raise
        provenance=dict(status='verified',endpoint=ENDPOINT,model=model,response_id=body.get('id'),
                        schema_version=SCHEMA_VERSION,reasoning_effort=effort or 'provider_default',elapsed_seconds=round(time.perf_counter()-started,4),cache_hit=False,
                        frame_count=len(selected),sample_count=len(sampled),confidence_available=False,usage=body.get('usage'))
        if cache:
            cache.parent.mkdir(parents=True,exist_ok=True)
            temp=cache.with_suffix('.tmp');temp.write_text(json.dumps(dict(result=raw,provenance=provenance)));temp.replace(cache)
        return restore_times(result),provenance
    except Exception as error:
        # Never log response bodies, auth headers or secret-bearing requests.
        status_code = error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
        if status_code:
            advice = {400:'Check image-input, structured-output and reasoning support for the configured model.',
                      401:'Existing API credentials were rejected.',403:'The configured account cannot access this request/model.',
                      404:'Check the configured model name and account access.',429:'Provider rate or account limit; retry a later session.'}.get(status_code,'Provider request failed; retry a later session.')
            reason = f'HTTP {status_code}. {advice}'
        elif isinstance(error,httpx.TimeoutException):
            reason = 'Identity request timed out after 40 seconds; attribution remains unavailable.'
        elif isinstance(error,(ValueError,KeyError,TypeError)):
            reason = ('Identity response failed validation: '+str(error)) if type(error) is ValueError else 'Identity response was malformed or failed frame/box validation.'
        else:
            reason = f'Identity processing failed ({type(error).__name__}).'
        return None, dict(diagnostic,status='unavailable',reason=reason,http_status=status_code,
                          elapsed_seconds=round(time.perf_counter()-started,4))
