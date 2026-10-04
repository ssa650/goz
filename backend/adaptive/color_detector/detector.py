"""Bounded color/shape recognizer for the two-character cartoon demo.

Names select allowed outputs, never supply presence. This is a domain-specific
heuristic, not reference verification or a calibrated identity probability.
No training, network, global OpenCV settings, or inherited boxes are used.
"""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class ColorDetectorConfig:
    image_size: int = 640
    fps: float = 4.0
    max_candidates_per_identity: int = 8

    def __post_init__(self):
        if isinstance(self.image_size, bool) or self.image_size not in (320, 480, 640):
            raise ValueError("image_size must be 320, 480 or 640")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)) or not math.isfinite(self.fps) or not 0 < self.fps <= 4:
            raise ValueError("fps must be finite and in (0, 4]")
        if isinstance(self.max_candidates_per_identity, bool) or not isinstance(self.max_candidates_per_identity, int) or not 1 <= self.max_candidates_per_identity <= 8:
            raise ValueError("max_candidates_per_identity must be in 1..8")


class ColorCharacterDetector:
    """One instance per clip/session; call reset() before reusing a host worker.

    step accepts RGB uint8 HxWx3, finite increasing media seconds. Scheduler
    owns decoding, CPU thread limits, cadence, expiry, cancellation and evidence
    freezing. Every result is new independent visual evidence; time continuity
    only breaks weak score ties and never rescues a failed visual candidate.
    """
    supported_names = ("SpongeBob", "Patrick")

    def __init__(self, names=supported_names, config=None, *, clip_id=None, session_id=None, generation_id=None):
        self.names = list(dict.fromkeys(names))
        if not self.names or any(n not in self.supported_names for n in self.names):
            raise ValueError("Color demo supports only SpongeBob and Patrick")
        self.config = config or ColorDetectorConfig()
        self.reset(clip_id=clip_id, session_id=session_id, generation_id=generation_id)

    def reset(self, *, clip_id=None, session_id=None, generation_id=None):
        self.clip_id, self.session_id, self.generation_id = clip_id, session_id, generation_id
        self.thumb, self.previous_t, self.previous_boxes, self.shot = None, None, {}, 0

    @staticmethod
    def _overlap(a, b):
        area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
        return area/union if union else 0

    @staticmethod
    def _same_location(a, b):
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
        smaller = min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
        return intersection/smaller >= .5 if smaller else False

    def _candidates(self, name, mask, white, brown, green):
        import cv2
        import numpy as np
        height, width = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort before expensive per-candidate features; bound noisy frame work.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.config.max_candidates_per_identity]
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            body = mask[y:y+h,x:x+w]
            area = float(np.count_nonzero(body))
            if area < max(36, width*height*.00025) or area > width*height*.55 or min(w,h) < 5:
                continue
            aspect, fill = w/h, area/(w*h)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = min(1,area/hull_area) if hull_area else 0
            local = np.zeros((h+2,w+2),np.uint8)
            local[1:-1,1:-1] = body
            thickness = float(cv2.distanceTransform(local, cv2.DIST_L2, 3).max()) / min(w,h)
            if not .18 <= aspect <= 2.8 or fill < .28 or solidity < .45 or thickness < .08:
                continue
            # Eye support in the upper torso rectangle accommodates protruding
            # cartoon eyes. Shape thickness and clothing remain independent
            # gates; the actual skin pixels (not filled contours) reject clouds.
            ey = min(h, max(1,round(h*.82)))
            eye = white[y:y+ey,x:x+w]
            eye_fraction = float(np.count_nonzero(eye != 0)) / max(1,area)
            # Clothing support below the torso; ignore distant matching colors.
            y0 = y+round(h*.55)
            y1 = min(height, y+h+round(h*.50))
            x0 = max(0,x-round(w*.10)); x1 = min(width,x+w+round(w*.10))
            cue = brown if name == "SpongeBob" else green
            clothing = cue[y0:y1,x0:x1]
            clothing_fraction = float(np.count_nonzero(clothing)) / max(1,area)
            has_eyes = .015 <= eye_fraction <= .45 and np.count_nonzero(eye) >= 6
            has_clothing = .012 <= clothing_fraction <= .8
            accepted = has_eyes and (has_clothing or (name == "SpongeBob" and fill >= .48 and solidity >= .68))
            if name == "SpongeBob" and (fill < .42 or solidity < .55):
                accepted = False
            # Colorful blobs without supporting eyes/clothes remain unknown.
            score = min(1,.30 + .20*min(1,fill/.65) + .15*min(1,thickness/.35) + .20*has_eyes + .15*has_clothing)
            # Approximate full visible body from skin plus bounded clothing band.
            bottom = y+h
            if has_clothing:
                ys, xs = np.nonzero(clothing)
                # Robust support extents ignore isolated stray pixels and sand.
                if len(ys) >= 4:
                    bottom = max(bottom,y0+int(np.percentile(ys,98))+1)
            # Yellow feet often disconnected from pants by black shoes/outline.
            if name == "SpongeBob" and has_clothing:
                bottom = min(height,bottom+round(h*.12))
            padx = max(1,round(w*.055)); pady = max(1,round(h*.025))
            box = [max(0,x-padx)/width,max(0,y-pady)/height,min(width,x+w+padx)/width,min(height,bottom+pady)/height]
            previous = self.previous_boxes.get(name)
            continuity = self._overlap(box,previous) if previous else 0
            candidates.append(dict(box=box, identity=name if accepted else None,
                candidate_identity=name, identity_status="demo_color_shape" if accepted else "unknown_insufficient_support",
                verification=dict(method="opencv_hsv_lab_shape",score_kind="uncalibrated_heuristic_score",score=round(float(score),6),
                    area_fraction=round(area/(width*height),6),aspect=round(aspect,6),fill=round(fill,6),solidity=round(solidity,6),
                    thickness=round(thickness,6),eye_fraction=round(eye_fraction,6),clothing_fraction=round(clothing_fraction,6),
                    continuity_iou=round(continuity,6))))
        return candidates

    def step(self, pixels, t):
        import cv2
        import numpy as np
        if not isinstance(pixels,np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 2:
            raise ValueError("Frame must be RGB uint8 HxWx3")
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t < 0 or (self.previous_t is not None and t <= self.previous_t):
            raise ValueError("Media timestamps must increase monotonically")
        started = time.perf_counter()
        h,w = pixels.shape[:2]
        scale = min(1,self.config.image_size/max(h,w))
        frame = cv2.resize(pixels,(max(2,round(w*scale)),max(2,round(h*scale))),interpolation=cv2.INTER_AREA) if scale < 1 else pixels
        thumb = cv2.resize(frame,(64,36),interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.thumb is not None and float(np.abs(thumb-self.thumb).mean()) > 38
        # Uniform frames never carry identity, including uniformly colored ones.
        blank = float(thumb.std(axis=(0,1)).mean()) < 5
        gap = self.previous_t is not None and t-self.previous_t > .75
        if cut or blank or gap:
            self.previous_boxes = {}
        self.shot += int(cut)
        self.thumb, self.previous_t = thumb,t
        boxes,regions = {},[]
        if not blank:
            hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(frame,cv2.COLOR_RGB2LAB)
            hu,sa,va = cv2.split(hsv); _,aa,bb = cv2.split(lab)
            masks = {
                "SpongeBob": ((hu>=19)&(hu<=37)&(sa>=65)&(va>=95)&(bb>=150)&(aa<=145)).astype(np.uint8)*255,
                "Patrick": (((hu<=22)|(hu>=165))&(sa>=45)&(sa<=220)&(va>=110)&(aa>=143)).astype(np.uint8)*255,
            }
            white = ((sa<65)&(va>165)).astype(np.uint8)*255
            brown = ((hu>=5)&(hu<=24)&(sa>70)&(va>=45)&(va<=215)&(aa<146)&(bb>132)).astype(np.uint8)*255
            green = ((hu>=32)&(hu<=88)&(sa>80)&(va>70)&(aa<126)).astype(np.uint8)*255
            kernel = np.ones((3,3),np.uint8)
            for name in self.names:
                mask = cv2.morphologyEx(masks[name],cv2.MORPH_CLOSE,kernel)
                candidates = self._candidates(name,mask,white,brown,green)
                accepted = [r for r in candidates if r['identity']]
                # Prefer full torso over a contained hand/foot with a higher score.
                accepted.sort(key=lambda r:(r['verification']['area_fraction'],r['verification']['score']+.025*r['verification']['continuity_iou']),reverse=True)
                if len(accepted) > 1:
                    best = accepted[0]
                    ambiguous = any(not self._same_location(best['box'],r['box'])
                        and r['verification']['area_fraction'] >= .35*best['verification']['area_fraction']
                        and abs(best['verification']['score']-r['verification']['score']) <= .08 for r in accepted[1:])
                    for r in accepted if ambiguous else accepted[1:]:
                        r.update(identity=None,identity_status="ambiguous_color_locations" if ambiguous else "suppressed_duplicate")
                regions.extend(candidates)
            boxes = {r['identity']:r['box'] for r in regions if r['identity']}
        self.previous_boxes = dict(boxes)
        status = "blank_frame" if blank else "observed" if boxes else "unavailable"
        candidate_counts = {}
        for region in regions:
            reason = region["identity_status"]
            candidate_counts[reason] = candidate_counts.get(reason,0)+1
        return dict(t=round(t,6),media_timestamp_s=round(t,6),boxes=boxes,regions=regions,
            unknown=[n for n in self.names if n not in boxes],shot_id=self.shot,cut=bool(cut),
            valid_until=round(t+min(.5,1/self.config.fps),6),source="opencv_color_shape_demo",status=status,
            abstention_reason=None if boxes else "blank_frame" if blank else "no_unambiguous_color_shape_match",
            coordinate_space="video-normalized",provider_confidence_available=False,experimental=True,
            candidate_counts=candidate_counts,
            clip_id=self.clip_id,session_id=self.session_id,generation_id=self.generation_id,
            inference_seconds=round(time.perf_counter()-started,6),
            resources=dict(device="cpu",recognition_fps=self.config.fps,image_size=self.config.image_size,
                max_candidates_per_identity=self.config.max_candidates_per_identity,thread_limit_owner="host_worker"))
