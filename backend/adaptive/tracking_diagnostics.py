"""Bounded, local detector and render evidence. Never raw EEG/camera samples."""
from collections import Counter
from pathlib import Path
import math
import time
from uuid import uuid4

from .decision_trace import Journal, safe_id

MAX_EVENTS = 1600


class TrackingJournal(Journal):
    def __init__(self, directory, session_id, clip_id, generation_id=None):
        if not safe_id(session_id) or not safe_id(clip_id):
            raise ValueError("Invalid diagnostic identity.")
        super().__init__(directory)
        self.path = Path(directory)/"adaptive"/session_id/f"clip-{clip_id}_tracking.jsonl"
        self.session_id, self.clip_id, self.generation_id = session_id, clip_id, generation_id
        self.owner, self.sequence, self.started = uuid4().hex[:12], 0, time.monotonic()

    def record(self, kind, **fields):
        if self.sequence >= MAX_EVENTS:
            return
        self.sequence += 1
        self.schedule(dict(schemaVersion=1, recordType="tracking_diagnostic",
            id=f"{self.owner}-{self.sequence}", sessionId=self.session_id, clipId=self.clip_id,
            generationId=self.generation_id, event=kind, wall=time.time(),
            elapsedMs=round((time.monotonic()-self.started)*1000, 3), **fields))


def for_clip(session, clip):
    journals = getattr(session, "_tracking_journals", None)
    if journals is None:
        journals = session._tracking_journals = {}
    generation = clip.get("trackingGenerationId") or clip.get("jobId") or clip.get("decisionId")
    key = (clip["id"], generation)
    if key not in journals:
        journals[key] = TrackingJournal(session.engine.directory, session.id, clip["id"], generation)
    return journals[key]


def for_video(video, session_id, clip_id, generation_id, directory=None):
    # Normal app media is data/media/<job>.mp4. Only autodetect an existing
    # session directory; standalone/offline callers can opt in explicitly.
    root = Path(directory) if directory is not None else Path(video).resolve().parent.parent
    if directory is None and (not safe_id(session_id) or not (root/"adaptive"/session_id).is_dir()):
        return None
    return TrackingJournal(root, session_id, clip_id, generation_id)


def frame_evidence(frame, playback=None):
    target = playback.get("media_t") if playback else None
    expiry = frame.get("valid_until")
    return dict(mediaTime=frame.get("t"), validUntil=expiry, cut=frame.get("cut"),
        source=frame.get("source"), status=frame.get("status"), inputGap=frame.get("input_gap"),
        candidateCounts=frame.get("candidate_counts"), resources=frame.get("resources"),
        abstentionReason=frame.get("abstention_reason"), boxes=frame.get("boxes", {}),
        candidates=frame.get("regions", [])[:16], unknown=frame.get("unknown", []),
        scheduling=frame.get("scheduling"), decode=frame.get("decode"),
        transport=frame.get("transport"), playbackMediaTime=target,
        expiredAtReceipt=target > expiry if target is not None and expiry is not None else None)


def publication_evidence(frames):
    return dict(records=len(frames), namedRecords=sum(bool(f.get("boxes")) for f in frames),
        throughMediaS=max((f.get("t", 0) for f in frames), default=None),
        candidateStatuses=dict(Counter(r.get("identity_status", "unspecified")
            for f in frames for r in f.get("regions", []))))


def read(directory, session_id):
    if not safe_id(session_id):
        raise ValueError("Invalid diagnostic session.")
    root = Path(directory)/"adaptive"/session_id
    events = []
    for path in sorted(root.glob("clip-*_tracking.jsonl"))[:16]:
        journal = TrackingJournal(directory, session_id, path.name[5:-15])
        journal.path = path
        events.extend(journal.read(session_id)[:MAX_EVENTS*2])
    return sorted(events, key=lambda event:event.get("wall", 0))


def overlay_evidence(value, names):
    """Accept only bounded browser-reported geometry; never trust URLs/raw data."""
    if not isinstance(value, dict):
        raise ValueError("Invalid overlay diagnostic.")
    result = {"reportedBy": "browser", "enabled": value.get("enabled") is True,
              "status": str(value.get("status", "unspecified"))[:48]}
    for key in ("mediaTime", "frameMediaTime", "validUntil", "devicePixelRatio"):
        number = value.get(key)
        if number is not None and (isinstance(number, bool) or not isinstance(number,(int,float))
                or not math.isfinite(number) or not -1e5 <= number <= 1e5):
            raise ValueError("Invalid overlay number.")
        result[key] = number
    for key in ("contentRect", "canvasRect", "videoRect"):
        rect = value.get(key)
        if rect is None:
            result[key] = None
            continue
        if not isinstance(rect, dict):
            raise ValueError("Invalid overlay rectangle.")
        result[key] = {}
        for axis in ("x", "y", "w", "h"):
            number = rect.get(axis)
            if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or abs(number)>1e5:
                raise ValueError("Invalid overlay rectangle.")
            result[key][axis] = number
    boxes = value.get("boxes", [])
    if not isinstance(boxes,list) or len(boxes)>6:
        raise ValueError("Invalid overlay boxes.")
    result["boxes"] = []
    for box in boxes:
        if not isinstance(box,dict) or box.get("name") not in names:
            raise ValueError("Invalid overlay target.")
        item = {"name": box["name"]}
        for key in ("normalized", "canvas"):
            coords = box.get(key)
            if not isinstance(coords,list) or len(coords)!=4 or any(isinstance(n,bool)
                or not isinstance(n,(int,float)) or not math.isfinite(n) or abs(n)>1e5 for n in coords):
                raise ValueError("Invalid overlay coordinates.")
            item[key] = coords
        result["boxes"].append(item)
    return result
