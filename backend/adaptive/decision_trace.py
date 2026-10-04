"""Local public decision records; never private reasoning or raw sensor samples."""
import asyncio
import fcntl
from copy import deepcopy
from difflib import unified_diff
import json
import math
import os
from pathlib import Path
import re
import threading
import time

SCHEMA_VERSION = 1
_APPEND_LOCK = threading.Lock()


def clean(value, secrets=()):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        for key, secret in os.environ.items():
            if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(secret) >= 6:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"https?://[^\s\"<>]+", "[url redacted]", value)
        value = re.sub(r"(?i)(bearer\s+|(?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", value)
        return value[:24000]
    if isinstance(value, dict):
        return {k: clean(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, secrets) for v in value[:32]]
    return value


def pick(value, keys):
    value = value if isinstance(value, dict) else {}
    return {k: value.get(k) for k in keys.split()}


def create(session, clip, source=None):
    decision = clip["decision"]
    secret = getattr(getattr(getattr(session, "engine", None), "adapter", None), "key", "")
    previous = session.clips[source[0]] if source else {}
    frozen = previous.get("frozenEvidence") or {}
    analysis = previous.get("analysis") or {}
    q = frozen.get("quality") or {}
    eeg = decision.get("eeg_policy") or {}
    characters = analysis.get("characters") if frozen.get("gaze") and frozen.get("track") else None
    gaze = pick(analysis, "valid_gaze_s gaze_confidence comparison_s look_away_frac")
    if not frozen.get("gaze"):
        gaze = {k: None for k in gaze}
    gaze["targets"] = ({name: pick(c, "dwell_s visible_s comparison_dwell_s comparison_visible_s attention")
                        for name, c in characters.items()} if characters is not None else None)
    return clean(dict(schemaVersion=SCHEMA_VERSION, id=clip["decisionId"], sessionId=session.id,
        clipId=clip["id"], sceneIndex=clip["index"], createdAt=clip["createdAt"],
        basePrompt=clip.get("basePrompt"), basePromptExact=clean(clip.get("basePrompt"), (secret,)) == clip.get("basePrompt"), submittedPrompt=None, promptExact=None, changed=None, diff=None,
        submission="not_submitted", generationStatus=None,
        action=decision.get("action"), appliedActions=decision.get("applied_actions"),
        policy=decision.get("policy"), policyReason=decision.get("reason"),
        blockedNeutralReason=decision.get("reason") if decision.get("action") == "keep" else None,
        writer=dict(id=clip["writer"], modelId=None,
                    returnedOutput=pick(clip.get("plan"), "scene_title summary video_prompt change_note prompt_changes scene_spec")
                        if clip["writer"] != "predefined" else None,
                    outputReference=None),
        providerModelId=None, returnedOutput=None,
        evidence=dict(gaze=gaze, gazeAudit=analysis.get("gaze_audit"),
             tracking=dict(provider=previous.get("detectionProvider"),
                sources=sorted({f.get("source") for f in frozen.get("track", []) if isinstance(f.get("source"),str)}),
                experimental=any(f.get("experimental") for f in frozen.get("track", [])),
                identityStatus="uncalibrated_color_shape_hypothesis" if previous.get("detectionProvider")=="color" else None),
             eeg=dict(**pick(q, "source confidence calibrated live connectionState"),
             **pick(eeg, "eligible state action reason applied suppressed_by valid_span_s"),
             quality=pick(q, "qualityVersion cleanSeconds cleanTimeMethod featureAgeSeconds effectiveHistorySeconds selectedChannels excludedChannels channelConfidenceCeiling rejectReasons inputDiagnostics streamMetadata filter"),
             channels={name: pick(value, "usable rejectReasons rawMinUV rawMaxUV rawOffsetUV railFraction hardClipFraction peakToPeakUV filteredPeakToPeakUV gapCount")
                       for name,value in list((q.get("channelQuality") or {}).items())[:4]}),
             decisionWindow=decision.get("observationWindow"),
             blink=dict(used="gaze validity gate" if frozen else None, rateDisplayOnly=True),
             headPose=dict(used="yaw gaze validity gate" if frozen else None, displayOnly=False if frozen else None)),
        references=dict(sourceClipId=previous.get("id"), sourceSessionId=session.id if source else None,
             events="events.jsonl", analysis=f"scene{source[0]+1}_signals.json#analysis" if source else None,
             storedClipId=safe_id((clip.get("bundle") or {}).get("id")), firstFrame=None, endFrame=None, settings=None),
        timing=dict(frozenAt=frozen.get("frozenAt"), submitAttemptAt=None, submittedAt=None, readyAt=None,
                    freezeToSubmitMs=None, freezeToReadyMs=None,
                    freezeWorkMs=frozen.get("freezeWorkMs"), trackingAtFreeze=frozen.get("trackingTiming"))), (secret,))


def attempted(job, payload, at, key=None):
    trace = job.get("decisionTrace")
    if trace is None:
        return
    prompt = payload.get("prompt")
    safe = clean(prompt, (key,))
    trace.update(submittedPrompt=safe, promptExact=safe == prompt if isinstance(prompt, str) else None, submission="attempted_unconfirmed",
                 providerModelId=job.get("model"), changed=prompt != job.get("basePrompt") if prompt is not None else None)
    trace["timing"]["submitAttemptAt"] = at / 1000
    frozen = trace["timing"]["frozenAt"]
    trace["timing"]["freezeToSubmitMs"] = max(0, at-frozen*1000) if frozen is not None else None


def safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value) else None


def refresh(job, key=None):
    trace = job.get("decisionTrace")
    if trace is None:
        return
    trace["generationStatus"] = job.get("status")
    if job.get("requestId"):
        trace["submission"] = "confirmed"
        trace["timing"]["submittedAt"] = job["submittedAt"]/1000 if job.get("submittedAt") is not None else None
    elif trace["submission"] == "attempted_unconfirmed" and job.get("status") == "failed":
        trace["submission"] = "failed_unconfirmed" if job.get("requestUncertain") else "rejected"
    trace["references"].update(firstFrame=safe_id(job.get("firstFrame")), endFrame=safe_id(job.get("endFrame")),
        settings=clean(pick(job, "duration resolution seed promptExpansionMode mode")))
    if job.get("expandedPrompt") is not None or job.get("video"):
        trace["returnedOutput"] = dict(expandedPrompt=clean(job["expandedPrompt"], (key,)) if isinstance(job.get("expandedPrompt"), str) else None,
            videoReference=f"/api/jobs/{job['id']}/video" if job.get("video") else None)
    if job.get("continuationReadyAt") is not None:
        ready = job["continuationReadyAt"]
        trace["timing"]["readyAt"] = ready
        frozen = trace["timing"]["frozenAt"]
        trace["timing"]["freezeToReadyMs"] = max(0, (ready-frozen)*1000) if frozen is not None else None


def prepared(trace):
    trace = clean(deepcopy(trace))
    trace["recordedAt"] = time.time()
    prompt = trace.get("submittedPrompt")
    if prompt is not None:
        trace["diff"] = "\n".join(unified_diff((trace["basePrompt"] or "").splitlines(),
            prompt.splitlines(), fromfile="original base", tofile="provider prompt", lineterm=""))[:24000]
    return trace


class Journal:
    """Coalesced worker writes bounded records, with append locking across engines.

    Lifecycle revisions share one decision ID. Readers return the latest complete
    revision; old sessions survive the ordinary job-history retention limit.
    """
    def __init__(self, directory):
        self.path = Path(directory)/"adaptive"/"decision-traces.jsonl"
        self.pending, self.task, self.error = {}, None, None
        self.scheduled = {}

    def schedule(self, trace):
        key = (trace["sessionId"], trace["clipId"], trace["id"])
        if self.scheduled.get(key) == trace:
            return
        snapshot = deepcopy(trace)
        self.pending[key] = snapshot
        self.scheduled[key] = snapshot
        if len(self.scheduled) > 100:
            self.scheduled.pop(next(iter(self.scheduled)))
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.drain())

    async def drain(self):
        while self.pending:
            records, self.pending = list(self.pending.values()), {}
            try:
                await asyncio.to_thread(self.append, records)
                self.error = None
            except OSError:
                self.error = "Decision trace journal write failed; job history may still contain records."

    def append(self, records):
        data = b"".join((json.dumps(prepared(r), ensure_ascii=True)+"\n").encode() for r in records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _APPEND_LOCK:
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                # Isolate a crash-truncated final row before the next complete row.
                size = os.lseek(fd, 0, os.SEEK_END)
                if size and os.pread(fd, 1, size-1) != b"\n":
                    os.write(fd, b"\n")
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)

    def read(self, session_id=None):
        latest = {}
        if self.path.exists():
            with self.path.open() as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
                for line in stream:
                    if not line.endswith("\n") or len(line) > 512000:
                        continue
                    try:
                        row = json.loads(line)
                        key = (row["sessionId"], row["clipId"], row["id"])
                        if session_id is None or row["sessionId"] == session_id:
                            latest[key] = row
                    except (ValueError, KeyError, TypeError):
                        continue
        return list(latest.values())

    async def flush(self):
        if self.task:
            await asyncio.shield(self.task)
