"""Cumulative viewing EEG trial; no inference of preference or brain state.

The session owns this collector. Call start_clip at playback start, collect at
viewing ticks (also after the opening decision), and compare at the existing
generation deadline. No waiting, acquisition changes, or remote calls are needed.
"""
import copy
import json
import math
import statistics

from . import eeg_policy as baseline

VERSION = "experimental-eeg-cumulative-prior-clips-v2"
RUN_MODE = "cumulative_prior_clips"
CURRENT_WINDOW_S = 12.0
MIN_REFERENCE_S = 12.0
CURRENT_COVERAGE = .8
BLOCK_COVERAGE = .75
RESULT_TTL_S = 5.0
MIN_CHANGE = .35
NOISE_MULTIPLIER = 2.0
RETAIN_FRACTION = .6
FEATURE_SUPPORT_S = 2.0


def compatibility_key(quality):
    """Fail closed unless the caller supplies the feed's calibration snapshot."""
    q = quality if isinstance(quality, dict) else {}
    c = q.get("calibration")
    channels = q.get("selectedChannels", q.get("goodChannels"))
    if (not isinstance(c, dict) or not isinstance(channels, (list, tuple))
            or len(channels) < 2 or not all(isinstance(v,str) and v for v in channels)
            or len(set(channels)) != len(channels)
            or not isinstance(c.get("channels"),(list,tuple))
            or list(c["channels"]) != list(channels)
            or not baseline.finite(c.get("cleanSeconds")) or c["cleanSeconds"] < 60):
        return None
    if q.get("source") in ("muse", "sim"):
        medians, scales = c.get("channelMedians"), c.get("channelScales")
        if (not isinstance(medians, (list, tuple)) or not isinstance(scales, (list, tuple))
                or len(medians) != len(channels) or len(scales) != len(channels)
                or not all(baseline.finite(v) for v in medians)
                or not all(baseline.finite(v) and v > 0 for v in scales)
                or not baseline.finite(c.get("calibratedAt"))):
            return None
    elif q.get("source") == "mindmonitor":
        if (not baseline.finite(c.get("median")) or not baseline.finite(c.get("scale"))
                or c["scale"] <= 0 or not baseline.finite(c.get("readyAt"))):
            return None
    else:
        return None
    if not isinstance(q.get("deviceId"),str) or not q["deviceId"] or c.get("deviceId") != q["deviceId"]:
        return None
    try:
        return json.dumps([q["source"], q["deviceId"], list(channels), c], sort_keys=True,allow_nan=False)
    except (TypeError,ValueError):
        return None


def _window_stats(rows, start, end):
    """Unique accepted intervals; gaps/artifacts provide no duration credit."""
    rows = sorted((r for r in rows if start <= r[0] <= end), key=lambda r: r[0])
    buckets, coverage, previous = {}, 0., None
    for t, value, z, artifact in rows:
        if artifact is not False or not baseline.finite(value) or value <= 0 or not baseline.finite(z) or abs(z) >= 5:
            previous = None
            continue
        if previous is not None and 0 < t-previous <= baseline.MAX_GAP_S+1e-6:
            coverage += t-previous
        previous = t
        buckets.setdefault(math.floor((t-start+1e-6)/baseline.BUCKET_S), []).append(z)
    values = [statistics.median(v) for _,v in sorted(buckets.items())]
    return dict(values=values, coverage_s=coverage,
                effective_count=int(coverage/FEATURE_SUPPORT_S),
                mean=statistics.fmean(values) if values else None)


def _spread(values):
    median = statistics.median(values)
    mad = 1.4826*statistics.median(abs(v-median) for v in values)
    quartiles = statistics.quantiles(values,n=4,method="inclusive")
    return max(.1,mad,(quartiles[2]-quartiles[0])/1.349)


class CumulativeEEGHistory:
    """Session-local sustained evidence and immutable prior-reference snapshots.

    A changed calibration/channel set retires incompatible history. A decision
    never sees samples from its current clip in its reference, even after more
    collection. Late collection for a closed clip or another session is ignored.
    """
    def __init__(self, session_id):
        self.reset(session_id)

    def reset(self, session_id):
        self.session_id = session_id
        self._clips = []
        self._active = None
        self._reference = ()
        self._decision = None
        self._last_state = None
        self._last_key = None

    def start_clip(self, clip_id, start, duration_s=15):
        if not baseline.finite(start) or not baseline.finite(duration_s) or duration_s <= 0:
            raise ValueError("Finite playback start and positive duration required")
        if any(c["id"] == clip_id for c in self._clips):
            raise ValueError("Clip playback may only start once per session")
        self._reference = tuple(copy.deepcopy(self._clips))
        self._active = dict(id=clip_id, start=start, end=start+duration_s, rows={}, key=None)
        self._clips.append(self._active)
        self._decision = None

    def collect(self, session_id, clip_id, samples, quality, *, observed_at):
        c = self._active
        if (session_id != self.session_id or c is None or clip_id != c["id"]
                or not baseline.finite(observed_at) or observed_at < c["start"]
                or not isinstance(samples, (tuple, list))):
            return 0
        q = quality if isinstance(quality, dict) else {}
        key = compatibility_key(q)
        valid_quality = key is not None and baseline.quality_failure(q) is None
        if key is not None and c["key"] != key:
            c["rows"].clear()  # Never combine normalized features from different baselines.
            c["key"] = key
        warmup = 2. if q.get("source") in ("muse", "sim") else 0.
        added = 0
        for row in samples:
            if (not isinstance(row, (tuple, list)) or len(row) != 4 or not baseline.finite(row[0])
                    or not c["start"]+warmup-1e-6 <= row[0] <= min(c["end"], observed_at)+1e-6
                    or observed_at-row[0] > baseline.FRESH_S+1e-6):
                continue
            t = row[0]
            candidate = tuple(row) if valid_quality else (t, None, None, True)
            old = c["rows"].get(t)
            if old is not None and old != candidate:
                candidate = (t, None, None, True)  # Conflicts cannot replace rejected rows with clean ones.
            if old is None:
                added += 1
            c["rows"][t] = candidate
        return added

    def compare(self, samples, quality, window):
        """Read a causal sustained result at the gaze deadline; never await EEG.

        The samples argument remains for the session API; only the continuously
        quality-attested collector ledger is used. A completed result may be up
        to 5s old while live raw quality must still pass its original freshness.
        """
        if self._decision is not None:
            return copy.deepcopy(self._decision)
        c = self._active
        q = quality if isinstance(quality,dict) else {}
        result = dict(policy=VERSION,run_mode=RUN_MODE,experimental=True,
            heuristic="12s sustained physiological change; engineering thresholds, not validated",
            eligible=False,state="unknown",action="keep",reason="Sustained EEG evidence unavailable",
            source=q.get("source"),median_z=None,delta=None,noise_floor=None,
            change_threshold=None,reference_mean=None,current_mean=None,confidence=None,
            confidence_basis="feed signal quality; not classifier confidence",
            reference_count=0,reference_clip_count=0,reference_clip_ids=[],
            reference_coverage_s=0.,samples=0,valid_span_s=0.,coverage=None,
            current_window=None,reference_window=None,evidence_age_s=None,
            raw_support_exclusion_s=FEATURE_SUPPORT_S if q.get("source") in ("muse","sim") else 0.,
            current_effective_count=0,reference_effective_count=0,
            result_ttl_s=RESULT_TTL_S,applied=False)
        def finish(reason):
            result["reason"]=reason
            if result["eligible"]:
                self._last_state=result["state"]
                self._last_key=key
            else:
                self._last_state=None
            self._decision=copy.deepcopy(result)
            return copy.deepcopy(result)
        if (c is None or not isinstance(window,(list,tuple)) or len(window)!=2
                or not all(baseline.finite(t) for t in window)
                or abs(window[0]-c["start"])>1e-6 or not c["start"]<window[1]<=c["end"]+1e-6):
            return finish("Frozen generation deadline unavailable")
        deadline=window[1]
        result["decision_deadline"]=deadline
        key=compatibility_key(q)
        if key is None:
            return finish("A genuine 60-clean-second baseline and stable EEG identity are required")
        failure=baseline.quality_failure(q)
        if failure:
            return finish(failure)
        if self._last_key!=key:
            self._last_state=None
        # Freeze older clips at current clip start. Do not admit delayed events
        # or changes to prior ledgers into either side of this comparison.
        prior=[p for p in self._reference if p["key"]==key]
        all_clips=prior+([c] if c["key"]==key else [])
        ledger={}
        for clip in all_clips:
            for t,row in clip["rows"].items():
                if t<=deadline:
                    old=ledger.get(t)
                    ledger[t]=row if old is None or old==row else (t,None,None,True)
        # A stopped/artifact raw signal cannot reuse even a recent old result.
        if not ledger or deadline-max(ledger)>baseline.FRESH_S+1e-6:
            return finish("No fresh viewing EEG ledger at the generation deadline")
        newest=ledger[max(ledger)]
        if newest[3] is not False or not baseline.finite(newest[2]) or abs(newest[2])>=5:
            return finish("Latest viewing EEG feature is invalid; abstain")
        candidates=sorted((t for t in ledger if deadline-RESULT_TTL_S<=t<=deadline),reverse=True)
        best_reason="Insufficient sustained viewing EEG evidence"
        for end in candidates:
            start=end-CURRENT_WINDOW_S
            current=_window_stats(ledger.values(),start,end)
            result.update(current_window=[start,end],evidence_age_s=deadline-end,
                          current_effective_count=current["effective_count"],
                          samples=len(current["values"]),valid_span_s=current["coverage_s"])
            if current["coverage_s"]+1e-6<CURRENT_COVERAGE*CURRENT_WINDOW_S:
                best_reason="12s current EEG window lacks 80% unique clean coverage"
                continue
            blocks=[_window_stats(ledger.values(),start+i*4,start+(i+1)*4) for i in range(3)]
            if any(b["coverage_s"]+1e-6<BLOCK_COVERAGE*4 for b in blocks):
                best_reason="Sustained EEG window lacks clean coverage in each 4s block"
                continue
            cutoff=start-result["raw_support_exclusion_s"]
            reference_rows=[r for p in prior for t,r in p["rows"].items() if t<=cutoff]
            reference_start=min((r[0] for r in reference_rows),default=cutoff)
            # Stats across clips are separated by invalid boundary rows so an
            # interval between independent playback clips earns no time credit.
            refs=[]
            for p in prior:
                stats=_window_stats(p["rows"].values(),reference_start,cutoff)
                if stats["values"]:
                    refs.append((p["id"],stats))
            values=[v for _,r in refs for v in r["values"]]
            coverage=sum(r["coverage_s"] for _,r in refs)
            result.update(reference_window=[reference_start,cutoff],reference_count=len(values),
                reference_coverage_s=coverage,reference_effective_count=int(coverage/FEATURE_SUPPORT_S),
                reference_clip_ids=[i for i,_ in refs],reference_clip_count=len(refs),
                coverage=dict(current_s=current["coverage_s"],current_fraction=current["coverage_s"]/CURRENT_WINDOW_S,
                    prior_clean_s=coverage,reference_duration_s=max(0.,cutoff-reference_start)))
            if coverage+1e-6<MIN_REFERENCE_S:
                best_reason="Need 12 unique clean seconds of disjoint prior viewing reference"
                continue
            ref_mean=statistics.fmean(values)
            delta=current["mean"]-ref_mean
            # Count non-overlapping raw-support durations, never packets or
            # overlapping 250ms feature buckets as independent observations.
            noise=max(_spread(values),_spread(current["values"]))*math.sqrt(
                1/max(1,current["effective_count"])+1/max(1,int(coverage/FEATURE_SUPPORT_S)))
            enter=max(MIN_CHANGE,NOISE_MULTIPLIER*noise)
            sign=1 if delta>0 else -1
            state="above_prior_clips" if sign>0 else "below_prior_clips"
            threshold=enter*(RETAIN_FRACTION if state==self._last_state else 1.)
            block_deltas=[b["mean"]-ref_mean for b in blocks]
            agreement=sum(sign*(v-ref_mean)>=threshold for v in current["values"])/len(current["values"])
            changed=abs(delta)>=threshold and agreement>=baseline.AGREEMENT and all(sign*d>=threshold for d in block_deltas)
            result.update(eligible=True,delta=delta,noise_floor=noise,change_threshold=threshold,
                entry_threshold=enter,reference_mean=ref_mean,current_mean=current["mean"],
                median_z=statistics.median(current["values"]),confidence=q["confidence"],
                agreement=agreement,persistence_block_deltas=block_deltas,
                state=state if changed else "near_prior_clips",
                action=("faster_pacing" if sign>0 else "slower_pacing") if changed else "keep",
                feature="existing calibrated per-channel EEG normalized index",
                window=[start,end])
            return finish(f"Experimental 12s sustained EEG delta {delta:+.3f}, noise {noise:.3f}; "
                f"{current['coverage_s']:.1f}s clean current vs {coverage:.1f}s disjoint reference. "
                "Engineering heuristic; cause and valence unknown. "+
                ("Try one bounded pacing cue." if changed else "No persistent qualified shift; preserve delivery."))
        return finish(best_reason)
