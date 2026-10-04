"""Offline measurement only; annotations never feed detector inference."""
import math
import statistics


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / area if area > 0 else 0


def evaluate(annotations, records, *, iou_threshold=.5, wall_seconds=None, first_progress_seconds=None):
    """Exact timestamp matching, identity+IoU precision/recall and abstentions.

    Identity switches are counted only for explicit human instance IDs, with
    consecutive visible samples in the same shot. No guessed temporal matching.
    Unknown rate = GT instances with no spatially matched named prediction / GT.
    A wrong named identity is a FP+FN and separately an identity error.
    """
    keys = [(a["clip"], round(a["t"], 6)) for a in annotations]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate annotation timestamp")
    by_key = {}
    for r in records:
        key = (r["clip"], round(r["t"], 6))
        if key in by_key:
            raise ValueError("Duplicate prediction timestamp")
        by_key[key] = r
    tp = fp = fn = unknown = identity_errors = switches = ground_truth_count = 0
    previous, latencies, first_valid = {}, [], {}
    for a in sorted(annotations, key=lambda a: (a["clip"], a["t"])):
        r = by_key.get((a["clip"], round(a["t"], 6)), {})
        predictions = list(r.get("boxes", {}).items())
        gt = a["detections"]
        ground_truth_count += len(gt)
        # Greedy strongest geometry match, once per instance (two cast members).
        pairs = sorted(((iou(g["box"], b), gi, pi) for gi, g in enumerate(gt) for pi, (_, b) in enumerate(predictions)), reverse=True)
        matched_gt, matched_predictions, assignment = set(), set(), {}
        for score, gi, pi in pairs:
            if score < iou_threshold or gi in matched_gt or pi in matched_predictions:
                continue
            matched_gt.add(gi); matched_predictions.add(pi)
            assignment[gi] = predictions[pi][0]
            if gt[gi]["identity"] == predictions[pi][0]:
                tp += 1
                first_valid.setdefault(a["clip"], a["t"])
            else:
                fp += 1; fn += 1; identity_errors += 1
        fp += len(predictions)-len(matched_predictions)
        fn += len(gt)-len(matched_gt)
        unknown += len(gt)-len(matched_gt)
        current = {}
        for gi, g in enumerate(gt):
            if "instance_id" not in g:
                continue
            key = (a["clip"], a.get("shot_id", 0), g["instance_id"])
            identity = assignment.get(gi)
            if identity and previous.get(key) and previous[key] != identity:
                switches += 1
            current[key] = identity
        previous = {k: v for k, v in previous.items() if k[0] != a["clip"]} | current
        latency = r.get("inference_seconds")
        if isinstance(latency, (int, float)) and math.isfinite(latency):
            latencies.append(latency)
    ordered = sorted(latencies)
    return dict(frames=len(annotations), predicted_frames=sum(k in by_key for k in keys),
        ground_truth_instances=ground_truth_count, true_positive=tp, false_positive=fp, false_negative=fn,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        unknown_rate=unknown/ground_truth_count if ground_truth_count else None,
        identity_errors=identity_errors, identity_switches=switches,
        inference_p50_seconds=statistics.median(latencies) if latencies else None,
        inference_p95_seconds=ordered[max(0, math.ceil(.95*len(ordered))-1)] if ordered else None,
        first_valid_media_seconds=first_valid, wall_seconds=wall_seconds,
        first_progress_seconds=first_progress_seconds, iou_threshold=iou_threshold)
