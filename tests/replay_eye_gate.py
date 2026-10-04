"""Offline eye eligibility replay; raw gaze features/pixels are not reconstructed.

Run from the repository root with --baseline pointing to the pre-change
gaze_pipeline.py snapshot. No hardware, model inference, sockets, UI or writes
to historical/calibration data occur. The only write is the requested report.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.gaze_pipeline import BlinkGate, calibration_eye_evidence
from backend.gaze_worker import validate_compatibility


def observation(score, lid, face=True):
    return SimpleNamespace(ok=face, blink=score,
        features=[0., 0., 0., lid, 0., 0., 0., lid]+[0.]*6)


def replay(run, model, baseline, repo):
    metadata = json.loads(model.with_suffix(".report.json").read_text())
    validate_compatibility(metadata, repo, metadata["profileId"], metadata["screen"])
    evidence = calibration_eye_evidence(model, metadata)
    if not evidence["enabled"]: raise ValueError(evidence["reason"])
    spec = importlib.util.spec_from_file_location("baseline_eye_gate", baseline)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    scenes = {str(n): json.loads((run/f"scene{n}_complete_signals.json").read_text())["gaze"] for n in range(1, 5)}
    unique = {(s["setupId"], s["frameSequence"]): s for rows in scenes.values() for s in rows}
    rows = sorted(unique.values(), key=lambda s: s["t"])
    old, new = module.BlinkGate("/tmp/goz-absent-eye-profile"), BlinkGate("/tmp/goz-absent-eye-profile", evidence)
    first = rows[0]
    for gate in (old, new):
        gate.frozen = bool(first["blink"])
        gate.gated_at = first["t"]-first["blinkDiagnostics"]["gatedSeconds"] if gate.frozen else None
    decisions = {}
    for s in rows:
        e = s["blinkDiagnostics"]
        obs = observation(e["score"], e["openness"], s["face"])
        prior = not old.update(obs, s["t"]) and not s["stale"]
        after = not new.update(obs, s["t"], fresh=not s["stale"])
        decisions[(s["setupId"], s["frameSequence"])] = (prior, after, new.diagnostics(s["t"]))
    scene_report = {}
    for name, samples in scenes.items():
        ds = [decisions[(s["setupId"], s["frameSequence"])] for s in samples]
        closure = [i for i,s in enumerate(samples) if s["blinkDiagnostics"]["score"] > .28
                   or s["blinkDiagnostics"]["openness"] < .16]
        scene_report[name] = dict(samples=len(samples), recordedValid=sum(s["valid"] for s in samples),
            baselineEyeEligible=sum(d[0] for d in ds), afterEyeEligible=sum(d[1] for d in ds),
            eligibleWithExplicitUncertainty=sum(d[1] and d[2]["uncertain"] for d in ds),
            closureEvidenceSamples=len(closure), acceptedClosureEvidence=sum(ds[i][1] for i in closure),
            reasons=dict(Counter(d[2]["reason"] for d in ds)))
    tests = {}
    for name, score, lid, face, fresh, quality, enabled in [
        ("persistent-closed-lids", .05, .03, True, True, None, True),
        ("persistent-high-blink-score", .8, .28, True, True, None, True),
        ("face-loss", .24, .275, False, True, None, True),
        ("stale-frames", .24, .275, True, False, None, True),
        ("invalid-predictions", .24, .275, True, True, "invalid-prediction", True),
        ("poor-lighting", .24, .275, True, True, "lighting-out-of-range", True),
        ("weak-lid-evidence", .24, .23, True, True, None, True),
        ("missing-calibration-evidence", .24, .275, True, True, None, False),
    ]:
        gate = BlinkGate("/tmp/goz-absent-eye-profile", evidence if enabled else None)
        gate.update(observation(.8, .03), 0.)
        eligible = sum(not gate.update(observation(score, lid, face), i/30,
                                      fresh=fresh, quality_reason=quality) for i in range(1, 301))
        tests[name] = dict(samples=300, eligible=eligible, falseAcceptances=eligible,
                           finalReason=gate.reason)
        assert eligible == 0, name
    gate = BlinkGate("/tmp/goz-absent-eye-profile", evidence)
    blink_accepts = eligible = 0
    for i in range(120):
        closure = 30 <= i < 36
        opened = not gate.update(observation(.8, .03) if closure else observation(.05, .275), i/30)
        eligible += opened; blink_accepts += closure and opened
    tests["synthetic-blink-200ms-and-strict-reopen"] = dict(samples=120, eligible=eligible,
        closureSamples=6, falseAcceptances=blink_accepts)
    assert blink_accepts == 0 and eligible > 0
    return dict(run=run.name, compatibilityVerified=True, baselineSha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        evidence=evidence, uniqueRecordedSamples=len(rows), scenes=scene_report, syntheticScenarios=tests,
        limits=["Recorded replay measures eye-gate eligibility only: raw feature vectors, lighting, face scale and predictions were not logged.",
                "High blink scores/collapsed lids are observed closure evidence, not human-labelled blink ground truth.",
                "Synthetic zero false acceptances do not estimate a real-world false acceptance rate.",
                "No live device, real-face inference, gaze accuracy or frontend mapping validation was performed."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.run, args.model, args.baseline, args.repo)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))
