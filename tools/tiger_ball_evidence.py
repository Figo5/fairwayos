"""Causal, seed-only pixel validation of archived classic ball proposals.

This research ablation does not rerun or feed back into the classic tracker. It
retains an archived position only if current native pixels support a compact
seed-compatible bright component. It cannot acquire a point the proposal tracker
never measured, establish physical ball identity, or learn from reference labels.
The entire clubhead block remains byte-for-byte unchanged.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ghostcaddie.demo import target_benchmark as bench

METHOD = "ball_evidence_template"
METHOD_REACQUIRED = "ball_evidence_template_reacquired"
POLICY = {"confirmation_frames": 2, "missing_before_lost": 3,
          "proposal_feedback": False, "backfill": False,
          "cut_is_terminal": True, "coordinate_refinement": False}


class ProposalGate:
    """Forward policy. A later confirmation never changes an earlier gap."""

    def __init__(self, seed_valid, max_jump_px, *, seed_frame=None):
        self.active = bool(seed_valid)
        self.seed_valid = bool(seed_valid)
        self.max_jump_px = float(max_jump_px)
        self.pending = None
        self.misses = 0
        self.previous_frame = seed_frame
        self.terminal = False

    def step(self, proposal, accepted, *, cut=False):
        frame = proposal["source_frame"]
        if self.previous_frame is not None and frame <= self.previous_frame:
            raise ValueError("Proposals must be in strictly increasing source order")
        consecutive = self.previous_frame is None or frame == self.previous_frame + 1
        self.previous_frame = frame
        if cut:
            self.terminal = True
        if not consecutive:
            self.active = False
            self.pending = None
        measured = (accepted and self.seed_valid and not self.terminal
                    and proposal["origin"] == "automatic"
                    and proposal["status"] in bench.AUTOMATIC_STATUSES)
        reacquired = False
        if measured and not self.active:
            xy = (proposal["x"], proposal["y"])
            if (self.pending is not None and consecutive
                    and math.dist(xy, self.pending) <= self.max_jump_px):
                self.active = True
                reacquired = True
            else:
                self.pending = xy
                measured = False
        elif not measured:
            self.pending = None
            self.active = False
        if measured:
            self.misses = 0
            self.pending = None
            return {**proposal, "method": METHOD_REACQUIRED if reacquired else METHOD,
                    "score": None}
        self.misses += 1
        return {"source_frame": frame, "x": None, "y": None,
                "status": "lost" if self.misses >= POLICY["missing_before_lost"] else "missing",
                "origin": "derived", "method": "", "score": None}


def filter_clip(source, block, *, build_seed, check_evidence):
    """Algorithm input is label-free: source path/geometry, seed, cuts, proposals."""
    import cv2
    seed = block["seed"]
    if seed is None:
        return copy.deepcopy(block), {"seed_valid": False, "reason": "no_seed", "raw": []}
    start, end = source["window"]["start_frame"], source["window"]["end_frame"]
    if not start <= seed["frame"] <= end:
        raise ValueError("Seed must lie inside the source window")
    if seed["frame"] in source["cut_frames"]:
        raise ValueError("Seed cannot lie on a structural cut")
    points = block["points"]
    if len({p["source_frame"] for p in points}) != len(points):
        raise ValueError("Duplicate proposal frames")
    if any(not start <= p["source_frame"] <= end or p["source_frame"] in source["cut_frames"]
           for p in points):
        raise ValueError("Proposal outside window or on a structural cut")
    if any(p["source_frame"] < seed["frame"] and
           (p["status"] not in ("missing", "lost") or p["x"] is not None or p["y"] is not None)
           for p in points):
        raise ValueError("Pre-seed proposals must be missing or lost")
    if any(p["origin"] == "operator" and p["source_frame"] != seed["frame"]
           for p in block["points"]):
        raise ValueError("Only the original seed may be an operator point")
    if any(p["origin"] == "derived" and p["x"] is not None for p in block["points"]):
        raise ValueError("Interpolated proposals are not eligible for pixel validation")
    cap = cv2.VideoCapture(source["path"])
    result = copy.deepcopy(block)
    proposals = {p["source_frame"]: p for p in block["points"]}
    output = []
    raw = []
    descriptor = None
    gate = None
    try:
        if not cap.isOpened():
            raise ValueError("Source cannot be decoded")
        # Sequential decode preserves source indices and consumes no future pixels.
        for frame in range(end + 1):
            ok, image = cap.read()
            if not ok or image.shape[:2] != (source["height"], source["width"]):
                raise ValueError("Native frame missing or geometry changed")
            if frame == seed["frame"]:
                descriptor = build_seed(image, seed["x"], seed["y"])
                gate = ProposalGate(descriptor["valid"],
                                    (block.get("options") or {}).get("max_jump_px", 60.0),
                                    seed_frame=seed["frame"])
            if gate is not None and frame in source["cut_frames"]:
                gate.terminal = True
            if frame not in proposals:
                continue
            proposal = proposals[frame]
            if frame <= seed["frame"]:
                output.append(copy.deepcopy(proposal))
                continue
            reason = ("invalid_original_seed" if not descriptor["valid"] else
                      "terminal_structural_cut" if gate.terminal else "no_automatic_proposal")
            evidence = {"accepted": False, "reason": reason}
            if (proposal["origin"] == "automatic" and descriptor["valid"]
                    and not gate.terminal):
                evidence = check_evidence(image, proposal["x"], proposal["y"], descriptor)
            point = gate.step(proposal, evidence["accepted"])
            output.append(point)
            raw.append({"source_frame": frame, "proposal": proposal,
                        "evidence": evidence, "point": point})
    finally:
        cap.release()
    result["points"] = output
    return result, {"seed_valid": descriptor["valid"], "descriptor": descriptor, "raw": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "baseline", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True,
                        help="Executed report checksum-bound to the archived proposals")
    args = parser.parse_args()
    from ghostcaddie.demo import ball_evidence as engine
    import cv2
    import numpy as np
    dataset, sha = bench.load_dataset(args.dataset, verify_hashes=True)
    if not args.out.name.endswith(".predictions.json"):
        raise ValueError("Output must end with .predictions.json")
    raw_path = args.out.with_name(args.out.name.replace(".predictions.json", ".raw.json"))
    writes = [args.out, raw_path, Path(str(args.out) + ".report.json"),
              Path(str(args.out) + ".report.md")]
    protected = {args.dataset.resolve(), args.baseline.resolve(), args.baseline_report.resolve()} | {
        Path(c["path"]).resolve() for c in dataset["clips"]}
    if any(p.resolve() in protected or p.exists() for p in writes):
        raise ValueError("Choose fresh outputs that preserve inputs")
    baseline = json.loads(args.baseline.read_text())
    bench.validate_predictions(baseline, dataset)
    baseline_report = json.loads(args.baseline_report.read_text())
    if baseline_report.get("predictions", {}).get("sha256") != bench.sha256_file(args.baseline):
        raise ValueError("Baseline report does not bind these prediction bytes")
    payload = copy.deepcopy(baseline)
    payload["variant"] = "compact_ball_evidence_research"
    source_index = {c["id"]: c for c in dataset["clips"]}
    measured = {}
    for clip in payload["clips"]:
        source = source_index[clip["id"]]
        # No later reference coordinates or reference state enters filter_clip.
        inputs = {k: source[k] for k in ("path", "width", "height", "window", "cut_frames")}
        clip["targets"]["ball"], measured[clip["id"]] = filter_clip(
            inputs, clip["targets"]["ball"], build_seed=engine.build_seed_descriptor,
            check_evidence=engine.check_ball_evidence)
    validated = bench.validate_predictions(payload, dataset)
    for old, new in zip(baseline["clips"], payload["clips"]):
        if bench.canonical_json(old["targets"]["clubhead"]) != bench.canonical_json(new["targets"]["clubhead"]):
            raise ValueError("Clubhead block changed")
    raw_text = bench.canonical_json({"profile": engine.PROFILE, "policy": POLICY, "clips": measured})
    prediction_text = bench.canonical_json(payload)
    prediction_sha = hashlib.sha256(prediction_text.encode("utf-8")).hexdigest()
    runtime = {"research_only": True, "production_eligible": False,
               "model_rerun": False, "target_coordinate_or_visibility_labels_read_by_policy": False,
               "operator_annotations_used": ["original_seed", "window", "cut_frames"],
               "physical_identity_established": False, "head_blocks_unchanged": True,
               "proposal_feedback": False, "profile": engine.PROFILE, "policy": POLICY,
               "baseline_path": str(args.baseline.resolve()),
               "baseline_sha256": bench.sha256_file(args.baseline),
               "baseline_report_path": str(args.baseline_report.resolve()),
               "baseline_report_sha256": bench.sha256_file(args.baseline_report),
               "baseline_executed_runtime": baseline_report["runtime"],
               "benchmark_sha256": bench.sha256_file(Path(bench.__file__)),
               "engine_sha256": bench.sha256_file(Path(engine.__file__)),
               "tool_sha256": bench.sha256_file(Path(__file__)),
               "raw_path": str(raw_path), "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
               "opencv": cv2.__version__, "numpy": np.__version__,
               "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "repository_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
               "note": "Previously inspected development footage; not independent holdout. "
               "Current pixel evidence filters archived classic proposals without feedback; "
               "two consecutive confirmations after refusal, no backfill. "
               "Cannot recover points that classic never proposed. Null score is not confidence."}
    report = bench.evaluate(dataset, validated, dataset_path=str(args.dataset), dataset_sha256=sha,
                            predictions_path=str(args.out), predictions_sha256=prediction_sha,
                            runtime=runtime)
    texts = [prediction_text, raw_text, bench.canonical_json(report), bench.render_markdown(report)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in zip(writes, texts):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(contents)


if __name__ == "__main__":
    main()
