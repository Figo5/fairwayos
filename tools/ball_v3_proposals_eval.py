"""Measure the ceiling: does the v3 proposal stage contain the ball at all?

Every later stage -- ranking, tracklets, abstention -- is bounded by this number,
so it is measured first and measured on its own. The v2 pipeline's equivalent
number is 31.0% (see ``tools/ball_v3_diagnose.py``), and no ranker can recover the
other 69%.

The ground truth is the legacy human-read corpus in ``data/ball_annotations/``.
That corpus is **development-consumed**: it was scored by an earlier pass, and it
is used here to choose proposal parameters. Nothing measured here is reported as
v3 accuracy, and the v3 evaluation corpus is not touched.

Usage:
    .venv-video-ai/bin/python3 tools/ball_v3_proposals_eval.py
    .venv-video-ai/bin/python3 tools/ball_v3_proposals_eval.py --sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_v3_proposals as bp          # noqa: E402

ANNOTATIONS = ROOT / "data" / "ball_annotations"
DEFAULT_OUT = ROOT / "out" / "ball_tracker_v3" / "proposal_recall.json"

#: (label, per_method_limit, work_max_side, max_per_frame)
SWEEP = [
    ("v2 baseline        ", 16, 1920, 10_000),
    ("limit 48           ", 48, 1920, 10_000),
    ("limit 96           ", 96, 1920, 10_000),
    ("limit 96  work 2880", 96, 2880, 10_000),
    ("limit 96  work 2880 budget 240", 96, 2880, 240),
    ("limit 160 work 2880", 160, 2880, 10_000),
]


def read_window(video: Path, first: int, last: int) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = {}
    for index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = frame
    capture.release()
    return frames


def load_clips(max_radius: float, frames_per_clip: int) -> list[dict]:
    clips = []
    for path in sorted(ANNOTATIONS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = ROOT / payload["video"]
        if not video.exists():
            continue
        labels = {int(k): v for k, v in payload["frames"].items()
                  if v.get("visible") and not v.get("ambiguous")
                  and (v.get("radius") or 5.0) <= max_radius}
        if not labels:
            continue
        wanted = sorted(labels)[:frames_per_clip]
        decoded = read_window(video, wanted[0], wanted[-1])
        indices = [i for i in range(wanted[0], wanted[-1] + 1) if i in decoded]
        if len(indices) < 2:
            continue
        clips.append({
            "clip_id": payload["clip_id"],
            "frames": [decoded[i] for i in indices],
            "indices": indices,
            "truth": {i: (float(labels[i]["x"]), float(labels[i]["y"]),
                          float(labels[i].get("radius") or 5.0))
                      for i in indices if i in labels},
        })
    return clips


#: The legacy labels carry their own noise floor, stated in data/ball_split.json:
#: "centre precision is roughly +/-15 px or 0.15x ball diameter, whichever is
#: larger". A proposal sitting on the ball can therefore be 15 px from the
#: recorded centre and still be correct -- observed directly on pexels_37080178,
#: where the label sits on the ball's upper edge. Recall is reported at both the
#: strict radius tolerance and this noise-aware one, and neither is presented
#: alone.
LABEL_NOISE_PX = 15.0


#: Method sets compared for the cost/recall trade. Per-generator cost on a 4K
#: frame at work_max_side 2880: local_contrast 4.35 s, brightness_cue 1.39 s,
#: radial_symmetry 1.30 s, local_distinctness 0.57 s, flow_residual 0.40 s,
#: frame_difference 0.05 s.
METHOD_SETS = {
    "all six": bp.METHOD_NAMES,
    "v2 five": tuple(m for m in bp.METHOD_NAMES if m != "local_distinctness"),
    "drop local_contrast": tuple(m for m in bp.METHOD_NAMES if m != "local_contrast"),
    "distinctness + brightness": ("local_distinctness", "brightness_cue"),
    "distinctness only": ("local_distinctness",),
}


def evaluate(clips: list[dict], params: bp.ProposalParams,
             methods=None) -> dict:
    found = missed = 0
    found_noise_aware = 0
    proposals_per_frame: list[int] = []
    positions: list[int] = []
    per_clip = {}
    started = time.time()
    for clip in clips:
        per_frame, info = bp.propose(clip["frames"], clip["indices"], params=params,
                                     methods=methods or bp.METHOD_NAMES)
        result = bp.proposal_recall(per_frame, clip["truth"])
        per_clip[clip["clip_id"]] = result
        found += result["found"]
        missed += result["missed"]
        proposals_per_frame.append(info["mean_proposals"])
        for index, (gx, gy, radius) in clip["truth"].items():
            tolerance = max(6.0, radius)
            distances = [math.hypot(proposal.x - gx, proposal.y - gy)
                         for proposal in (per_frame.get(index) or [])]
            hits = [p for p, d in enumerate(distances) if d <= tolerance]
            if hits:
                positions.append(min(hits) + 1)
            if any(d <= max(tolerance, LABEL_NOISE_PX) for d in distances):
                found_noise_aware += 1
    total = max(1, found + missed)
    return {
        "frames": found + missed,
        "recall": round(found / total, 4),
        "recall_within_label_noise": round(found_noise_aware / total, 4),
        "mean_proposals_per_frame": round(float(np.mean(proposals_per_frame)), 1),
        "median_position_of_the_ball": float(np.median(positions)) if positions else None,
        "seconds": round(time.time() - started, 1),
        "per_clip": {k: v["recall"] for k, v in per_clip.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="compare proposal settings instead of scoring defaults")
    parser.add_argument("--methods", action="store_true",
                        help="compare generator sets for cost against recall")
    parser.add_argument("--max-radius", type=float, default=40.0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    print("decoding labelled windows...", flush=True)
    clips = load_clips(args.max_radius, args.frames)
    print(f"{len(clips)} clips, "
          f"{sum(len(c['truth']) for c in clips)} labelled frames\n", flush=True)

    results = {}
    if args.methods:
        print(f"{'generator set':28s} {'recall':>7s} {'+noise':>7s} "
              f"{'cand/frame':>11s} {'sec/frame':>10s}", flush=True)
        for label, methods in METHOD_SETS.items():
            result = evaluate(clips, bp.ProposalParams(), methods=methods)
            results[label] = {"methods": list(methods), **result}
            frames = max(1, result["frames"])
            print(f"{label:28s} {result['recall']:7.3f} "
                  f"{result['recall_within_label_noise']:7.3f} "
                  f"{result['mean_proposals_per_frame']:11.1f} "
                  f"{result['seconds'] / frames:10.2f}", flush=True)
    elif args.sweep:
        print(f"{'config':32s} {'recall':>7s} {'cand/frame':>11s} "
              f"{'ball at #':>10s} {'sec':>6s}", flush=True)
        for label, limit, work, budget in SWEEP:
            params = bp.ProposalParams(per_method_limit=limit, work_max_side=work,
                                       max_per_frame=budget)
            result = evaluate(clips, params)
            results[label.strip()] = {"params": {"per_method_limit": limit,
                                                 "work_max_side": work,
                                                 "max_per_frame": budget},
                                      **result}
            print(f"{label:32s} {result['recall']:7.3f} "
                  f"{result['mean_proposals_per_frame']:11.1f} "
                  f"{str(result['median_position_of_the_ball']):>10s} "
                  f"{result['seconds']:6.1f}", flush=True)
    else:
        params = bp.ProposalParams()
        result = evaluate(clips, params)
        results["defaults"] = result
        print(json.dumps(result, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "origin": "HUMAN_READ_LEGACY_ANNOTATIONS",
        "status": "development-consumed; parameter selection only, never accuracy",
        "max_radius_px": args.max_radius,
        "results": results}, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
