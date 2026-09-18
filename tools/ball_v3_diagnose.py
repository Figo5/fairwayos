"""Where does the v2 ball pipeline actually lose the ball?

This is the Phase 0 takeover measurement for the v3 pass. It exists because the
v2 handoff names *one* failure -- "the 64x64 crop detector saturates, so it
cannot rank the true ball above ball-like clutter" -- and that turned out to be
the smallest of three losses, not the dominant one.

The measurement runs the real v2 pipeline stage by stage against the only real
ball ground truth in this repository (``data/ball_annotations/``, 325 frames
read by eye in an earlier pass) and attributes every lost frame to the stage
that lost it:

1. ``raw_never_proposed``   -- no candidate generator fired within one ball
   radius of the labelled ball. Nothing downstream can recover this frame.
2. ``killed_by_min_score``  -- a generator *did* propose the ball, and
   ``ConsensusParams.min_score`` deleted it before clustering.
3. ``absorbed_into_other_cluster`` -- the proposal survived the floor but greedy
   clustering merged it into a higher-scoring cluster centred somewhere else.
4. ``survives_to_the_ranker`` -- a cluster sits on the ball, so the detector
   finally gets a chance to rank it.

Only then is the ranking question asked, and it is asked two ways: the rank the
detector gives the surviving cluster, and an *oracle* crop placed exactly on the
labelled ball, which is the fairest possible test of the saturation claim.

Labels are read from the existing annotation files; nothing here writes a label
and nothing here is human ground truth beyond what those files already are.

Usage:
    .venv-video-ai/bin/python3 tools/ball_v3_diagnose.py
    .venv-video-ai/bin/python3 tools/ball_v3_diagnose.py --max-radius 40 --frames 12
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_candidates as bc          # noqa: E402
from ghostcaddie.video import ball_consensus as bcon         # noqa: E402
from ghostcaddie.video import ball_detector as bdet          # noqa: E402
from ghostcaddie.video import ball_tracker_v2 as btr         # noqa: E402

ANNOTATIONS = ROOT / "data" / "ball_annotations"
DEFAULT_WEIGHTS = ROOT / "out" / "ball_tracker_v2" / "detector" / "detector_state.pt"
DEFAULT_OUT = ROOT / "out" / "ball_tracker_v3" / "v2_loss_attribution.json"


def read_window(video: Path, first: int, last: int) -> dict[int, np.ndarray]:
    """Decode [first, last] once. Seeking per frame on 4K costs more than reading."""
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: dict[int, np.ndarray] = {}
    for index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = frame
    capture.release()
    return frames


def visible_labels(payload: dict) -> dict[int, dict]:
    """Only unambiguous visible labels can score a localisation claim."""
    return {int(key): value for key, value in payload["frames"].items()
            if value.get("visible") and not value.get("ambiguous")}


def presence_of(model, crops, device) -> np.ndarray:
    if not crops:
        return np.zeros(0, dtype=np.float32)
    import torch
    with torch.no_grad():
        batch = np.stack(crops).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        _, presence_logits, _ = model(torch.from_numpy(batch).to(device))
        return torch.sigmoid(presence_logits).squeeze(1).cpu().numpy()


def diagnose(*, weights: Path, max_radius: float, frames_per_clip: int,
             annotations: Path) -> dict:
    device = bdet.device()
    model = bdet.load_detector(weights, device_override=device)
    params = bcon.ConsensusParams()

    stages = collections.Counter()
    rows: list[dict] = []

    for path in sorted(annotations.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = ROOT / payload["video"]
        if not video.exists():
            continue
        labels = visible_labels(payload)
        if not labels:
            continue
        wanted = sorted(labels)[:frames_per_clip]
        decoded = read_window(video, wanted[0], wanted[-1])
        indices = [i for i in range(wanted[0], wanted[-1] + 1) if i in decoded]
        sequence = [decoded[i] for i in indices]
        if len(sequence) < 2:
            continue
        contexts, _info = bc.build_contexts(sequence, indices)
        per_frame = bc.ensemble_candidates(contexts)

        for index, frame in zip(indices, sequence):
            label = labels.get(index)
            if label is None:
                continue
            gx, gy = float(label["x"]), float(label["y"])
            radius = float(label.get("radius") or 5.0)
            in_regime = radius <= max_radius
            # "found" means within one ball radius. A prediction further away than
            # the ball is wide is a prediction on a different object.
            tolerance = max(6.0, radius)
            raw = per_frame.get(index, [])
            on_ball = [c for c in raw
                       if math.hypot(c.x - gx, c.y - gy) <= tolerance]

            row = {"clip_id": payload["clip_id"], "frame": index, "radius": radius,
                   "in_regime": in_regime, "raw_candidates": len(raw),
                   "raw_proposed": bool(on_ball)}

            clusters = bcon.cluster_candidates(raw, params=params)
            row["clusters"] = len(clusters)
            on_cluster = [c for c in clusters
                          if math.hypot(c.x - gx, c.y - gy) <= tolerance]

            if not on_ball:
                stage = "1_raw_never_proposed"
            elif not [c for c in on_ball if c.score >= params.min_score]:
                stage = "2_killed_by_min_score"
                row["best_ball_candidate_score"] = round(
                    max(c.score for c in on_ball), 4)
            elif not on_cluster:
                stage = "3_absorbed_into_other_cluster"
            else:
                stage = "4_survives_to_the_ranker"
            row["lost_at"] = stage
            if in_regime:
                stages[stage] += 1

            # ranking, measured only where a cluster actually reached the detector
            crops, kept = [], []
            for cluster in clusters:
                crop, _factor = btr._crop_for_detector(
                    frame, cluster.x, cluster.y, cluster.radius)
                if crop is not None:
                    crops.append(crop)
                    kept.append(cluster)
            scores = presence_of(model, crops, device)
            hit_positions = [k for k, c in enumerate(kept)
                             if math.hypot(c.x - gx, c.y - gy) <= tolerance]
            if len(scores) and hit_positions:
                order = np.argsort(-scores)
                best_hit = max(hit_positions, key=lambda k: scores[k])
                row["detector_rank"] = int(np.where(order == best_hit)[0][0]) + 1

            # the oracle: a crop centred exactly on the labelled ball. If this
            # cannot beat the best distractor, saturation is real independently
            # of how good the candidate generator is.
            oracle_crop, _factor = btr._crop_for_detector(frame, gx, gy, radius)
            if oracle_crop is not None:
                oracle = float(presence_of(model, [oracle_crop], device)[0])
                row["oracle_presence"] = round(oracle, 4)
                distractors = [scores[k] for k in range(len(kept))
                               if k not in hit_positions]
                if distractors:
                    row["oracle_margin"] = round(oracle - float(max(distractors)), 4)
            rows.append(row)

    regime = [r for r in rows if r["in_regime"]]
    total = max(1, len(regime))
    ranked = [r for r in regime if r.get("detector_rank")]
    margins = [r["oracle_margin"] for r in regime if "oracle_margin" in r]
    summary = {
        "frames_with_a_visible_labelled_ball": len(regime),
        "max_radius_px": max_radius,
        "stage_losses": dict(sorted(stages.items())),
        "raw_ensemble_proposal_rate": round(
            sum(1 for r in regime if r["raw_proposed"]) / total, 4),
        "survives_to_the_ranker_rate": round(
            stages["4_survives_to_the_ranker"] / total, 4),
        "detector_ranks_it_first_when_it_arrives": round(
            float(np.mean([r["detector_rank"] == 1 for r in ranked])), 4)
        if ranked else None,
        "oracle_beats_best_distractor": round(
            float(np.mean([m > 0 for m in margins])), 4) if margins else None,
        "oracle_median_margin": round(float(np.median(margins)), 4) if margins else None,
    }
    out_of_regime = [r for r in rows if not r["in_regime"]]
    if out_of_regime:
        oom = [r["oracle_margin"] for r in out_of_regime if "oracle_margin" in r]
        summary["out_of_regime"] = {
            "frames": len(out_of_regime),
            "oracle_beats_best_distractor": round(
                float(np.mean([m > 0 for m in oom])), 4) if oom else None,
            "oracle_median_margin": round(float(np.median(oom)), 4) if oom else None,
            "note": ("balls larger than the detector's training regime; a scale "
                     "failure, not a discrimination failure"),
        }
    return {"summary": summary, "per_frame": rows,
            "weights": str(weights), "origin": "HUMAN_READ_LEGACY_ANNOTATIONS"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--max-radius", type=float, default=40.0,
                        help="largest ball radius the v2 detector was trained on")
    parser.add_argument("--frames", type=int, default=12,
                        help="labelled frames per clip")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.weights.exists():
        print(f"detector weights not found: {args.weights}", file=sys.stderr)
        return 2
    result = diagnose(weights=args.weights, max_radius=args.max_radius,
                      frames_per_clip=args.frames, annotations=args.annotations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
