"""Evaluate the v3 tracker. Two modes, kept apart on purpose.

**Synthetic** inserts a ball with exact known geometry into a clip's real frames
and scores localisation against it. This is the only mode that can produce an
accuracy number, because it is the only mode with ground truth.

**Real** runs the tracker on the untouched footage and reports *behaviour*, not
accuracy: how often it accepts a track, how often it abstains, how often it locks
onto a body or onto static scenery. On the clips tagged ``scenery`` -- which hold
no playable ball -- behaviour becomes a measurement after all: **every accepted
track on those clips is a fabrication**, and that is the number the v2 pass most
needed and did not have.

The two are never averaged together. A synthetic recall and a real-footage
abstention rate are different claims about different things, and the v2 results
document is careful about this; so is this one.

The final-test split is opt-in and refuses to run without ``--final-test``, which
also records the exact configuration hash it ran under.

    .venv-video-ai/bin/python3 tools/ball_v3_eval.py --split validation
    .venv-video-ai/bin/python3 tools/ball_v3_eval.py --split final_test --final-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_synthetic as bsyn                # noqa: E402
from ghostcaddie.video import ball_v3_body as bbody                 # noqa: E402
from ghostcaddie.video import ball_v3_proposals as bp               # noqa: E402
from ghostcaddie.video import ball_v3_ranker as brank               # noqa: E402
from ghostcaddie.video import ball_v3_tracklets as btrk             # noqa: E402
from ghostcaddie.video import ball_v3_tracker as btrack             # noqa: E402

WORK = ROOT / "out" / "ball_tracker_v3"
V3 = ROOT / "data" / "ball_dataset_v3_eval"
BODY_CACHE = WORK / "body_cache"
EVAL_SEED = 20260918

#: Trajectories the synthetic evaluation exercises. Mixed deliberately: a tracker
#: that only handles a ball in flight cannot hold a tee'd ball at address, and
#: most of this corpus is at address.
TRAJECTORY_MIX = ("stationary", "slow", "linear", "fast", "accelerating",
                  "entrance", "exit")


def load_split(name: str, allow_final: bool) -> list[dict]:
    split = json.loads((V3 / "split.json").read_text("utf-8"))
    sources = {c["clip_id"]: c
               for c in json.loads((V3 / "sources.json").read_text("utf-8"))["clips"]}
    if name == "final_test" and not allow_final:
        raise SystemExit(
            "the final-test split is sealed: pass --final-test to consume it, and "
            "understand that it is scored once")
    out = []
    for clip in split["clips"]:
        if clip["split"] != name:
            continue
        record = sources[clip["clip_id"]]
        media = ROOT / record["media_path"]
        if media.exists():
            out.append({**clip, "media": media, "width": record["width"],
                        "height": record["height"],
                        "frames": record["frames"] or 0})
    return out


def read_window(video: Path, first: int, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def poses_for(clip_id: str, indices) -> dict:
    cache = bbody.BodyCache(BODY_CACHE)
    return {index: cache.pose(clip_id, index) for index in indices}


def score_synthetic(result, samples, indices) -> dict:
    """Localisation against exact synthetic geometry.

    A prediction on the wrong object is a false positive, not a near miss -- the
    v1 tracker was withdrawn for exactly that confusion, and the v2 evaluator was
    rebuilt around it.
    """
    truth = {}
    for position, sample in enumerate(samples):
        if sample.visibility == "VISIBLE" and getattr(sample, "verified", True):
            truth[indices[position]] = (sample.x, sample.y, sample.radius_px)

    states = {s.frame_index: s for s in result.states}
    true_positive = false_positive = false_negative = correct_silence = 0
    errors = []
    for index in indices:
        state = states.get(index)
        observed = state is not None and state.state == btrack.OBSERVED
        target = truth.get(index)
        if target is None:
            if observed:
                false_positive += 1
            else:
                correct_silence += 1
            continue
        if not observed:
            false_negative += 1
            continue
        distance = math.hypot(state.x - target[0], state.y - target[1])
        if distance <= max(6.0, target[2]):
            true_positive += 1
            errors.append(distance)
        else:
            false_positive += 1
    with_ball = len(truth)
    return {
        "frames": len(indices),
        "frames_with_ball": with_ball,
        "frames_without_ball": len(indices) - with_ball,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "correct_silence": correct_silence,
        "precision": round(true_positive / max(1, true_positive + false_positive), 4),
        # TP / (TP + FN + FP-on-a-ball-frame). One denominator, stated: v2 reported
        # precision and recall over different denominators, which made its F1 mix
        # two conventions.
        "recall": round(true_positive / max(1, with_ball), 4),
        "median_center_error_px": round(float(np.median(errors)), 2) if errors else None,
        "p90_center_error_px": round(float(np.percentile(errors, 90)), 2)
        if errors else None,
        "accepted": result.accepted,
    }


def run_clip(clip: dict, *, ranker, discriminator, params, proposal_params,
             length: int, trajectory: str, synthetic: bool, seed: int) -> dict:
    total = clip["frames"] or 0
    if total < length + 8:
        return {}
    first = max(2, (total - length) // 2)
    frames = read_window(clip["media"], first, length)
    if len(frames) < 3:
        return {}
    indices = list(range(first, first + len(frames)))
    samples = None
    if synthetic:
        frames, samples = bsyn.synthesize_sequence(
            frames, seed=seed, forced_kind=trajectory, wide_scale=True)
    started = time.time()
    result = btrack.track_ball_v3(
        frames, indices, ranker=ranker, discriminator=discriminator,
        poses=poses_for(clip["clip_id"], indices), params=params,
        proposal_params=proposal_params)
    elapsed = time.time() - started

    row = {
        "clip_id": clip["clip_id"],
        "content_class": clip.get("content_class"),
        "holds_playable_ball": clip.get("holds_playable_ball"),
        "frame_start": first,
        "accepted": result.accepted,
        "confidence": result.confidence,
        "rejection_reasons": result.rejection_reasons,
        "timings": {**result.timings,
                    "total_s": round(elapsed, 2),
                    "ms_per_frame": round(1000 * elapsed / max(1, len(frames)), 1)},
    }
    if synthetic:
        row["trajectory"] = trajectory
        row.update(score_synthetic(result, samples, indices))
    else:
        counts = result.confidence["state_counts"]
        row["observed_frames"] = counts.get(btrack.OBSERVED, 0)
        row["abstention_rate"] = result.confidence["abstention_rate"]
        # on a clip that holds no playable ball, an accepted track is fabricated
        row["fabricated_track"] = bool(
            result.accepted and clip.get("holds_playable_ball") is False)
    return row


def configuration_hash(params, proposal_params, ranker_path, tracklet_path) -> dict:
    """Exactly what was frozen, and its digest. Recorded before the final test."""
    payload = {
        "tracker_params": asdict(params),
        "proposal_params": asdict(proposal_params),
        "generators": list(bp.METHOD_NAMES),
        "ranker_sha256": brank.weights_hash(ranker_path) if ranker_path.exists() else None,
        "tracklet_sha256": (brank.weights_hash(tracklet_path)
                            if tracklet_path and tracklet_path.exists() else None),
        "context": {"grid": 64, "tight_radii": 8.0, "context_radii": 32.0,
                    "channels": 7},
        "eval_seed": EVAL_SEED,
        "trajectory_mix": list(TRAJECTORY_MIX),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"configuration": payload, "configuration_sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation",
                        choices=("validation", "final_test"))
    parser.add_argument("--final-test", action="store_true",
                        help="consume the sealed split; it is scored once")
    parser.add_argument("--ranker", type=Path, default=WORK / "ranker" / "ranker.pt")
    parser.add_argument("--tracklet", type=Path, default=WORK / "ranker" / "tracklet.pt")
    parser.add_argument("--no-tracklet-model", action="store_true",
                        help="ablation: run without the tracklet discriminator")
    parser.add_argument("--no-body", action="store_true",
                        help="ablation: run without the body pose cache")
    parser.add_argument("--length", type=int, default=9)
    parser.add_argument("--label", default=None, help="name for this run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.ranker.exists():
        print(f"ranker weights not found: {args.ranker}", file=sys.stderr)
        return 2
    clips = load_split(args.split, args.final_test)
    if not clips:
        print("no media on disk for this split", file=sys.stderr)
        return 2

    ranker = brank.load_ranker(args.ranker)
    discriminator = None
    if args.tracklet.exists() and not args.no_tracklet_model:
        discriminator = btrk.load_discriminator(args.tracklet,
                                                device_override=brank.device())
    if args.no_body:
        global BODY_CACHE
        BODY_CACHE = WORK / "body_cache_absent"

    params = btrack.TrackerParams()
    proposal_params = bp.ProposalParams()
    frozen = configuration_hash(params, proposal_params, args.ranker,
                                None if args.no_tracklet_model else args.tracklet)
    print(f"configuration {frozen['configuration_sha256'][:16]}", flush=True)

    synthetic_rows, real_rows = [], []
    for position, clip in enumerate(clips):
        trajectory = TRAJECTORY_MIX[position % len(TRAJECTORY_MIX)]
        row = run_clip(clip, ranker=ranker, discriminator=discriminator,
                       params=params, proposal_params=proposal_params,
                       length=args.length, trajectory=trajectory, synthetic=True,
                       seed=EVAL_SEED + position)
        if row:
            synthetic_rows.append(row)
            print(f"  synth {clip['clip_id']:20s} {trajectory:12s} "
                  f"TP={row['true_positives']:2d} FP={row['false_positives']:2d} "
                  f"FN={row['false_negatives']:2d} accepted={row['accepted']} "
                  f"{row['timings']['ms_per_frame']} ms/frame", flush=True)
        row = run_clip(clip, ranker=ranker, discriminator=discriminator,
                       params=params, proposal_params=proposal_params,
                       length=args.length, trajectory=trajectory, synthetic=False,
                       seed=EVAL_SEED + position)
        if row:
            real_rows.append(row)
            print(f"  real  {clip['clip_id']:20s} {str(clip.get('content_class')):16s} "
                  f"accepted={row['accepted']} observed={row['observed_frames']} "
                  f"abstain={row['abstention_rate']}"
                  f"{'  FABRICATED' if row['fabricated_track'] else ''}", flush=True)

    pooled = {key: sum(r[key] for r in synthetic_rows)
              for key in ("frames", "frames_with_ball", "frames_without_ball",
                          "true_positives", "false_positives", "false_negatives",
                          "correct_silence")}
    true_positive = pooled["true_positives"]
    pooled["precision"] = round(
        true_positive / max(1, true_positive + pooled["false_positives"]), 4)
    pooled["recall"] = round(true_positive / max(1, pooled["frames_with_ball"]), 4)
    pooled["f1"] = round(
        2 * pooled["precision"] * pooled["recall"]
        / max(1e-9, pooled["precision"] + pooled["recall"]), 4)
    errors = [r["median_center_error_px"] for r in synthetic_rows
              if r["median_center_error_px"] is not None]
    pooled["median_of_clip_median_center_error_px"] = (
        round(float(np.median(errors)), 2) if errors else None)

    ball_free = [r for r in real_rows if r["holds_playable_ball"] is False]
    behaviour = {
        "clips": len(real_rows),
        "accepted_track_rate": round(
            float(np.mean([r["accepted"] for r in real_rows])), 4) if real_rows else None,
        "mean_abstention_rate": round(
            float(np.mean([r["abstention_rate"] for r in real_rows])), 4)
        if real_rows else None,
        "ball_free_clips": len(ball_free),
        "fabricated_track_rate_on_ball_free_clips": round(
            float(np.mean([r["fabricated_track"] for r in ball_free])), 4)
        if ball_free else None,
        "mean_body_lock_rate": round(float(np.mean(
            [r["confidence"]["body_lock_rate"] for r in real_rows])), 4)
        if real_rows else None,
        "mean_static_fraction": round(float(np.mean(
            [r["confidence"]["static_fraction"] for r in real_rows
             if r["confidence"]["static_fraction"] is not None])), 4)
        if real_rows else None,
        "median_best_vs_second_margin": round(float(np.median(
            [r["confidence"]["median_margin"] for r in real_rows
             if r["confidence"]["median_margin"] is not None])), 4)
        if real_rows else None,
    }
    timings = [r["timings"] for r in synthetic_rows + real_rows]
    runtime = {
        "median_ms_per_frame": round(float(np.median(
            [t["ms_per_frame"] for t in timings])), 1) if timings else None,
        "median_proposals_s": round(float(np.median(
            [t["proposals_s"] for t in timings])), 3) if timings else None,
        "median_ranking_s": round(float(np.median(
            [t["ranking_s"] for t in timings])), 3) if timings else None,
        "median_association_s": round(float(np.median(
            [t["association_s"] for t in timings])), 3) if timings else None,
        "median_motion_s": round(float(np.median(
            [t["motion_s"] for t in timings])), 3) if timings else None,
    }

    label = args.label or ("final_test" if args.split == "final_test" else "validation")
    payload = {
        "split": args.split,
        "label": label,
        "scored_once": args.split == "final_test",
        "ablations": {"no_tracklet_model": args.no_tracklet_model,
                      "no_body": args.no_body},
        **frozen,
        "synthetic": {"note": "exact synthetic ground truth inserted into this "
                              "split's real backgrounds; the only mode here with "
                              "ground truth",
                      "pooled": pooled, "per_clip": synthetic_rows},
        "real_footage_behaviour": {
            "note": "no labels exist for these clips, so these are behavioural "
                    "diagnostics and not accuracy -- except on the ball-free "
                    "clips, where an accepted track is a fabrication",
            **behaviour, "per_clip": real_rows},
        "runtime": runtime,
    }
    out = args.out or (WORK / f"eval_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("\nsynthetic pooled:", json.dumps(pooled, indent=1))
    print("real behaviour:", json.dumps(behaviour, indent=1))
    print("runtime:", json.dumps(runtime, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
