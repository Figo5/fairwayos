"""Generate listwise training data and train the v3 ranker.

Subcommands:

``gen``            build listwise examples from TRAIN clips: insert a synthetic
                   ball through a real window, run the real proposal stage on the
                   modified frames, and record which candidate is the ball among
                   the genuine distractors from that same photograph.
``train-ranker``   listwise softmax over [candidates..., null].
``eval-ranker``    score the ranker on held-out synthetic lists and, separately,
                   on the legacy real labelled frames.

Splits are enforced, not assumed: ``gen`` refuses any clip that is not in the v2
``train`` split, and nothing in this file can read the v3 evaluation corpus.

Everything here needs torch: use ``.venv-video-ai/bin/python3``. The body pose
cache is built separately by ``tools/ball_v3_body_cache.py`` under ``.venv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_dataset as bds                # noqa: E402
from ghostcaddie.video import ball_synthetic as bsyn             # noqa: E402
from ghostcaddie.video import ball_v3_body as bbody              # noqa: E402
from ghostcaddie.video import ball_v3_dataset as bdata           # noqa: E402
from ghostcaddie.video import ball_v3_proposals as bp            # noqa: E402
from ghostcaddie.video import ball_v3_ranker as brank            # noqa: E402
from ghostcaddie.video import ball_v3_tracklets as btrk          # noqa: E402

WORK = ROOT / "out" / "ball_tracker_v3"
BODY_CACHE = WORK / "body_cache"
V2_SOURCES = ROOT / "data" / "ball_dataset_v2" / "sources.json"
V2_SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"
SEED = 20260918

#: Trajectory kinds exercised, from the brief's list. A tracker that only ever
#: sees a ball in flight cannot hold a tee'd ball at address, which is the state
#: most of this corpus is actually in.
TRAJECTORIES = ("stationary", "stationary", "slow", "linear", "fast",
                "accelerating", "entrance", "exit")


def train_clips() -> list[dict]:
    sources = json.loads(V2_SOURCES.read_text("utf-8"))
    split = json.loads(V2_SPLIT.read_text("utf-8"))
    by_id = {c["clip_id"]: c for c in sources["clips"]}
    wanted = [c["clip_id"] for c in split["clips"] if c["split"] == "train"]
    out = []
    for clip_id in wanted:
        record = by_id.get(clip_id)
        if not record:
            continue
        media = ROOT / record["media_path"]
        if media.exists():
            out.append({"clip_id": clip_id, "media": media,
                        "frames": record.get("frames") or 0})
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


def cmd_gen(args) -> int:
    clips = train_clips()
    if not clips:
        print("no training media on disk", file=sys.stderr)
        return 2
    # the split guard: this must raise rather than quietly train on evaluation data
    split = bds.load_split(V2_SPLIT)
    bsyn.assert_training_clips(split, [c["clip_id"] for c in clips])

    cache = bbody.BodyCache(BODY_CACHE)
    params = bp.ProposalParams(
        per_method_limit=args.per_method_limit,
        work_max_side=args.work_max_side,
        max_per_frame=args.max_per_frame)
    methods = tuple(args.methods.split(",")) if args.methods else bp.METHOD_NAMES

    out_dir = args.out
    writer = bdata.ExampleWriter(out_dir)
    rng = np.random.default_rng(args.seed)
    picker = random.Random(args.seed)
    started = time.time()
    nulls = 0

    for round_index in range(args.sequences_per_clip):
        for clip in clips:
            total = clip["frames"] or 0
            if total < args.length + 8:
                continue
            first = picker.randrange(2, max(3, total - args.length - 2))
            frames = read_window(clip["media"], first, args.length)
            if len(frames) < args.length:
                continue
            seed = args.seed + 1000 * round_index + \
                int(hashlib.sha256(clip["clip_id"].encode()).hexdigest()[:8], 16) % 997
            kind = picker.choice(TRAJECTORIES)
            modified, samples = bsyn.synthesize_sequence(
                frames, seed=seed, forced_kind=kind, wide_scale=True)

            def pose_for(frame_index: int, _clip=clip["clip_id"]):
                return cache.pose(_clip, frame_index)

            indices = list(range(first, first + len(modified)))
            # the proposal stage runs on the MODIFIED frames, so the candidate list
            # holds the synthetic ball and the real clutter from the same footage
            for example in bdata.examples_from_sequence(
                    modified, samples, clip_id=clip["clip_id"],
                    frame_indices=indices, params=params, methods=methods,
                    pose_for=pose_for, numeric_fn=brank.numeric_features,
                    rng=rng, list_size=args.list_size):
                if example.target is None and example.truth is None:
                    nulls += 1
                writer.add(example)
            elapsed = time.time() - started
            print(f"[round {round_index + 1}/{args.sequences_per_clip}] "
                  f"{clip['clip_id']:20s} {kind:12s} lists={writer.written} "
                  f"misses={writer.missed} nulls={nulls} "
                  f"{elapsed / 60:.1f} min", flush=True)
            if args.limit and writer.written >= args.limit:
                break
        if args.limit and writer.written >= args.limit:
            break

    summary = writer.close()
    total_ball_frames = summary["lists"] - nulls + summary["proposal_misses"]
    summary.update({
        "nulls": nulls,
        "proposal_recall_on_synthetic": round(
            (summary["lists"] - nulls) / max(1, total_ball_frames), 4),
        "params": {"per_method_limit": params.per_method_limit,
                   "work_max_side": params.work_max_side,
                   "max_per_frame": params.max_per_frame,
                   "list_size": args.list_size,
                   "methods": list(methods)},
        "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds",
        "clips": len(clips),
        "minutes": round((time.time() - started) / 60, 1),
    })
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def cmd_train_ranker(args) -> int:
    import torch

    reader = bdata.ExampleReader(args.data)
    if not len(reader):
        print("no training lists found; run gen first", file=sys.stderr)
        return 2

    # split by CLIP, not by list: two lists from one clip share a background, and
    # splitting by list would let the same turf appear on both sides
    clips = sorted({r["clip_id"] for r in reader.records})
    rng = random.Random(args.seed)
    rng.shuffle(clips)
    holdout = set(clips[:max(1, len(clips) // 5)])
    train = [r for r in reader.records if r["clip_id"] not in holdout]
    validation = [r for r in reader.records if r["clip_id"] in holdout]
    print(f"{len(train)} training lists, {len(validation)} validation lists "
          f"({len(holdout)} held-out clips)")

    device = brank.device()
    model = brank.CandidateRanker().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_top1, best_state, best_epoch = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        total = 0.0
        optimiser.zero_grad()
        for position, record in enumerate(train):
            stacks = reader.stacks_for(record)
            numeric = reader.numeric_for(record)
            images = torch.from_numpy(
                stacks.transpose(0, 3, 1, 2).astype(np.float32) / 255.0).to(device)
            numbers = torch.from_numpy(numeric).to(device)
            scores = model(images, numbers)
            loss = brank.listwise_loss(scores, record["target"], model.null_logit)
            (loss / args.accumulate).backward()
            total += float(loss.item())
            if (position + 1) % args.accumulate == 0:
                optimiser.step()
                optimiser.zero_grad()
        optimiser.step()
        optimiser.zero_grad()

        metrics = evaluate_lists(model, reader, validation, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = round(total / max(1, len(train)), 4)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["top1"] >= best_top1:
            best_top1, best_epoch = metrics["top1"], epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = brank.save_ranker(model, args.out)
    report = {
        "architecture": "CandidateRanker: 7-channel two-scale stack + 19 numeric "
                        "features -> listwise softmax with a learned null slot",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "seed": args.seed,
        "train_lists": len(train),
        "validation_lists": len(validation),
        "held_out_clips": sorted(holdout),
        "weights": args.out.name,
        "weights_sha256": digest,
        "history": history,
        "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds",
    }
    (args.out.parent / "ranker_report.json").write_text(
        json.dumps(report, indent=1) + "\n")
    print(f"best epoch {best_epoch} top1 {best_top1:.4f}; wrote {args.out}")
    return 0


def evaluate_lists(model, reader, records, device) -> dict:
    """Top-1, abstention behaviour and margin on whole candidate lists."""
    import torch

    model.eval()
    correct = total = 0
    null_correct = null_total = 0
    margins, null_probabilities = [], []
    with torch.no_grad():
        for record in records:
            stacks = reader.stacks_for(record)
            numeric = reader.numeric_for(record)
            result = brank.rank_frame(model, stacks, numeric, device=device)
            target = record["target"]
            if target is None:
                null_total += 1
                if result["null_probability"] >= result["best_probability"]:
                    null_correct += 1
                null_probabilities.append(result["null_probability"])
                continue
            total += 1
            if result["best"] == target:
                correct += 1
                margins.append(result["margin"])
    return {
        "lists_with_a_ball": total,
        "top1": round(correct / max(1, total), 4),
        "null_lists": null_total,
        "correct_abstention": round(null_correct / max(1, null_total), 4)
        if null_total else None,
        "median_margin_when_right": round(float(np.median(margins)), 4)
        if margins else None,
        "median_null_probability_on_empty": round(
            float(np.median(null_probabilities)), 4) if null_probabilities else None,
    }


def positions_from(record: dict) -> list[tuple[float, float, float]]:
    """Recover native-pixel geometry from the stored numeric features.

    ``ball_v3_context.geometry_features`` writes x/width, y/height and
    radius/short_side as its first three entries, so the inverse is exact up to
    float rounding and nothing has to be stored twice.
    """
    width, height = record["width"], record["height"]
    short_side = max(1, min(width, height))
    out = []
    for row in record["numeric"]:
        out.append((float(row[0]) * width, float(row[1]) * height,
                    max(1.0, float(row[2]) * short_side)))
    return out


def camera_shifts_for(clip_media: dict, needed: dict[str, set[int]]) -> dict:
    """Phase-correlate only the frames the stored lists actually use.

    Far cheaper than re-running the proposal stage: this is two decoded frames and
    one FFT per index, and the motion features are meaningless without it.
    """
    shifts: dict[str, dict[int, tuple[float, float]]] = {}
    for clip_id, indices in needed.items():
        media = clip_media.get(clip_id)
        if media is None:
            continue
        per_clip: dict[int, tuple[float, float]] = {}
        capture = cv2.VideoCapture(str(media))
        for index in sorted(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index - 1))
            ok_previous, previous = capture.read()
            ok_current, current = capture.read()
            if not (ok_previous and ok_current):
                per_clip[index] = (0.0, 0.0)
                continue
            a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
            window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), _response = cv2.phaseCorrelate(a, b, window)
            per_clip[index] = (float(dx), float(dy))
        capture.release()
        shifts[clip_id] = per_clip
        print(f"  shifts {clip_id}: {len(per_clip)} frames", flush=True)
    return shifts


def cmd_gen_tracklets(args) -> int:
    """Build labelled tracklets from the stored listwise data and a trained ranker.

    A tracklet is positive when it sits on the true ball for most of the frames it
    holds. Everything else is a negative, and the negatives are the point: they
    are real distractors that the ranker already scored highly enough to be in the
    top-K, linked over time from the same footage.
    """
    import torch

    reader = bdata.ExampleReader(args.data)
    ranker = brank.load_ranker(args.ranker)
    device = brank.device()

    by_clip: dict[str, list[dict]] = {}
    for record in reader.records:
        by_clip.setdefault(record["clip_id"], []).append(record)
    media = {c["clip_id"]: c["media"] for c in train_clips()}
    needed = {clip: {r["frame_index"] for r in records}
              for clip, records in by_clip.items()}
    print("computing camera shifts", flush=True)
    shifts = camera_shifts_for(media, needed)

    rows = []
    for clip_id, records in sorted(by_clip.items()):
        records.sort(key=lambda r: r["frame_index"])
        # group into runs of consecutive frames: a tracklet cannot span a cut
        runs: list[list[dict]] = []
        for record in records:
            if runs and record["frame_index"] == runs[-1][-1]["frame_index"] + 1:
                runs[-1].append(record)
            else:
                runs.append([record])
        clip_shifts = shifts.get(clip_id, {})
        for run in runs:
            if len(run) < 3:
                continue
            nodes_by_frame: dict[int, list[btrk.Node]] = {}
            truth: dict[int, tuple] = {}
            for record in run:
                stacks = reader.stacks_for(record)
                numeric = reader.numeric_for(record)
                result = brank.rank_frame(ranker, stacks, numeric, device=device)
                geometry = positions_from(record)
                order = np.argsort(-result["probabilities"])
                nodes = []
                for rank, slot in enumerate(order[:args.top_k]):
                    x, y, radius = geometry[int(slot)]
                    row = record["numeric"][int(slot)]
                    nodes.append(btrk.Node(
                        frame_index=record["frame_index"], x=x, y=y, radius=radius,
                        probability=float(result["probabilities"][int(slot)]),
                        margin=result["margin"] if rank == 0 else 0.0,
                        is_top1=rank == 0,
                        inside_torso=float(row[4 + 1]),
                        distance_to_hands=float(row[4 + 3]) * 200.0))
                nodes_by_frame[record["frame_index"]] = nodes
                if record["truth"]:
                    truth[record["frame_index"]] = tuple(record["truth"])
            tracks = btrk.build_tracklets(nodes_by_frame, clip_shifts,
                                          top_k=args.top_k, min_length=3)
            for track in tracks:
                hits = 0
                scored = 0
                for node in track.nodes:
                    target = truth.get(node.frame_index)
                    if target is None:
                        continue
                    scored += 1
                    if math.hypot(node.x - target[0],
                                  node.y - target[1]) <= max(6.0, target[2]):
                        hits += 1
                if scored == 0:
                    continue
                label = 1 if hits / scored >= args.positive_fraction else 0
                rows.append({
                    "clip_id": clip_id,
                    "first_frame": track.nodes[0].frame_index,
                    "label": label,
                    "on_ball_fraction": round(hits / scored, 3),
                    "features": btrk.tracklet_features(
                        track, window=len(run), camera_shift=clip_shifts),
                })
        print(f"{clip_id}: {len(rows)} tracklets so far", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    positives = sum(r["label"] for r in rows)
    payload = {"tracklets": rows, "positives": positives,
               "negatives": len(rows) - positives,
               "feature_names": list(btrk.FEATURE_NAMES),
               "ranker": str(args.ranker),
               "positive_fraction": args.positive_fraction,
               "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds"}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{len(rows)} tracklets, {positives} positive; wrote {args.out}")
    return 0


def cmd_train_tracklet(args) -> int:
    import torch

    payload = json.loads(args.data.read_text("utf-8"))
    rows = payload["tracklets"]
    if not rows:
        print("no tracklets; run gen-tracklets first", file=sys.stderr)
        return 2

    clips = sorted({r["clip_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(clips)
    holdout = set(clips[:max(1, len(clips) // 5)])
    train = [r for r in rows if r["clip_id"] not in holdout]
    validation = [r for r in rows if r["clip_id"] in holdout]

    mask = btrk.mask_for_ablation(args.drop.split(",") if args.drop else [])
    device = brank.device()

    def tensors(records):
        features = np.asarray([r["features"] for r in records], dtype=np.float32) * mask
        labels = np.asarray([r["label"] for r in records], dtype=np.float32)
        return (torch.from_numpy(features).to(device),
                torch.from_numpy(labels).to(device))

    train_x, train_y = tensors(train)
    validation_x, validation_y = tensors(validation)
    model = btrk.TrackletDiscriminator().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # the classes are lopsided -- most tracklets are clutter -- so the positive
    # term is weighted rather than letting the model win by always saying no
    positive = max(1.0, float(train_y.sum()))
    weight = torch.tensor(
        (len(train_y) - positive) / positive, device=device).clamp(1.0, 20.0)
    history = []
    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(train_x), train_y, pos_weight=weight)
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(validation_x)).cpu().numpy()
        auc = _auc(scores, validation_y.cpu().numpy())
        history.append({"epoch": epoch + 1, "loss": round(float(loss), 4),
                        "validation_auc": None if auc is None else round(auc, 4)})
        if auc is not None and auc >= best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    digest = btrk.save_discriminator(model, args.out)
    report = {"parameters": int(sum(p.numel() for p in model.parameters())),
              "train": len(train), "validation": len(validation),
              "positives": sum(r["label"] for r in rows),
              "best_validation_auc": best_auc,
              "dropped_features": args.drop or None,
              "weights_sha256": digest, "history": history,
              "held_out_clips": sorted(holdout)}
    (args.out.parent / f"tracklet_report{('_' + args.drop.replace(',', '_')) if args.drop else ''}.json"
     ).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1))
    return 0


def _auc(scores: np.ndarray, labels: np.ndarray):
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    order = np.argsort(np.r_[positives, negatives])
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    n1, n0 = len(positives), len(negatives)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="build listwise training examples")
    gen.add_argument("--out", type=Path, default=WORK / "listwise_train")
    gen.add_argument("--sequences-per-clip", type=int, default=4)
    gen.add_argument("--length", type=int, default=9,
                     help="frames per synthetic sequence")
    gen.add_argument("--list-size", type=int, default=bdata.LIST_SIZE)
    gen.add_argument("--per-method-limit", type=int, default=96)
    gen.add_argument("--work-max-side", type=int, default=2880)
    gen.add_argument("--max-per-frame", type=int, default=240)
    gen.add_argument("--methods", default=None,
                     help="comma-separated generator names")
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--seed", type=int, default=SEED)
    gen.set_defaults(func=cmd_gen)

    train = sub.add_parser("train-ranker", help="listwise training")
    train.add_argument("--data", type=Path, default=WORK / "listwise_train")
    train.add_argument("--out", type=Path, default=WORK / "ranker" / "ranker.pt")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--accumulate", type=int, default=8,
                       help="lists per optimiser step; one list is one softmax")
    train.add_argument("--seed", type=int, default=SEED)
    train.set_defaults(func=cmd_train_ranker)

    gt = sub.add_parser("gen-tracklets", help="labelled tracklets from stored lists")
    gt.add_argument("--data", type=Path, default=WORK / "listwise_train")
    gt.add_argument("--ranker", type=Path, default=WORK / "ranker" / "ranker.pt")
    gt.add_argument("--out", type=Path, default=WORK / "tracklets.json")
    gt.add_argument("--top-k", type=int, default=8)
    gt.add_argument("--positive-fraction", type=float, default=0.6,
                    help="fraction of its scored frames a tracklet must sit on the "
                         "ball to count as the ball")
    gt.set_defaults(func=cmd_gen_tracklets)

    tt = sub.add_parser("train-tracklet", help="fit the tracklet discriminator")
    tt.add_argument("--data", type=Path, default=WORK / "tracklets.json")
    tt.add_argument("--out", type=Path, default=WORK / "ranker" / "tracklet.pt")
    tt.add_argument("--epochs", type=int, default=400)
    tt.add_argument("--learning-rate", type=float, default=0.05)
    tt.add_argument("--drop", default=None,
                    help="comma-separated feature groups to zero (motion, body, "
                         "ranking) for an ablation")
    tt.add_argument("--seed", type=int, default=SEED)
    tt.set_defaults(func=cmd_train_tracklet)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
