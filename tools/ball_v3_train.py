"""Generate listwise training data, build tracklets, and train the v3 models.

Subcommands:

``gen``            build listwise examples from TRAIN clips: insert a synthetic
                   ball through a real window, run the real proposal stage on the
                   modified frames, and record which candidate is the ball among
                   the genuine distractors from that same photograph. One
                   *shard* per (round, clip); a shard is one synthetic sequence.
``train-ranker``   listwise softmax over [candidates..., null].
``gen-tracklets``  build labelled tracklets from the stored shards and a trained
                   ranker. Grouping is by shard, so two different insertions can
                   never be linked into one track.
``train-tracklet`` fit the temporal discriminator.

Splits are enforced, not assumed:

* ``gen`` refuses any clip that is not in the v2 ``train`` split;
* nothing in this file can read the v3 evaluation corpus;
* the temporal train/validation split is by **shoot family**, and every family
  containing a clip the ranker held out is held out here too. The ranker's
  probabilities are features of the temporal model, so a temporal validation
  clip the ranker trained on would make those features optimistic -- and a
  temporal *training* clip the ranker held out would make them pessimistic.
  Aligning the two splits is the only way that comparison means anything.

Everything here needs torch: use ``.venv-video-ai/bin/python3``. The body pose
cache is built separately by ``tools/ball_v3_body_cache.py`` under ``.venv``.

Nothing here writes to ``out/`` unless asked to; the default output root is
``out/ball_tracker_v3``, and every command takes an explicit path.
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
V3_EVAL_SPLIT = ROOT / "data" / "ball_dataset_v3_eval" / "split.json"
SEED = 20260918

#: Trajectory kinds exercised, from the brief's list. A tracker that only ever
#: sees a ball in flight cannot hold a tee'd ball at address, which is the state
#: most of this corpus is actually in.
TRAJECTORIES = ("stationary", "stationary", "slow", "linear", "fast",
                "accelerating", "entrance", "exit")


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def sealed_clip_ids() -> set[str]:
    """v3 final-test clip names. Metadata only; the media is never opened."""
    if not V3_EVAL_SPLIT.exists():
        return set()
    payload = json.loads(V3_EVAL_SPLIT.read_text(encoding="utf-8"))
    return {c["clip_id"] for c in payload["clips"] if c["split"] == "final_test"}


def eval_clip_ids() -> set[str]:
    """Every v3 evaluation clip, validation and final test alike."""
    if not V3_EVAL_SPLIT.exists():
        return set()
    payload = json.loads(V3_EVAL_SPLIT.read_text(encoding="utf-8"))
    return {c["clip_id"] for c in payload["clips"]}


def assert_not_evaluation(clip_ids) -> None:
    clash = set(clip_ids) & eval_clip_ids()
    if clash:
        raise SystemExit(f"refusing to train on v3 evaluation clips: {sorted(clash)}")


def source_records() -> dict[str, dict]:
    sources = json.loads(V2_SOURCES.read_text(encoding="utf-8"))
    return {c["clip_id"]: c for c in sources["clips"]}


def train_clips() -> list[dict]:
    split = json.loads(V2_SPLIT.read_text(encoding="utf-8"))
    by_id = source_records()
    wanted = [c["clip_id"] for c in split["clips"] if c["split"] == "train"]
    out = []
    for clip_id in wanted:
        record = by_id.get(clip_id)
        if not record:
            continue
        media = ROOT / record["media_path"]
        if media.exists():
            out.append({"clip_id": clip_id, "media": media,
                        "frames": record.get("frames") or 0,
                        "sha256": record.get("sha256", ""),
                        "source_group": record.get("source_group", clip_id)})
    return out


def family_split(clip_ids, ranker_holdout) -> dict[str, str]:
    """``train``/``validation`` by shoot family, aligned with the ranker holdout.

    Splitting by clip is not enough when two clips share a shoot: the same turf,
    the same golfer and the same lighting appear on both sides. Splitting by
    family is what ``data/ball_split.json`` already does for the legacy corpus,
    and it is the right unit here too.
    """
    holdout = set(ranker_holdout or ())
    groups: dict[str, list[str]] = {}
    for clip_id in clip_ids:
        groups.setdefault(source_records().get(clip_id, {})
                          .get("source_group", clip_id), []).append(clip_id)
    assignment: dict[str, str] = {}
    for _group, members in sorted(groups.items()):
        label = "validation" if set(members) & holdout else "train"
        for clip_id in members:
            assignment[clip_id] = label
    return assignment


def ranker_holdout(ranker_path: Path) -> list[str]:
    """The clips the ranker held out, read from its report.

    Accepts either the report itself or the weights beside it.
    """
    path = Path(ranker_path)
    report = path if path.suffix == ".json" else path.parent / "ranker_report.json"
    if not report.exists():
        return []
    return json.loads(report.read_text(encoding="utf-8")).get("held_out_clips", [])


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


def digest_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest() -> str:
    """A digest of the code that produced a result, so it can be tied to a run."""
    parts = []
    for relative in ("ghostcaddie/video/ball_v3_tracklets.py",
                     "ghostcaddie/video/ball_v3_dataset.py",
                     "ghostcaddie/video/ball_v3_proposals.py",
                     "ghostcaddie/video/ball_v3_ranker.py",
                     "ghostcaddie/video/ball_v3_tracker.py",
                     "tools/ball_v3_train.py"):
        parts.append(digest_of(ROOT / relative))
    return hashlib.sha256("".join(parts).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def cmd_gen(args) -> int:
    clips = train_clips()
    if not clips:
        print("no training media on disk", file=sys.stderr)
        return 2
    # the split guard: this must raise rather than quietly train on evaluation data
    split = bds.load_split(V2_SPLIT)
    bsyn.assert_training_clips(split, [c["clip_id"] for c in clips])
    assert_not_evaluation(c["clip_id"] for c in clips)

    families = family_split([c["clip_id"] for c in clips],
                            ranker_holdout(args.ranker_report)
                            if args.ranker_report else [])
    cache = bbody.BodyCache(BODY_CACHE)
    params = bp.ProposalParams(
        per_method_limit=args.per_method_limit,
        work_max_side=args.work_max_side,
        max_per_frame=args.max_per_frame)
    methods = tuple(args.methods.split(",")) if args.methods else bp.METHOD_NAMES

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    picker = random.Random(args.seed)
    started = time.time()
    written = skipped = misses = 0
    nulls = 0
    shards: list[dict] = []

    if args.pilot:
        clips = clips[: args.pilot_clips]
        rounds = 1
    else:
        rounds = args.rounds

    for round_index in range(rounds):
        for clip in clips:
            total = clip["frames"] or 0
            if total < args.length + 8:
                continue
            sequence_id = f"{clip['clip_id']}#r{round_index:02d}"
            shard_dir = out_root / f"r{round_index:02d}-{clip['clip_id']}"
            if (shard_dir / "manifest.json").exists() and not args.force:
                skipped += 1
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
            writer = bdata.ExampleWriter(shard_dir, meta={
                "sequence_id": sequence_id,
                "round_index": round_index,
                "clip_id": clip["clip_id"],
                "source_sha256": clip["sha256"],
                "source_group": clip["source_group"],
                "split": families.get(clip["clip_id"], "train"),
                "synthetic_seed": seed,
                "synthetic_kind": kind,
                "first_frame": first,
                "length": len(modified),
                "frame_spacing": 1,
                "params": {"per_method_limit": params.per_method_limit,
                           "work_max_side": params.work_max_side,
                           "max_per_frame": params.max_per_frame,
                           "list_size": args.list_size,
                           "methods": list(methods)},
                "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds",
                "negative_semantics": "candidate is not the inserted synthetic "
                                      "ball; a real ball elsewhere in the frame "
                                      "would be labelled a negative",
                "background_ball_annotation": False,
            })
            # the proposal stage runs on the MODIFIED frames, so the candidate list
            # holds the synthetic ball and the real clutter from the same footage
            for example in bdata.examples_from_sequence(
                    modified, samples, clip_id=clip["clip_id"],
                    frame_indices=indices, params=params, methods=methods,
                    pose_for=pose_for, numeric_fn=brank.numeric_features,
                    rng=rng, list_size=args.list_size, sequence_id=sequence_id,
                    round_index=round_index, source_sha256=clip["sha256"],
                    split=families.get(clip["clip_id"], "train"),
                    source_group=clip["source_group"]):
                if example.target is None and example.truth is None:
                    nulls += 1
                writer.add(example)
            summary = writer.close()
            summary["first_frame"] = first
            written += summary["lists"]
            misses += summary["proposal_misses"]
            shards.append(summary)
            elapsed = time.time() - started
            print(f"[round {round_index + 1}/{rounds}] {clip['clip_id']:20s} "
                  f"{kind:12s} lists={summary['lists']} "
                  f"misses={summary['proposal_misses']} "
                  f"total={written} {elapsed / 60:.1f} min", flush=True)
            if args.limit and written >= args.limit:
                break
        if args.limit and written >= args.limit:
            break

    run = {
        "shards": len(shards),
        "lists": written,
        "nulls": nulls,
        "proposal_misses": misses,
        "rounds": rounds,
        "clips": len(clips),
        "seconds": round(time.time() - started, 1),
        "params": {"per_method_limit": params.per_method_limit,
                   "work_max_side": params.work_max_side,
                   "max_per_frame": params.max_per_frame,
                   "list_size": args.list_size, "methods": list(methods)},
        "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds",
        "skipped_complete_shards": skipped,
        "code_sha256": code_digest(),
        "out_root": str(out_root),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(run, indent=1) + "\n",
                                                encoding="utf-8")
    print(json.dumps(run, indent=1))
    return 0


# ---------------------------------------------------------------------------
# listwise ranker
# ---------------------------------------------------------------------------

def augment(stacks: np.ndarray, numeric: np.ndarray,
            rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Flips and photometric jitter, applied to a whole candidate list at once.

    Needed because the store is small: 178k parameters over a few hundred lists
    memorised the training set by the second epoch (loss 1.478 -> 0.0014 over
    twelve), and model selection then had nothing but noise to choose between.

    A horizontal flip also has to flip the x geometry feature, or the network is
    handed a picture and a coordinate that disagree -- which teaches it that the
    coordinate means nothing.
    """
    stacks = np.ascontiguousarray(stacks)
    if rng.random() < 0.5:
        stacks = stacks[:, :, ::-1, :]
        numeric[:, 0] = 1.0 - numeric[:, 0]
    if rng.random() < 0.5:
        stacks = stacks[:, ::-1, :, :]
        numeric[:, 1] = 1.0 - numeric[:, 1]
    if rng.random() < 0.7:
        # the colour streams only; the motion channel is a difference and scaling
        # it would misrepresent how much actually moved
        gain = rng.uniform(0.75, 1.3)
        bias = rng.uniform(-18.0, 18.0)
        colour = stacks[..., :6].astype(np.float32) * gain + bias
        stacks = np.concatenate(
            [np.clip(colour, 0, 255).astype(np.uint8), stacks[..., 6:]], axis=-1)
    return np.ascontiguousarray(stacks), numeric


def _reader(args) -> bdata.MultiReader:
    if getattr(args, "data_root", None):
        return bdata.MultiReader.from_root(Path(args.data_root))
    return bdata.MultiReader(args.data)


def cmd_train_ranker(args) -> int:
    import torch

    reader = _reader(args)
    if not len(reader):
        print("no training lists found; run gen first", file=sys.stderr)
        return 2
    if reader.skipped:
        print(f"skipping {len(reader.skipped)} incomplete shards", file=sys.stderr)

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
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    history = []
    best_top1, best_state, best_epoch = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        total = 0.0
        optimiser.zero_grad()
        for position, record in enumerate(train):
            stacks = reader.stacks_for(record)
            numeric = reader.numeric_for(record).copy()
            stacks, numeric = augment(stacks, numeric, rng)
            # ascontiguousarray matters: the store is memory-mapped, so a
            # transposed view spans two contiguous subspaces and autograd's
            # backward pass rejects it rather than failing at the forward pass.
            images = torch.from_numpy(np.ascontiguousarray(
                stacks.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0).to(device)
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
        "architecture": "CandidateRanker: 7-channel two-scale stack + "
                        f"{brank.NUMERIC_FEATURES} numeric features -> listwise "
                        "softmax with a learned null slot",
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


# ---------------------------------------------------------------------------
# tracklets
# ---------------------------------------------------------------------------

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


#: Where the per-generator scores sit inside a stored numeric row:
#: geometry (4) + body (7) + witnesses/spread (2), then one score per method.
METHOD_SCORE_OFFSET = 4 + len(bbody.FEATURE_NAMES) + 2


def scores_from(record: dict) -> list[dict[str, float]]:
    out = []
    for row in record["numeric"]:
        out.append({name: float(row[METHOD_SCORE_OFFSET + index])
                    for index, name in enumerate(bp.ALL_METHOD_NAMES)
                    if METHOD_SCORE_OFFSET + index < len(row)})
    return out


def shifts_for_window(media: Path, indices) -> dict[int, tuple[float, float]]:
    """Camera shift into each frame of one window, by phase correlation."""
    frames = read_window(media, min(indices), max(indices) - min(indices) + 1)
    if not frames:
        return {}
    shifts: dict[int, tuple[float, float]] = {}
    previous_gray = None
    for offset, frame in enumerate(frames):
        index = min(indices) + offset
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous_gray is None:
            shifts[index] = (0.0, 0.0)
        else:
            window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]),
                                             cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(previous_gray, gray, window)
            shifts[index] = (float(dx), float(dy))
        previous_gray = gray
    return shifts


def sequences_from_records(records) -> tuple[dict[str, list[dict]], list[str]]:
    """Group stored lists by ``sequence_id`` and flag interrupted sequences.

    Grouping by sequence id -- not by clip, and not by frame adjacency -- is the
    whole point. Two synthetic insertions on the same clip can land on adjacent
    frame numbers; linking them would produce a track whose truth is the union of
    two different balls. The second return value names the sequences whose frames
    are not consecutive, which means the shard was interrupted mid-write and
    linking across it would invent continuity.
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        key = record.get("sequence_id") or f"legacy-{record['clip_id']}"
        grouped.setdefault(key, []).append(record)
    broken: list[str] = []
    for key, group in grouped.items():
        group.sort(key=lambda r: r["frame_index"])
        indices = [r["frame_index"] for r in group]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            broken.append(key)
    return grouped, broken


def cmd_gen_tracklets(args) -> int:
    """Build labelled tracklets from the stored shards and a trained ranker.

    A tracklet is positive when it sits on the true ball for most of the frames it
    holds. Everything else is a negative, and the negatives are the point: they
    are real distractors that the ranker already scored highly enough to be in the
    top-K, linked over time from the same footage.

    **Grouping is by shard, never by frame adjacency.** A shard is one synthetic
    insertion on one clip in one round, so two insertions whose windows happen to
    touch on the frame axis cannot be joined into a single track whose truth is
    the union of both.
    """
    import torch

    reader = _reader(args)
    if not len(reader):
        print("no stored lists found", file=sys.stderr)
        return 2
    if reader.skipped:
        print(f"skipping {len(reader.skipped)} incomplete shards", file=sys.stderr)
    ranker = brank.load_ranker(args.ranker)
    device = brank.device()
    cache = bbody.BodyCache(args.body_cache)
    media = {c["clip_id"]: c["media"] for c in train_clips()}

    by_sequence, broken = sequences_from_records(reader.records)

    rows: list[dict] = []
    gaps_seen: dict[str, int] = {}
    body_full = body_partial = body_none = body_unknown = 0
    #: sequences with at least one row, distinct from shards on disk: the header
    #: used to report the nonempty-shard count under the name "sequences", which
    #: is 189 shards against 119 distinct sequence ids in the rows.
    represented: set[str] = set()
    for sequence_id, records in sorted(by_sequence.items()):
        indices = [r["frame_index"] for r in records]
        if sequence_id in broken:
            # a shard is written frame by frame; a gap means the sequence was
            # interrupted, and linking across it would invent continuity
            print(f"  {sequence_id}: non-consecutive frames, skipped", flush=True)
            continue
        clip_id = records[0]["clip_id"]
        if clip_id in sealed_clip_ids():
            raise SystemExit(f"refusing to build tracklets on sealed clip {clip_id}")
        if len(records) < 3:
            continue
        clip_media = media.get(clip_id)
        if clip_media is None:
            print(f"  {sequence_id}: no media for {clip_id}, skipped", flush=True)
            continue
        shifts = shifts_for_window(clip_media, indices)

        nodes_by_frame: dict[int, list[btrk.Node]] = {}
        pool_sizes: dict[int, int] = {}
        truth: dict[int, tuple] = {}
        for record in records:
            stacks = reader.stacks_for(record)
            numeric = reader.numeric_for(record)
            result = brank.rank_frame(ranker, stacks, numeric, device=device)
            geometry = positions_from(record)
            scores = scores_from(record)
            short_side = max(1, min(record["width"], record["height"]))
            pose = cache.pose(clip_id, record["frame_index"])
            centre = bbody.torso_centre(pose)
            # Availability is a pose fact, recorded per node. It is NOT inferred
            # from the other node fields being set, and NOT inferred from the body
            # vector being zero -- a candidate genuinely far from every landmark
            # is legitimately zero.
            pose_usable = len(bbody.visible_points(pose)) >= 4
            order = np.argsort(-result["probabilities"])
            nodes = []
            for rank, slot in enumerate(order[: args.top_k]):
                slot = int(slot)
                x, y, radius = geometry[slot]
                body = bbody.features(pose, x, y, radius)
                stack = stacks[slot]
                nodes.append(btrk.Node(
                    frame_index=record["frame_index"], x=x, y=y, radius=radius,
                    probability=float(result["probabilities"][slot]),
                    margin=result["margin"] if rank == 0 else 0.0,
                    is_top1=rank == 0,
                    inside_torso=body[bbody.FEATURE_NAMES.index("inside_torso")],
                    distance_to_hands=body[bbody.FEATURE_NAMES.index(
                        "distance_to_hands_over_radius")] * 200.0,
                    torso_centre=centre,
                    chroma_score=scores[slot].get("local_distinctness", -1.0),
                    brightness_score=scores[slot].get("brightness_cue", -1.0),
                    radius_over_short_side=radius / short_side,
                    appearance=_stack_appearance(stack),
                    body_available=pose_usable))
            nodes_by_frame[record["frame_index"]] = nodes
            pool_sizes[record["frame_index"]] = int(record.get("pool_size") or 0)
            if record.get("truth"):
                truth[record["frame_index"]] = tuple(record["truth"])

        tracks = btrk.build_tracklets(nodes_by_frame, shifts, top_k=args.top_k,
                                      min_length=args.min_length,
                                      pool_sizes=pool_sizes)
        for track in tracks:
            # The drop decision covers the caller fields only. A missing pose is a
            # coverage fact about the corpus, not a builder bug, so it is counted
            # rather than dropping the tracklet: 241 of 776 tracklets had no pose
            # anywhere and were previously invisible to this audit.
            gaps = btrk.node_gaps(track, include_body=False)
            if gaps:
                for name in gaps:
                    gaps_seen[name] = gaps_seen.get(name, 0) + 1
                continue
            coverage = btrk.body_coverage(track)
            if coverage is None:
                body_unknown += 1
            elif coverage < 1.0:
                body_partial += 1
                if coverage == 0.0:
                    body_none += 1
            else:
                body_full += 1
            hits = scored = 0
            for node in track.nodes:
                target = truth.get(node.frame_index)
                if target is None:
                    continue
                scored += 1
                if math.hypot(node.x - target[0],
                              node.y - target[1]) <= max(6.0, target[2]):
                    hits += 1
            if scored == 0:
                # a tracklet whose frames carry no truth is neither a positive nor
                # a negative; it is not supervision, so it is dropped and counted
                continue
            label = 1 if hits / scored >= args.positive_fraction else 0
            rows.append({
                "sequence_id": sequence_id,
                "clip_id": clip_id,
                "source_group": records[0].get("source_group", ""),
                "split": records[0].get("split", ""),
                "round_index": records[0].get("round_index", 0),
                "source_sha256": records[0].get("source_sha256", ""),
                "first_frame": track.nodes[0].frame_index,
                "frames": len(track.nodes),
                "label": label,
                "on_ball_fraction": round(hits / scored, 3),
                "cycle_agreement": round(track.cycle_agreement, 3),
                # None means "no pose availability was recorded for this tracklet",
                # never 0.0: a missing pose and an unasked question are different
                # answers and only one of them is a measurement.
                "body_coverage": (None if coverage is None
                                  else round(coverage, 3)),
                "features": btrk.tracklet_features(
                    track, window=len(records), camera_shift=shifts),
            })
            represented.add(sequence_id)
        print(f"  {sequence_id}: {len(records)} lists -> {len(tracks)} tracklets "
              f"({len(rows)} total)", flush=True)

    payload = {
        "tracklets": rows,
        "positives": sum(r["label"] for r in rows),
        "negatives": sum(1 for r in rows if r["label"] == 0),
        "feature_names": list(btrk.FEATURE_NAMES),
        "unavailable_features": btrk.UNAVAILABLE_FEATURES,
        "ranker": str(args.ranker),
        "ranker_weights_sha256": brank.weights_hash(args.ranker),
        "positive_fraction": args.positive_fraction,
        "sequences": len(by_sequence),
        "sequences_with_tracklets": len(represented),
        "sequence_id_counting_note": (
            "`sequences` counts nonempty shards; `sequences_with_tracklets` counts "
            "distinct sequence_id values that produced a tracklet row"),
        "clips": len({r["clip_id"] for r in rows}),
        "gaps_dropped": gaps_seen,
        "body_coverage": {
            "tracklets_with_pose_everywhere": body_full,
            "tracklets_with_pose_somewhere": body_partial,
            "tracklets_with_no_pose_at_all": body_none,
            "tracklets_with_unknown_pose_status": body_unknown,
            "note": ("a zero body vector is not evidence of a missing pose; this "
                     "counts the per-node pose availability flag the builder "
                     "recorded, not whether the body features came back zero"),
        },
        "origin": "SYNTHETIC_GROUND_TRUTH on real TRAIN backgrounds",
        "label_semantics": "positive = tracklet sits within one ball radius of "
                           "the inserted synthetic ball on at least "
                           f"{args.positive_fraction:.0%} of its scored frames",
        "negative_semantics": "not the inserted synthetic ball. A real ball "
                              "elsewhere in the frame would be labelled negative; "
                              "the train clips carry no human ball annotations, so "
                              "this is not exhaustive ground truth",
        "grouping": "by shard (one synthetic insertion, one clip, one round); "
                    "frame adjacency is never used as a grouping key",
        "code_sha256": code_digest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"{len(rows)} tracklets, {payload['positives']} positive, "
          f"{payload['negatives']} negative; wrote {args.out}")
    if gaps_seen:
        print(f"dropped tracklets with unset node fields: {gaps_seen}")
    print(f"body coverage: {body_full} full, {body_partial} partial "
          f"({body_none} with no pose at all), {body_unknown} unknown")
    return 0


def _stack_appearance(stack: np.ndarray) -> tuple[float, float, float]:
    tight = stack[..., :3].astype(np.float32)
    context = stack[..., 3:6].astype(np.float32)
    return (float(tight.mean()), float(tight.std()), float(context.std()))


# ---------------------------------------------------------------------------
# temporal discriminator
# ---------------------------------------------------------------------------

def split_assignment(rows, holdout, rule: str) -> dict[str, str]:
    """``train``/``validation`` for the temporal model, by one of two rules.

    ``family``  every shoot family containing a ranker-held-out clip is held out.
                This is the strict rule: a family shares turf, golfer and light,
                so it must not straddle the split. It is also coarse, because the
                v2 ``source_group`` prefixes are coarse -- six held-out clips pull
                24 of the 34 clips into validation, which is reported rather than
                hidden.
    ``clip``    only the ranker-held-out clips themselves are held out. Balanced,
                and still free of ranker optimism, but families do straddle the
                split, so a shared background can appear on both sides.
    """
    if rule == "family":
        return family_split({r["clip_id"] for r in rows}, holdout)
    if rule == "clip":
        return {clip_id: ("validation" if clip_id in set(holdout) else "train")
                for clip_id in {r["clip_id"] for r in rows}}
    raise SystemExit(f"unknown split rule {rule!r}")


def cmd_train_tracklet(args) -> int:
    import torch

    payload = json.loads(args.data.read_text(encoding="utf-8"))
    rows = payload["tracklets"]
    if not rows:
        print("no tracklets; run gen-tracklets first", file=sys.stderr)
        return 2

    holdout = set(ranker_holdout(args.ranker)) if args.ranker else set()
    assignment = split_assignment(rows, holdout, args.split)
    train = [r for r in rows if assignment.get(r["clip_id"]) == "train"]
    validation = [r for r in rows if assignment.get(r["clip_id"]) == "validation"]
    if not validation or not train:
        print("split left one side empty", file=sys.stderr)
        return 2

    mask = btrk.mask_for_ablation(btrk.groups_for_mask(args.drop))
    device = brank.device()

    def tensors(records):
        # The mask is a buffer inside the model, so the raw features go in
        # unmasked. Applying it in two places would still be correct (it is 0/1)
        # but it would hide whether the checkpoint carries its own mask.
        features = np.asarray([r["features"] for r in records], dtype=np.float32)
        labels = np.asarray([r["label"] for r in records], dtype=np.float32)
        return (torch.from_numpy(features).to(device),
                torch.from_numpy(labels).to(device))

    train_x, train_y = tensors(train)
    validation_x, validation_y = tensors(validation)
    model = btrk.TrackletDiscriminator(mask=mask).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # the classes are lopsided -- most tracklets are clutter -- so the positive
    # term is weighted rather than letting the model win by always saying no
    positive = max(1.0, float(train_y.sum()))
    weight = torch.tensor(
        (len(train_y) - positive) / positive, device=device).clamp(1.0, 20.0)
    history = []
    best_auc, best_state, best_epoch = -1.0, None, 0
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
            best_auc, best_epoch = auc, epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    digest = btrk.save_discriminator(model, args.out)
    config = {
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "drop": args.drop or None, "seed": args.seed, "split": args.split,
        "positive_fraction": payload.get("positive_fraction"),
        "features": len(btrk.FEATURE_NAMES),
    }
    report = {"parameters": int(sum(p.numel() for p in model.parameters())),
              "train": len(train), "validation": len(validation),
              "positives": sum(r["label"] for r in rows),
              "train_positives": int(train_y.sum()),
              "validation_positives": int(validation_y.sum()),
              "best_validation_auc": best_auc,
              "best_epoch": best_epoch,
              "dropped_features": args.drop or None,
              "drop_groups": btrk.groups_for_mask(args.drop),
              "masked_features": btrk.dropped_feature_names(
                  btrk.groups_for_mask(args.drop)),
              "mask_applied_in_forward": True,
              "auc_estimator": "tie-corrected mid-rank Mann-Whitney",
              "weights_sha256": digest,
              "history": history,
              "held_out_clips": sorted({r["clip_id"] for r in validation}),
              "train_clips": sorted({r["clip_id"] for r in train}),
              "split_rule": args.split,
              "split_note": ("family: coarse because the v2 source_group prefixes "
                             "are coarse; every family containing a ranker-held-out "
                             "clip is held out" if args.split == "family" else
                             "clip: only the ranker-held-out clips are held out; "
                             "families straddle the split"),
              "ranker_holdout": sorted(holdout),
              "feature_names": list(btrk.FEATURE_NAMES),
              "unavailable_features": btrk.UNAVAILABLE_FEATURES,
              "config": config,
              "config_sha256": hashlib.sha256(
                  json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
              "data_sha256": digest_of(args.data),
              "code_sha256": code_digest(),
              "origin": payload.get("origin")}
    suffix = ("_" + args.drop.replace(",", "_")) if args.drop else ""
    if args.split != "family":
        suffix += f"_{args.split}"
    report_path = args.out.parent / f"tracklet_report{suffix}.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1))
    return 0


def _auc(scores: np.ndarray, labels: np.ndarray):
    """ROC AUC, tie-corrected by mid-ranks.

    The estimator this replaces ranked with ``np.argsort`` over
    ``concatenate(positives, negatives)`` and assigned *distinct* ranks, so every
    tie was broken in the positive class's favour. It is not a small bias: a
    model whose output had collapsed to a constant scored **0.674** where the
    correct answer is 0.5, and the no-motion ablation reported 0.9820 against a
    tie-correct 0.9636 (435 of its 570 validation scores were tied). A metric
    that rewards collapse is worse than no metric.

    Ties now contribute exactly 0.5 each, which is the Mann-Whitney definition.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = int((labels > 0.5).sum())
    negatives = int((labels <= 0.5).sum())
    if positives == 0 or negatives == 0:
        return None
    # 1-based mid-rank of every observation, shared by tied values
    _, inverse, counts = np.unique(scores, return_inverse=True,
                                   return_counts=True)
    cumulative = np.cumsum(counts)
    midrank_of_value = (cumulative - counts + 1 + cumulative) / 2.0
    ranks = midrank_of_value[inverse]
    total = float(ranks[labels > 0.5].sum())
    return float((total - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def _auc_ties(scores: np.ndarray) -> int:
    """How many observations share a score with another. Reported alongside every
    AUC so a high number cannot be mistaken for resolution."""
    values, counts = np.unique(np.asarray(scores, dtype=np.float64),
                               return_counts=True)
    return int(sum(counts[counts > 1]))


def cmd_recompute_auc(args) -> int:
    """Score saved checkpoints without retraining, on the exact same split.

    One number per checkpoint: the checkpoint's own mask applied, as it was
    during training. There is deliberately no second "as deployed without a
    mask" number. The mask is intrinsic to ``TrackletDiscriminator.forward``, so
    every path applies it; scoring with ``mask=None`` would return a bit-for-bit
    duplicate of the first column, and a reader would conclude masking changed
    nothing. Review 1 measured the opposite independently -- loading an ablation
    checkpoint and feeding the dropped columns live values collapses validation
    AUC from 0.9713 to 0.5612 -- and that measurement, not a duplicate, is the
    evidence the defect was real.

    The number is not a model-selection result. ``best_epoch`` in the original
    reports was chosen with the faulty estimator, so these are fixed-checkpoint
    scores of the weights that exist, not a re-validated optimum.
    """
    import torch

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    rows = payload["tracklets"]
    holdout = set(ranker_holdout(args.ranker))
    device = brank.device()

    def frame(records, mask):
        x = np.asarray([r["features"] for r in records], dtype=np.float32)
        if mask is not None:
            x = x * mask
        y = np.asarray([r["label"] for r in records], dtype=np.float32)
        return torch.from_numpy(x).to(device), y

    results = []
    for entry in spec:
        weights = Path(entry["weights"])
        drop = entry.get("drop") or []
        split = entry.get("split", "family")
        assignment = split_assignment(rows, holdout, split)
        train = [r for r in rows if assignment.get(r["clip_id"]) == "train"]
        validation = [r for r in rows
                      if assignment.get(r["clip_id"]) == "validation"]
        model = btrk.load_discriminator(weights, device_override=device)
        own_mask = btrk.discriminator_mask(model)
        row = {
            "name": entry["name"],
            "weights": str(weights),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "split": split,
            "drop_groups": drop,
            "mask_in_checkpoint": (None if own_mask is None
                                   else [float(v) for v in own_mask]),
            "masked_features": btrk.masked_feature_names(model),
            "train": len(train), "validation": len(validation),
            "train_positives": int(sum(r["label"] for r in train)),
            "validation_positives": int(sum(r["label"] for r in validation)),
        }
        scores = {}
        for part, records in (("train", train), ("validation", validation)):
            if not records:
                continue
            x, y = frame(records, own_mask)
            with torch.no_grad():
                out = torch.sigmoid(model(x)).cpu().numpy()
            scores[part] = (out, y)
        entry_row = {}
        for part, (out, y) in scores.items():
            entry_row[f"{part}_auc"] = _auc(out, y)
            entry_row[f"{part}_tied_scores"] = _auc_ties(out)
            entry_row[f"{part}_distinct_scores"] = int(
                len(np.unique(np.round(out, 6))))
            entry_row[f"{part}_constant"] = bool(
                len(np.unique(np.round(out, 6))) == 1)
        row["as_trained"] = entry_row
        row["as_deployed_no_mask"] = {
            "measurement": None,
            "reason": ("not measured: the ablation mask is intrinsic to "
                       "TrackletDiscriminator.forward, so no path bypasses it. "
                       "Scoring with mask=None reproduces as_trained exactly and "
                       "would read as 'masking changed nothing'"),
        }
        row["nonconverged_constant_output"] = bool(
            row["as_trained"].get("validation_constant"))
        results.append(row)

    out = {
        "note": ("tie-corrected AUC of saved checkpoints on the exact splits they "
                 "were trained on, with each checkpoint's own persisted ablation "
                 "mask applied. as_deployed_no_mask is null: the mask is intrinsic "
                 "to forward, so an unmasked score cannot be measured -- it would "
                 "be a duplicate. best_epoch was selected with the faulty "
                 "estimator, so these are fixed-checkpoint scores, NOT a newly "
                 "valid model selection. Synthetic-candidate-list measures; not "
                 "real-ball accuracy."),
        "estimator": "mid-rank Mann-Whitney (ties contribute 0.5)",
        "data": str(args.data),
        "data_sha256": digest_of(args.data),
        "ranker": str(args.ranker),
        "code_sha256": code_digest(),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in results:
        print(f"{row['name']:26s} split={row['split']:6s} "
              f"val AUC (mask applied) = "
              f"{_fmt(row['as_trained'].get('validation_auc'))} "
              f"(ties {row['as_trained'].get('validation_tied_scores')}) "
              f"as-deployed-unmasked = null "
              f"({row['as_deployed_no_mask'].get('reason')})")
    return 0


def _fmt(value) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="build listwise training examples")
    gen.add_argument("--out-root", type=Path, default=WORK / "listwise_shards",
                     help="one shard subdirectory per (round, clip)")
    gen.add_argument("--rounds", type=int, default=6)
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
    gen.add_argument("--force", action="store_true",
                     help="regenerate shards that are already complete")
    gen.add_argument("--pilot", action="store_true",
                     help="one round over the first --pilot-clips clips, to "
                          "measure cost and disk before committing")
    gen.add_argument("--pilot-clips", type=int, default=3)
    gen.add_argument("--ranker-report", type=Path, default=None,
                     help="used to align the family split with the ranker holdout")
    gen.set_defaults(func=cmd_gen)

    train = sub.add_parser("train-ranker", help="listwise training")
    train.add_argument("--data", type=Path, action="append", default=None,
                       help="a listwise store; repeat to train on several")
    train.add_argument("--data-root", type=Path, default=None,
                       help="a shard root; every complete shard is used")
    train.add_argument("--out", type=Path, default=WORK / "ranker" / "ranker.pt")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--accumulate", type=int, default=8,
                       help="lists per optimiser step; one list is one softmax")
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=SEED)
    train.set_defaults(func=cmd_train_ranker)

    gt = sub.add_parser("gen-tracklets", help="labelled tracklets from stored lists")
    gt.add_argument("--data", type=Path, action="append", default=None)
    gt.add_argument("--data-root", type=Path, default=None)
    gt.add_argument("--ranker", type=Path, default=WORK / "ranker" / "ranker.pt")
    gt.add_argument("--body-cache", type=Path, default=BODY_CACHE)
    gt.add_argument("--out", type=Path, default=WORK / "tracklets.json")
    gt.add_argument("--top-k", type=int, default=8)
    gt.add_argument("--min-length", type=int, default=3)
    gt.add_argument("--positive-fraction", type=float, default=0.6,
                    help="fraction of its scored frames a tracklet must sit on the "
                         "ball to count as the ball")
    gt.set_defaults(func=cmd_gen_tracklets)

    tt = sub.add_parser("train-tracklet", help="fit the tracklet discriminator")
    tt.add_argument("--data", type=Path, default=WORK / "tracklets.json")
    tt.add_argument("--out", type=Path, default=WORK / "ranker" / "tracklet.pt")
    tt.add_argument("--ranker", type=Path, default=WORK / "ranker" / "ranker.pt",
                    help="its report defines the holdout the family split aligns to")
    tt.add_argument("--epochs", type=int, default=400)
    tt.add_argument("--learning-rate", type=float, default=0.05)
    tt.add_argument("--split", default="family", choices=("family", "clip"),
                    help="family (strict, coarse) or clip (balanced, families "
                         "straddle the split); both align to the ranker holdout")
    tt.add_argument("--drop", default=None,
                    help="comma-separated feature groups to zero (motion, body, "
                         "ranking, appearance, chroma, scale) for an ablation")
    tt.add_argument("--seed", type=int, default=SEED)
    tt.set_defaults(func=cmd_train_tracklet)

    ra = sub.add_parser(
        "recompute-auc",
        help="tie-correct AUC of saved checkpoints on their exact prior splits")
    ra.add_argument("--spec", type=Path, required=True,
                    help="JSON list of {name, weights, drop, split} entries")
    ra.add_argument("--data", type=Path, required=True,
                    help="the tracklets.json the checkpoints were trained on")
    ra.add_argument("--ranker", type=Path, required=True,
                    help="the ranker whose holdout defines the split")
    ra.add_argument("--out", type=Path, required=True)
    ra.set_defaults(func=cmd_recompute_auc)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "train-ranker" and not args.data \
            and not args.data_root:
        args.data = [WORK / "listwise_train"]
    if getattr(args, "command", None) == "gen-tracklets" and not args.data \
            and not args.data_root:
        args.data = [WORK / "listwise_train"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
