"""Research CLI for tracker v2: synthetic supervision, training, evaluation, tracking.

Everything here is research-only. Nothing in this file may be reached from the
canonical workflow, and the canonical ball layer stays UNAVAILABLE unless a future
promotion decision says otherwise.

Split discipline is enforced rather than trusted:

* training data is generated from **train** clips only;
* validation clips may be used for *evaluation* (synthetic insertions with exact
  ground truth, behavioural diagnostics), never for training;
* final-test clips are reached only through ``final-test``, which requires an
  explicit ``--final-test`` flag, and its numbers are reported once, frozen.

Usage:
    python3 tools/ball_track_v2.py gen-synth
    python3 tools/ball_track_v2.py train
    python3 tools/ball_track_v2.py eval-synth
    python3 tools/ball_track_v2.py track-synth --split validation
    python3 tools/ball_track_v2.py diagnostics --split validation
    python3 tools/ball_track_v2.py track --clip-id pexels_14980420 --out out/x
    python3 tools/ball_track_v2.py final-test --final-test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_candidates as bc  # noqa: E402
from ghostcaddie.video import ball_consensus as bcon  # noqa: E402
from ghostcaddie.video import ball_detector as bdet  # noqa: E402
from ghostcaddie.video import ball_eval_v2 as bev  # noqa: E402
from ghostcaddie.video import ball_synthetic as bs  # noqa: E402
from ghostcaddie.video import ball_tracker_v2 as btr  # noqa: E402
from ghostcaddie.video.errors import VideoContractError  # noqa: E402

SOURCES = ROOT / "data" / "ball_dataset_v2" / "sources.json"
SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"
MEDIA = ROOT / "out" / "ball_dataset_v2" / "media"
WORK = ROOT / "out" / "ball_tracker_v2"
SYNTHETIC = WORK / "synthetic_train"
DETECTOR_DIR = WORK / "detector"
DETECTOR_STATE = DETECTOR_DIR / "detector_state.pt"

TRAIN_SEED = 20260918
EVAL_SEED = 771103


def dataset() -> tuple[dict, dict]:
    if not (SOURCES.exists() and SPLIT.exists()):
        raise SystemExit("dataset v2 manifest/split not found")
    return (json.loads(SOURCES.read_text()), json.loads(SPLIT.read_text()))


def clips_for(split_name: str, *, allow_final_test: bool = False) -> list[dict]:
    sources, split = dataset()
    if split_name == "final_test" and not allow_final_test:
        raise SystemExit("final test is sealed: pass --final-test to reach it")
    wanted = {clip["clip_id"] for clip in split["clips"] if clip["split"] == split_name}
    return [clip for clip in sources["clips"] if clip["clip_id"] in wanted]


def read_window(clip: dict, start: int, count: int) -> list:
    media = MEDIA / Path(clip["media_path"]).name
    return bs._read_clip_window(media, start, count)


def load_detector_or_none() -> object | None:
    if not DETECTOR_STATE.exists():
        print("no trained detector yet: running the classical ensemble only")
        return None
    return bdet.load_detector(DETECTOR_STATE)


def detector_provenance() -> dict:
    report_path = DETECTOR_DIR / "detector_report.json"
    if not report_path.exists():
        return {"detector": "none"}
    report = json.loads(report_path.read_text())
    return {"detector_weights": report.get("weights"),
            "detector_sha256": report.get("weights_sha256"),
            "detector_epochs": report.get("epochs")}


# ---------------------------------------------------------------------------
# synthetic supervision and training
# ---------------------------------------------------------------------------

def cmd_gen_synth(args) -> int:
    sources, split = dataset()
    train = clips_for("train")
    bs.assert_training_clips(split, [clip["clip_id"] for clip in train])
    start = time.time()
    manifest = bs.generate_synthetic_crops(
        train_clips=train, media_root=MEDIA, out_dir=SYNTHETIC,
        crop_size=bdet.CROP_SIZE, sequences=args.sequences,
        sequence_length=args.sequence_length, seed=args.seed)
    manifest["elapsed_s"] = round(time.time() - start, 1)
    manifest["split"] = "train"
    manifest["purpose"] = "detector training (synthetic supervision on real backgrounds)"
    (SYNTHETIC / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"positives={manifest['positives']} negatives={manifest['negatives']} "
          f"skipped_unverified={manifest['unverified_skipped']} "
          f"elapsed={manifest['elapsed_s']}s")
    return 0


def cmd_train(args) -> int:
    records = bs.load_crop_records(SYNTHETIC / "crops.jsonl")
    if not records:
        raise SystemExit("no synthetic crops yet: run `gen-synth` first")
    extra: list[dict] = []
    pseudo_path = WORK / "pseudo_labels" / "crops.jsonl"
    if args.use_pseudo and pseudo_path.exists():
        extra = bs.load_crop_records(pseudo_path)
        print(f"adding {len(extra)} high-confidence pseudo-label crops")
    if args.hard_negatives:
        mined_path = WORK / "hard_negatives.jsonl"
        if not mined_path.exists():
            raise SystemExit("no mined hard negatives yet: run `consensus` first")
        # Mined failures are copied into the training crop directory rather than
        # referenced from a second root: a record and its image must stay together or
        # the loader silently skips samples that look present in the manifest.
        import shutil
        copied = 0
        for line in mined_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = f"hn_{record['path']}"
            target = SYNTHETIC / "crops" / name
            if not target.exists():
                source = WORK / "hard_negatives" / record["path"]
                if not source.exists():
                    continue
                shutil.copy2(source, target)
            extra.append({"path": name, "positive": False,
                          "label_x": bdet.CROP_SIZE / 2, "label_y": bdet.CROP_SIZE / 2,
                          "origin": "HARD_NEGATIVE_MINING",
                          "radius": record.get("radius"),
                          "mined_from": record.get("clip_id")})
            copied += 1
        print(f"adding {copied} mined hard negatives (real non-ball candidates)")
    report = bdet.train_detector(records + extra, SYNTHETIC / "crops",
                                 out_dir=DETECTOR_DIR, epochs=args.epochs,
                                 seed=args.seed, learning_rate=args.lr,
                                 heatmap_weight=args.heatmap_weight)
    print(f"trained {report['parameters']} parameters on {report['train_crops']} crops; "
          f"weights sha256={report['weights_sha256'][:16]}...")
    val = report["history"][-1]["val"]
    print(f"held-out synthetic crops: precision={val['precision']} recall={val['recall']} "
          f"f1={val['f1']} median_center_error={val['median_center_error_px']} px")
    return 0


def cmd_eval_synth(args) -> int:
    """Detector-level synthetic evaluation on a *different* seed and clip window."""
    model = bdet.load_detector(DETECTOR_STATE)
    sources, split = dataset()
    clips = clips_for(args.split, allow_final_test=args.final_test)
    if not clips:
        raise SystemExit(f"no clips in split {args.split}")
    import random
    rng = random.Random(EVAL_SEED)
    positives, negatives = [], []
    for clip in clips[:args.clips]:
        for _ in range(max(1, args.sequences // max(1, len(clips[:args.clips])))):
            length = clip.get("frames") or 200
            start = rng.randint(0, max(0, length - args.sequence_length - 1))
            frames = read_window(clip, start, args.sequence_length)
            if len(frames) < 4:
                continue
            modified, samples = bs.synthesize_sequence(frames, seed=rng.randint(0, 2**31))
            for frame, sample in zip(modified, samples):
                if not sample.verified or sample.visibility == "OCCLUDED":
                    continue
                crop = bs.extract_crop(frame, sample.x, sample.y, bdet.CROP_SIZE)
                if crop is None:
                    continue
                lx, ly = bs.crop_label(sample.x, sample.y, bdet.CROP_SIZE)
                positives.append({"image": crop, "label_x": lx, "label_y": ly,
                                  "radius": sample.radius_px, "positive": True})
                far = bs.extract_crop(frame, sample.x + 12 * max(4.0, sample.radius_px),
                                      sample.y, bdet.CROP_SIZE)
                if far is not None:
                    negatives.append({"image": far, "label_x": bdet.CROP_SIZE / 2,
                                      "label_y": bdet.CROP_SIZE / 2, "positive": False})
    if not positives:
        raise SystemExit("no verified synthetic positives were produced")
    import numpy as np
    images = np.stack([item["image"] for item in positives + negatives])
    labels = [{"label_x": item["label_x"], "label_y": item["label_y"],
               "radius": item.get("radius", 3.0), "positive": item["positive"]}
              for item in positives + negatives]
    metrics = bdet.evaluate_detector(model, images, labels, device=bdet.device())
    metrics["split"] = args.split
    metrics["seed"] = EVAL_SEED
    out = WORK / f"synthetic_eval_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=1) + "\n")
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in ("per_clip",)}, indent=1))
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# tracker evaluation
# ---------------------------------------------------------------------------

def synthetic_tracker_run(clip: dict, *, seed: int, sequence_length: int,
                          detector, params, interpolate: bool = False) -> dict:
    import random
    rng = random.Random(seed)
    length = clip.get("frames") or 200
    start = rng.randint(0, max(0, length - sequence_length - 1))
    frames = read_window(clip, start, sequence_length)
    if len(frames) < 4:
        return {}
    modified, samples = bs.synthesize_sequence(frames, seed=rng.randint(0, 2**31))
    indices = [start + sample.frame_index for sample in samples]
    result = btr.track_ball_v2(modified, indices, detector=detector, params=params,
                               interpolate=interpolate,
                               provenance={"source": "synthetic_insertion",
                                           "clip_id": clip["clip_id"]})
    payload = result.to_dict()
    metrics = bev.synthetic_sequence_metrics(payload["frames"],
                                             [sample.as_label() | {"frame_index": sample.frame_index,
                                                                   "x": sample.x, "y": sample.y,
                                                                   "radius_px": sample.radius_px}
                                              for sample in samples],
                                             width=clip["width"], height=clip["height"])
    metrics["clip_id"] = clip["clip_id"]
    metrics["frame_start"] = start
    metrics["accepted"] = result.accepted
    metrics["confidence"] = result.confidence
    return {"metrics": metrics, "result": payload, "samples": samples,
            "clip": clip, "start": start}


def cmd_track_synth(args) -> int:
    model = load_detector_or_none()
    params = btr.TrackerParams()
    clips = clips_for(args.split, allow_final_test=args.final_test)[:args.clips]
    per_clip, runs = [], []
    for clip in clips:
        run = synthetic_tracker_run(clip, seed=EVAL_SEED + hash(clip["clip_id"]) % 1000,
                                    sequence_length=args.sequence_length,
                                    detector=model, params=params)
        if not run:
            continue
        per_clip.append(run["metrics"])
        runs.append(run)
        print(f"{clip['clip_id']:22s} P={run['metrics']['precision']:.3f} "
              f"R={run['metrics']['recall']:.3f} "
              f"FP={run['metrics']['false_positives']:3d} "
              f"FN={run['metrics']['false_negatives']:3d} "
              f"silence_ok={run['metrics']['correct_silence']:3d} "
              f"median_err={run['metrics']['median_center_error_px']} "
              f"accepted={run['metrics']['accepted']}")
    pooled = bev.aggregate_synthetic_metrics(per_clip)
    pooled["split"] = args.split
    payload = {"split": args.split, "seed": EVAL_SEED, "pooled": pooled,
               "params": params.__dict__,
               "per_clip": [{key: value for key, value in clip.items()
                             if key != "confidence"} for clip in per_clip]}
    out = WORK / f"tracker_synthetic_{args.split}.json"
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print(f"\nPOOLED {args.split}: precision={pooled.get('precision')} "
          f"recall={pooled.get('recall')} f1={pooled.get('f1')} "
          f"FP={pooled.get('false_positives')} FN={pooled.get('false_negatives')} "
          f"median_err={pooled.get('median_of_clip_median_center_error_px')}")
    print(f"wrote {out}")
    return 0


def cmd_negatives(args) -> int:
    """Negative sequences: real footage with no inserted ball.

    Nothing should be accepted here. This measures undeserved confidence, which is
    the failure that matters more than a miss."""
    model = load_detector_or_none()
    params = btr.TrackerParams()
    clips = clips_for(args.split, allow_final_test=args.final_test)[:args.clips]
    results = []
    for clip in clips:
        frames = read_window(clip, args.start, args.sequence_length)
        if len(frames) < 4:
            continue
        indices = list(range(args.start, args.start + len(frames)))
        result = btr.track_ball_v2(frames, indices, detector=model, params=params,
                                   provenance={"source": "no_ball_inserted",
                                               "clip_id": clip["clip_id"]})
        results.append(result.to_dict())
        print(f"{clip['clip_id']:22s} accepted={result.accepted} "
              f"observed={result.confidence['observed_frames']} "
              f"coverage={result.confidence['coverage']}")
    report = bev.negative_sequence_report(results)
    report["split"] = args.split
    out = WORK / f"negatives_{args.split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


def cmd_diagnostics(args) -> int:
    """Real-video behavioural diagnostics: consensus tiers, locks, abstention."""
    model = load_detector_or_none()
    params = btr.TrackerParams()
    clips = clips_for(args.split, allow_final_test=args.final_test)[:args.clips]
    summaries, results = [], []
    for clip in clips:
        frames = read_window(clip, args.start, args.sequence_length)
        if len(frames) < 4:
            continue
        indices = list(range(args.start, args.start + len(frames)))
        contexts, _ = bc.build_contexts(frames, indices)
        candidates = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window(clip["clip_id"], contexts,
                                              candidates_by_frame=candidates)
        summary = bcon.summarize_consensus(consensus)
        summaries.append(summary)
        result = btr.track_ball_v2(frames, indices, detector=model, params=params,
                                   provenance={"source": "real_footage",
                                               "clip_id": clip["clip_id"]})
        results.append(result.to_dict())
        print(f"{clip['clip_id']:22s} tiers={summary['tiers']} "
              f"trainable={summary['trainable_labels']} "
              f"accepted={result.accepted} coverage={result.confidence['coverage']}")
    report = bev.real_video_diagnostics(summaries, results)
    report["split"] = args.split
    out = WORK / f"diagnostics_{args.split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "negatives"}, indent=1))
    print(f"wrote {out}")
    return 0


def cmd_consensus(args) -> int:
    """Mine hard negatives and collect HIGH-confidence pseudo-labels from train."""
    clips = clips_for("train")
    objects_dir = WORK / "hard_negatives"
    objects_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = WORK / "pseudo_labels"
    (pseudo_dir / "crops").mkdir(parents=True, exist_ok=True)
    import cv2
    negatives, pseudo = [], []
    for clip in clips[:args.clips]:
        frames = read_window(clip, args.start, args.sequence_length)
        if len(frames) < 4:
            continue
        indices = list(range(args.start, args.start + len(frames)))
        contexts, _ = bc.build_contexts(frames, indices)
        candidates = bc.ensemble_candidates(contexts)
        consensus = bcon.consensus_for_window(clip["clip_id"], contexts,
                                              candidates_by_frame=candidates)
        mined = bcon.mine_hard_negatives(clip["clip_id"], contexts,
                                         candidates_by_frame=candidates,
                                         consensus=consensus, out_dir=objects_dir,
                                         max_per_clip=args.max_negatives)
        for record in mined:
            left, top, size, _ = record["crop"]
            # The candidate's own position has to be in the filename: several rejected
            # candidates share a frame, and a per-frame name silently overwrote most of
            # them, leaving records that pointed at a different candidate's pixels.
            target = objects_dir / (f"hn_{clip['clip_id']}_{record['frame_index']}_"
                                    f"{int(record['x'])}_{int(record['y'])}.jpg")
            cv2.imwrite(str(target), frames[indices.index(record["frame_index"])]
                        [top:top + size, left:left + size])
            record["path"] = target.name
            negatives.append(record)
        for label in bcon.training_labels(consensus):
            if label.x is None or label.y is None:
                continue                      # never invent a coordinate
            index = indices.index(label.frame_index)
            crop = bs.extract_crop(frames[index], label.x, label.y, bdet.CROP_SIZE)
            if crop is None:
                continue
            lx, ly = bs.crop_label(label.x, label.y, bdet.CROP_SIZE)
            name = f"pseudo_{clip['clip_id']}_{label.frame_index}.jpg"
            cv2.imwrite(str(pseudo_dir / "crops" / name), crop)
            pseudo.append({"path": name, "label_x": lx, "label_y": ly,
                           "radius": label.radius, "positive": True,
                           "origin": bcon.ORIGIN_CONSENSUS,
                           "tier": label.tier, "methods": label.methods,
                           "temporal": label.temporal, "clip_id": clip["clip_id"],
                           "frame_index": label.frame_index})
        print(f"{clip['clip_id']:22s} hard_negatives={len(mined)} "
              f"pseudo={sum(1 for p in pseudo if p['clip_id'] == clip['clip_id'])}")
    import numpy as np
    if negatives:
        spreads = [record["radius"] for record in negatives]
        print(f"\nhard negatives: n={len(negatives)} median radius={np.median(spreads):.1f} px")
    if pseudo:
        methods = [len(record["methods"]) for record in pseudo]
        print(f"pseudo-labels (HIGH only): n={len(pseudo)} "
              f"median methods agreeing={np.median(methods):.0f}")
    (WORK / "hard_negatives.jsonl").write_text(
        "\n".join(json.dumps(record, default=str) for record in negatives) + "\n")
    (pseudo_dir / "crops.jsonl").write_text(
        "\n".join(json.dumps(record, default=str) for record in pseudo) + "\n")
    print(f"wrote {WORK / 'hard_negatives.jsonl'} and "
          f"{pseudo_dir / 'crops.jsonl'}")
    return 0


def train_and_eval_rounds(args) -> int:
    """Round 1 on synthetic only; round 2 adds HIGH pseudo-labels, with a gate."""
    print("=== round 1: synthetic ground truth only")
    cmd_train(argparse.Namespace(epochs=args.epochs, seed=TRAIN_SEED,
                                 use_pseudo=False, rounds=1))
    images, labels = load_synthetic_holdout()
    base = bdet.evaluate_detector(bdet.load_detector(DETECTOR_STATE), images, labels,
                                  device=bdet.device())
    print(f"round 1 held-out: {json.dumps(base)}")
    rounds = [{"round": 1, "use_pseudo": False, "holdout": base}]

    pseudo_path = WORK / "pseudo_labels" / "crops.jsonl"
    if args.self_train and pseudo_path.exists():
        print("=== round 2: synthetic + HIGH-consensus pseudo-labels")
        cmd_train(argparse.Namespace(epochs=args.epochs, seed=TRAIN_SEED + 1,
                                     use_pseudo=True, rounds=2))
        images, labels = load_synthetic_holdout()
        after = bdet.evaluate_detector(bdet.load_detector(DETECTOR_STATE), images,
                                       labels, device=bdet.device())
        print(f"round 2 held-out: {json.dumps(after)}")
        rounds.append({"round": 2, "use_pseudo": True, "holdout": after})
        # the gate: a round that does not help is discarded, not kept
        if after.get("f1", 0.0) <= base.get("f1", 0.0):
            print("round 2 did not improve held-out F1: reverting to round 1 weights. "
                  "Self-training that makes validation worse is stopped, not tuned "
                  "around.")
            cmd_train(argparse.Namespace(epochs=args.epochs, seed=TRAIN_SEED,
                                         use_pseudo=False, rounds=1))
            rounds[-1]["reverted"] = True
    (WORK / "training_rounds.json").write_text(json.dumps(rounds, indent=1, default=str) + "\n")
    return 0


def load_synthetic_holdout():
    """A held-out slice of the synthetic set, used only to gate self-training."""
    records = bs.load_crop_records(SYNTHETIC / "crops.jsonl")
    subset = records[::7][:1200]
    images, labels = bdet.load_crops(subset, SYNTHETIC / "crops")
    return images, labels


def cmd_track(args) -> int:
    """Track one clip and write the result plus a QA sheet."""
    import cv2
    sources, split = dataset()
    clip = next((c for c in sources["clips"] if c["clip_id"] == args.clip_id), None)
    if clip is None:
        raise SystemExit(f"unknown clip {args.clip_id}")
    split_of = {c["clip_id"]: c["split"] for c in split["clips"]}
    model = load_detector_or_none()
    params = btr.TrackerParams()
    frames = read_window(clip, args.start, args.sequence_length)
    indices = list(range(args.start, args.start + len(frames)))
    provenance = {"clip_id": clip["clip_id"], "split": split_of.get(clip["clip_id"]),
                  "source_sha256": clip["sha256"],
                  "license": clip["license"], **detector_provenance()}
    result = btr.track_ball_v2(frames, indices, detector=model, params=params,
                               interpolate=args.interpolate, provenance=provenance)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "track.json").write_text(json.dumps(result.to_dict(), indent=1,
                                                   default=str) + "\n")
    _qa_sheet(frames, indices, result, clip, out_dir / "qa.jpg")
    print(f"{clip['clip_id']} ({split_of.get(clip['clip_id'])}): accepted={result.accepted}")
    print(json.dumps(result.confidence, indent=1))
    if result.reasons:
        print("not accepted because:")
        for reason in result.reasons:
            print(f"  - {reason}")
    print(f"wrote {out_dir / 'track.json'} and {out_dir / 'qa.jpg'}")
    return 0


def _qa_sheet(frames, indices, result, clip, path: Path, max_tiles: int = 12) -> None:
    """Draw the track on its own frames: state, trail, alternatives, exclusions."""
    import cv2
    import numpy as np
    states = result.states_by_index()
    step = max(1, len(frames) // max_tiles)
    tiles = []
    for position in range(0, len(frames), step):
        frame = frames[position].copy()
        index = indices[position]
        colour_map = {btr.STATE_OBSERVED: (0, 255, 255), btr.STATE_AMBIGUOUS: (0, 165, 255),
                      btr.STATE_OCCLUDED: (255, 128, 0), btr.STATE_LOST: (0, 0, 255),
                      btr.STATE_UNAVAILABLE: (128, 128, 128)}
        state = states.get(index)
        if state is not None and state.x is not None and state.state in (
                btr.STATE_OBSERVED, btr.STATE_AMBIGUOUS):
            centre = (int(state.x), int(state.y))
            colour = colour_map[state.state]
            cv2.drawMarker(frame, centre, colour, cv2.MARKER_CROSS, 28, 2)
            if state.radius:
                cv2.circle(frame, centre, int(max(3, state.radius)), colour, 2)
            if state.interpolated:
                cv2.putText(frame, "INTERPOLATED", (centre[0] + 6, centre[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        label = (f"f{index} {state.state if state else '?'}"
                 f"{' ' + ','.join(state.diagnostics) if state and state.diagnostics else ''}")
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 620), 30), (0, 0, 0), -1)
        cv2.putText(frame, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        width = 480
        factor = width / frame.shape[1]
        tiles.append(cv2.resize(frame, (width, max(1, int(frame.shape[0] * factor)))))
    if not tiles:
        return
    height = max(tile.shape[0] for tile in tiles)
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((height * rows, 480 * columns, 3), dtype="uint8")
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        sheet[row * height:row * height + tile.shape[0],
              column * 480:column * 480 + tile.shape[1]] = tile
    cv2.imwrite(str(path), sheet)


def cmd_final_test(args) -> int:
    """The sealed evaluation. Refuses to run without an explicit flag."""
    if not args.final_test:
        raise SystemExit("final test is opt-in: pass --final-test. It is scored once, "
                         "after thresholds are frozen, and tuning on it is forbidden.")
    model = load_detector_or_none()
    params = btr.TrackerParams()
    clips = clips_for("final_test", allow_final_test=True)[:args.clips]
    per_clip = []
    for clip in clips:
        run = synthetic_tracker_run(clip, seed=EVAL_SEED + 500,
                                    sequence_length=args.sequence_length,
                                    detector=model, params=params)
        if not run:
            continue
        per_clip.append(run["metrics"])
        print(f"{clip['clip_id']:22s} P={run['metrics']['precision']:.3f} "
              f"R={run['metrics']['recall']:.3f} FP={run['metrics']['false_positives']} "
              f"median_err={run['metrics']['median_center_error_px']}")
    pooled = bev.aggregate_synthetic_metrics(per_clip)
    payload = {"procedure": "synthetic insertions on final-test backgrounds, plus "
                            "temporal/cycle diagnostics; no human labels exist, so "
                            "no real-footage accuracy number is claimed",
               "scored_once": True, "pooled": pooled, "per_clip": per_clip,
               "detector": detector_provenance(), "params": params.__dict__}
    out = WORK / "final_test_result.json"
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print(f"\nFINAL TEST (once): precision={pooled.get('precision')} "
          f"recall={pooled.get('recall')} f1={pooled.get('f1')} "
          f"FP_on_absent={pooled.get('false_positive_rate_on_absent')}")
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-synth")
    p.add_argument("--sequences", type=int, default=340)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.set_defaults(func=cmd_gen_synth)

    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--use-pseudo", action="store_true")
    p.add_argument("--hard-negatives", action="store_true")
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--heatmap-weight", type=float, default=0.4,
                   help="weight of the localisation map; it is a regulariser here, "
                        "not a co-equal objective, because the ensemble supplies "
                        "coordinates and this head supplies the decision")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("rounds")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--self-train", action="store_true")
    p.set_defaults(func=train_and_eval_rounds)

    p = sub.add_parser("eval-synth")
    p.add_argument("--split", default="validation")
    p.add_argument("--clips", type=int, default=8)
    p.add_argument("--sequences", type=int, default=24)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--final-test", action="store_true")
    p.set_defaults(func=cmd_eval_synth)

    p = sub.add_parser("consensus")
    p.add_argument("--clips", type=int, default=10)
    p.add_argument("--start", type=int, default=30)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--max-negatives", type=int, default=40)
    p.set_defaults(func=cmd_consensus)

    p = sub.add_parser("track-synth")
    p.add_argument("--split", default="validation")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--final-test", action="store_true")
    p.set_defaults(func=cmd_track_synth)

    p = sub.add_parser("negatives")
    p.add_argument("--split", default="validation")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--start", type=int, default=30)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--final-test", action="store_true")
    p.set_defaults(func=cmd_negatives)

    p = sub.add_parser("diagnostics")
    p.add_argument("--split", default="validation")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--start", type=int, default=30)
    p.add_argument("--sequence-length", type=int, default=12)
    p.add_argument("--final-test", action="store_true")
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser("track")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--start", type=int, default=30)
    p.add_argument("--sequence-length", type=int, default=24)
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("final-test")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--clips", type=int, default=11)
    p.add_argument("--sequence-length", type=int, default=12)
    p.set_defaults(func=cmd_final_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VideoContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
