#!/usr/bin/env python3
"""Run a ball-tracking method over the labelled corpus and score it honestly.

Methods are run per *labelled run* of consecutive frames, with a short lead-in
of unlabelled frames so temporal association has something to open a track with.
Only the labelled frames are scored. No method ever sees a label.

    python3 tools/ball_research.py run --split train --method pipeline \
        --out out/ball_research/train_pipeline.json
    python3 tools/ball_research.py report out/ball_research/train_pipeline.json
    python3 tools/ball_research.py overlay --clip-id pexels_6573621 \
        --results out/ball_research/train_pipeline.json --out out/ball_research/qa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostcaddie.video.ball_annotations import annotation_summary, load_annotation  # noqa: E402
from ghostcaddie.video.ball_evaluation import aggregate, evaluate_clip  # noqa: E402
from ghostcaddie.video.ball_pipeline import (  # noqa: E402
    BallParams, _work_scale, generate_candidates, track_ball,
)
from ghostcaddie.video.source_binding import bind_source  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ball_annotations"
SPLIT_PATH = PROJECT_ROOT / "data" / "ball_split.json"

#: Unlabelled frames decoded before each labelled run so association can open a
#: track. Nothing inside the lead-in is ever scored.
LEAD_IN = 8


def _split_clips(split: str) -> list[str]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split == "all":
        return [clip["clip_id"] for clip in manifest["clips"]]
    return [clip["clip_id"] for clip in manifest["clips"] if clip["split"] == split]


def _decode(video: Path, first: int, last: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    frames: list[tuple[int, np.ndarray]] = []
    index = 0
    try:
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= first:
                frames.append((index, frame))
            index += 1
    finally:
        capture.release()
    return frames


# --------------------------------------------------------------------------
# Methods under test
# --------------------------------------------------------------------------

def method_pipeline(frames, indices, params):
    """M1: the full five-stage pipeline."""
    return track_ball(frames, indices, params).as_dict()["observations"]


def method_peak(frames, indices, params):
    """B1: this pipeline's candidate generator with the temporal stages removed.

    Emits the single highest-scoring candidate on every frame. It isolates how
    much of the result comes from detection and how much from association and
    track validation. It uses exactly the same preprocessing as the full
    pipeline -- same working resolution, same colour reference -- so the
    comparison measures the temporal stages and nothing else.
    """
    height, width = frames[0].shape[:2]
    scale = _work_scale(width, height)
    rows = []
    for frame, index in zip(frames, indices):
        if scale < 1.0:
            frame = cv2.resize(frame, (max(16, int(round(width * scale))),
                                       max(16, int(round(height * scale)))),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        candidates = generate_candidates(gray, index, params, None,
                                         frame.astype(np.float32), scale)
        if not candidates:
            rows.append({"frame_index": index, "state": "unavailable", "x": None, "y": None,
                         "radius": None, "score": 0.0, "reason": "no candidate"})
            continue
        best = max(candidates, key=lambda c: c.score)
        rows.append({"frame_index": index, "state": "observed", "x": best.x, "y": best.y,
                     "radius": best.radius, "score": best.score, "reason": ""})
    return rows


def method_heuristic(frames, indices, params):
    """B0: the previously withdrawn heuristic tracker, measured as-is.

    Imported here rather than reimplemented, so the baseline is the real thing
    and not a charitable paraphrase of it.
    """
    from ghostcaddie.video.research_ball import ResearchBallTracker

    tracker = ResearchBallTracker()
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    result = tracker.track(rgb, indices)
    rows = []
    for item in result.items:
        if item.center is None:
            rows.append({"frame_index": item.frame_index, "state": "unavailable", "x": None,
                         "y": None, "radius": None, "score": 0.0,
                         "reason": ",".join(item.warnings)})
        else:
            rows.append({"frame_index": item.frame_index, "state": "observed",
                         "x": float(item.center[0]), "y": float(item.center[1]),
                         "radius": None, "score": float(item.confidence), "reason": ""})
    return rows


METHODS = {"pipeline": method_pipeline, "peak": method_peak, "heuristic": method_heuristic}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    params = BallParams()
    runner = METHODS[args.method]
    clips = [args.clip_id] if args.clip_id else _split_clips(args.split)
    results = []

    for clip_id in clips:
        annotation = load_annotation(ANNOTATION_DIR / f"{clip_id}.json")
        binding = bind_source(PROJECT_ROOT / annotation["video"], annotation["source_sha256"])
        summary = annotation_summary(annotation)
        observations: list[dict] = []
        started = time.time()
        for run in summary["runs"]:
            first = max(0, run["start"] - LEAD_IN)
            decoded = _decode(PROJECT_ROOT / annotation["video"], first, run["end"])
            if not decoded:
                continue
            indices = [index for index, _ in decoded]
            rows = runner([frame for _, frame in decoded], indices, params)
            observations.extend(row for row in rows if run["start"] <= row["frame_index"] <= run["end"])
        elapsed = time.time() - started

        result = evaluate_clip(annotation, observations)
        result["method"] = args.method
        result["seconds"] = elapsed
        result["source_sha256"] = binding["source_sha256"]
        result["observations"] = observations
        results.append(result)
        print(f"{clip_id:22s} P={result['precision']:.3f} R={result['recall']:.3f} "
              f"F1={result['f1']:.3f} FPabs={result['fp_rate_on_absent']:.3f} "
              f"wrong={result['wrong_object_rate_on_visible']:.3f} "
              f"lock={result['longest_static_lock']:3d} ({elapsed:.1f}s)")

    payload = {"method": args.method, "split": args.clip_id or args.split,
               "params": vars(params), "lead_in": LEAD_IN,
               "clips": results, "aggregate": aggregate(results)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n{out}")
    _print_aggregate(payload["aggregate"])
    return 0


def _print_aggregate(totals: dict) -> None:
    print(f"  clips={totals['clips']} scored={totals['scored_frames']} "
          f"(visible {totals['visible_frames']}, absent {totals['not_visible_frames']}, "
          f"ambiguous {totals['ambiguous_frames']})")
    print(f"  precision={totals['precision']:.3f} recall={totals['recall']:.3f} f1={totals['f1']:.3f}")
    median = totals["median_error_px"]
    p90 = totals["p90_error_px"]
    print(f"  median error={median if median is None else round(median, 1)} px  "
          f"p90={p90 if p90 is None else round(p90, 1)} px")
    print(f"  FP on absent={totals['fp_rate_on_absent']:.3f} "
          f"wrong-object on visible={totals['wrong_object_rate_on_visible']:.3f} "
          f"longest static lock={totals['longest_static_lock']} "
          f"identity switches={totals['identity_switches']}")


def _markdown(payload: dict) -> str:
    """The same table, as markdown, so the results doc quotes evidence verbatim."""
    lines = [f"**method `{payload['method']}` — split `{payload['split']}`**", "",
             "| clip | visible | absent | ambiguous | P | R | F1 | median err px | p90 err px |"
             " FP rate on absent | wrong-object rate | longest static lock | longest correct run |"
             " identity switches |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip in payload["clips"]:
        median, p90 = clip["median_error_px"], clip["p90_error_px"]
        lines.append(
            f"| `{clip['clip_id']}` | {clip['visible_frames']} | {clip['not_visible_frames']} |"
            f" {clip['ambiguous_frames']} | {clip['precision']:.3f} | {clip['recall']:.3f} |"
            f" {clip['f1']:.3f} | {'—' if median is None else f'{median:.1f}'} |"
            f" {'—' if p90 is None else f'{p90:.1f}'} | {clip['fp_rate_on_absent']:.3f} |"
            f" {clip['wrong_object_rate_on_visible']:.3f} | {clip['longest_static_lock']} |"
            f" {clip['longest_correct_run']} | {clip['identity_switches']} |")
    totals = payload["aggregate"]
    median, p90 = totals["median_error_px"], totals["p90_error_px"]
    lines.append(
        f"| **pooled** | {totals['visible_frames']} | {totals['not_visible_frames']} |"
        f" {totals['ambiguous_frames']} | **{totals['precision']:.3f}** |"
        f" **{totals['recall']:.3f}** | **{totals['f1']:.3f}** |"
        f" {'—' if median is None else f'{median:.1f}'} |"
        f" {'—' if p90 is None else f'{p90:.1f}'} | **{totals['fp_rate_on_absent']:.3f}** |"
        f" **{totals['wrong_object_rate_on_visible']:.3f}** | {totals['longest_static_lock']} |"
        f" — | {totals['identity_switches']} |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if args.markdown:
        print(_markdown(payload))
        return 0
    print(f"method={payload['method']} split={payload['split']}")
    header = (f"{'clip':22s} {'vis':>4s} {'abs':>4s} {'amb':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} "
              f"{'med':>6s} {'p90':>6s} {'FPabs':>6s} {'wrong':>6s} {'lock':>5s} {'run':>5s} {'sw':>3s}")
    print(header)
    print("-" * len(header))
    for clip in payload["clips"]:
        median, p90 = clip["median_error_px"], clip["p90_error_px"]
        print(f"{clip['clip_id']:22s} {clip['visible_frames']:4d} {clip['not_visible_frames']:4d} "
              f"{clip['ambiguous_frames']:4d} {clip['precision']:6.3f} {clip['recall']:6.3f} "
              f"{clip['f1']:6.3f} {('-' if median is None else f'{median:6.1f}')} "
              f"{('-' if p90 is None else f'{p90:6.1f}')} {clip['fp_rate_on_absent']:6.3f} "
              f"{clip['wrong_object_rate_on_visible']:6.3f} {clip['longest_static_lock']:5d} "
              f"{clip['longest_correct_run']:5d} {clip['identity_switches']:3d}")
    print()
    _print_aggregate(payload["aggregate"])
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    """Consecutive-frame visual QA: prediction, ground truth, error line, state."""
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    clip = next((c for c in payload["clips"] if c["clip_id"] == args.clip_id), None)
    if clip is None:
        raise SystemExit(f"{args.clip_id} not in {args.results}")
    annotation = load_annotation(ANNOTATION_DIR / f"{args.clip_id}.json")
    predicted = {int(o["frame_index"]): o for o in clip["observations"]}
    outcomes = {int(row["frame"]): row["outcome"] for row in clip["per_frame"]}

    runs = annotation_summary(annotation)["runs"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for run in runs:
        decoded = _decode(PROJECT_ROOT / annotation["video"], run["start"], run["end"])
        tiles = []
        for index, frame in decoded:
            tiles.append(_overlay_tile(frame, index, annotation, predicted.get(index),
                                       outcomes.get(index, ""), args.crop, args.scale))
        if not tiles:
            continue
        columns = args.columns
        rows = [np.hstack(tiles[i:i + columns] + [np.zeros_like(tiles[0])] * (columns - len(tiles[i:i + columns])))
                for i in range(0, len(tiles), columns)]
        path = out_dir / f"{args.clip_id}_{run['start']}-{run['end']}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        written.append(path)
    for path in written:
        print(path)
    return 0


_GT = (80, 220, 80)
_PRED = (70, 70, 255)
_LINE = (0, 220, 220)


def _overlay_tile(frame, index, annotation, observation, outcome, crop, scale):
    label = annotation["frames"].get(str(index))
    height, width = frame.shape[:2]
    anchor = None
    if label and label["visible"]:
        anchor = (label["x"], label["y"])
    elif observation and observation.get("x") is not None:
        anchor = (observation["x"], observation["y"])
    if anchor is None:
        anchor = (width / 2, height / 2)
    x0 = int(round(anchor[0] - crop / 2))
    y0 = int(round(anchor[1] - crop / 2))
    tile = np.zeros((crop, crop, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + crop), min(height, y0 + crop)
    if sx1 > sx0 and sy1 > sy0:
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    tile = cv2.resize(tile, (int(crop * scale), int(crop * scale)), interpolation=cv2.INTER_NEAREST)

    def to_tile(point):
        return (int(round((point[0] - x0) * scale)), int(round((point[1] - y0) * scale)))

    truth = pred = None
    if label and label["visible"]:
        truth = to_tile((label["x"], label["y"]))
        radius = int(round(float(label.get("radius", 8)) * scale))
        cv2.circle(tile, truth, max(3, radius), _GT, 2, cv2.LINE_AA)
    if observation and observation.get("x") is not None:
        pred = to_tile((observation["x"], observation["y"]))
        cv2.drawMarker(tile, pred, _PRED, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    if truth and pred:
        cv2.line(tile, truth, pred, _LINE, 1, cv2.LINE_AA)

    state = observation["state"] if observation else "none"
    truth_text = "visible" if (label and label["visible"]) else ("absent" if label else "unlabelled")
    header = np.zeros((28, tile.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"f{index} gt:{truth_text} pred:{state} {outcome}", (4, 19),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, tile])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a method and score it")
    run.add_argument("--split", default="train", choices=["train", "validation", "held_out", "all"])
    run.add_argument("--clip-id", help="score a single clip instead of a whole split")
    run.add_argument("--method", default="pipeline", choices=sorted(METHODS))
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="print a per-clip table from a results file")
    report.add_argument("results")
    report.add_argument("--markdown", action="store_true",
                        help="emit a markdown table for the results document")
    report.set_defaults(func=cmd_report)

    overlay = subparsers.add_parser("overlay", help="write consecutive-frame QA sheets")
    overlay.add_argument("--clip-id", required=True)
    overlay.add_argument("--results", required=True)
    overlay.add_argument("--out", required=True)
    overlay.add_argument("--crop", type=int, default=320)
    overlay.add_argument("--scale", type=float, default=1.0)
    overlay.add_argument("--columns", type=int, default=5)
    overlay.set_defaults(func=cmd_overlay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
