"""Cache body keypoints for v3 candidate features.

Why this is a separate pass with its own interpreter: MoveNet needs
``ai_edge_litert``, which is installed in ``.venv``; the v3 models need torch,
which is installed in ``.venv-video-ai``. Nothing in this repository has both, and
that separation is deliberate (``RuntimeSpec(target='body', ...,
purpose='MoveNet TFLite: no torch, no pickle surface')``). So body runs first and
writes a cache; the ranking pass reads the cache and never imports a pose model.

**What this output is, and is not.** This is whole-frame single-pose MoveNet. It
is not "the golfer" -- the canonical pipeline is explicit that whole-frame
single-pose output may never be called that (``docs/video-analysis-user-guide.md``
§6), and the cache records ``attribution: "whole_frame_singlepose"`` on every
frame so a consumer cannot forget. For v3 that is enough, because body is used as
*features* -- how far is this candidate from the hands, the feet, the torso; does
it move with the torso -- and never as an exclusion region. The v2 pass never ran
a pose model at all, which is why every ``body_lock_rate`` it reported is zero by
absence of data.

Usage:
    .venv/bin/python3 tools/ball_v3_body_cache.py --corpus v3_eval
    .venv/bin/python3 tools/ball_v3_body_cache.py --corpus legacy --stride 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "models" / "movenet-singlepose-lightning-v3.tflite"
CACHE = ROOT / "out" / "ball_tracker_v3" / "body_cache"


def clips_for(corpus: str) -> list[tuple[str, Path]]:
    """(clip_id, media path) for a named corpus. Media is local and gitignored."""
    if corpus == "v3_eval":
        payload = json.loads(
            (ROOT / "data/ball_dataset_v3_eval/sources.json").read_text("utf-8"))
        return [(c["clip_id"], ROOT / c["media_path"]) for c in payload["clips"]]
    if corpus == "v2":
        payload = json.loads(
            (ROOT / "data/ball_dataset_v2/sources.json").read_text("utf-8"))
        return [(c["clip_id"], ROOT / c["media_path"]) for c in payload["clips"]]
    if corpus == "legacy":
        out = []
        for path in sorted((ROOT / "data/ball_annotations").glob("*.json")):
            payload = json.loads(path.read_text("utf-8"))
            out.append((payload["clip_id"], ROOT / payload["video"]))
        return out
    raise SystemExit(f"unknown corpus {corpus!r}")


def run_clip(net, video: Path, *, stride: int, limit: int) -> dict:
    import cv2
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: dict[str, dict] = {}
    index = 0
    started = time.time()
    while index < total and len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            # Whole frame, letterboxed by the model wrapper. No subject box: a
            # subject box is an identity claim, and this cache makes none.
            points, letterbox = net.infer(frame, (0, 0))
            visible = [p for p in points if p["visible"]]
            frames[str(index)] = {
                "keypoints": {p["name"]: [p["x"], p["y"], p["score"]] for p in points},
                "visible_keypoints": len(visible),
                "attribution": "whole_frame_singlepose",
            }
        index += 1
    capture.release()
    return {"width": width, "height": height, "frames_in_clip": total,
            "stride": stride, "frames": frames,
            "seconds": round(time.time() - started, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="v3_eval",
                        choices=("v3_eval", "v2", "legacy"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stride", type=int, default=1,
                        help="run every Nth frame")
    parser.add_argument("--limit", type=int, default=900,
                        help="max frames cached per clip (processing stays bounded)")
    parser.add_argument("--out", type=Path, default=CACHE)
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"pose model not found: {args.model}\n"
              "Nothing is downloaded automatically; see `python3 -m ghostcaddie models`.",
              file=sys.stderr)
        return 2
    try:
        from ghostcaddie.video.canonical_worker import MoveNet
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"cannot import the pose worker: {exc}", file=sys.stderr)
        return 2
    try:
        net = MoveNet(str(args.model))
    except Exception as exc:
        print(f"cannot load MoveNet ({type(exc).__name__}: {exc}).\n"
              "This command needs the body interpreter: .venv/bin/python3",
              file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or ())
    done = 0
    for clip_id, video in clips_for(args.corpus):
        if wanted and clip_id not in wanted:
            continue
        if not video.exists():
            print(f"{clip_id}: media missing at {video}", file=sys.stderr)
            continue
        destination = args.out / f"{clip_id}.json"
        if destination.exists():
            print(f"{clip_id}: cached")
            done += 1
            continue
        result = run_clip(net, video, stride=args.stride, limit=args.limit)
        result.update({"clip_id": clip_id, "model": args.model.name,
                       "origin": "MODEL_INFERENCE",
                       "attribution": "whole_frame_singlepose",
                       "not_ground_truth": True})
        destination.write_text(json.dumps(result) + "\n")
        usable = sum(1 for f in result["frames"].values()
                     if f["visible_keypoints"] >= 8)
        print(f"{clip_id}: {len(result['frames'])} frames cached, "
              f"{usable} with >=8 visible keypoints, {result['seconds']}s")
        done += 1
    print(f"cached {done} clips into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
