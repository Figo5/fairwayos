#!/usr/bin/env python3
"""Local golf-ball ground-truth annotation utility.

Labelling a 6-pixel object on a 4K frame by eye needs magnification, and
labelling 400 of them needs the magnified frames side by side. So the working
unit here is a *montage*: N consecutive frames, each cropped to the same native
-resolution window around where the ball was last seen, blown up with nearest
-neighbour so individual pixels are visible, and overlaid with a coordinate grid
labelled in **native frame pixels**, plus a red crosshair at the crop centre
whose native coordinate is printed in the tile header. Reading a position as an
offset from that nearby crosshair is far more reliable by eye than reading an
absolute coordinate off a montage that is several thousand pixels wide.

The annotator reads a centre straight off the grid and records it with
``set-tile``, which only needs the sidecar to map a tile back to its frame
index -- there is no crop-origin arithmetic to get wrong. Frames where the ball is gone, occluded or a smear
get ``--not-visible`` or ``--ambiguous`` instead. Nothing in this tool proposes
a position; it only magnifies pixels and records what a person read off them.

Usage:

    python3 tools/ball_annotate.py init --video research_stock/pexels_6573644.mp4 \
        --clip-id pexels_6573644 --annotator "name"
    python3 tools/ball_annotate.py montage --clip-id pexels_6573644 \
        --start 0 --count 12 --size 240 --scale 3 --out /tmp/sheet.png
    python3 tools/ball_annotate.py set-tile --clip-id pexels_6573644 \
        --sidecar /tmp/sheet.json --tile 0 --x 1412 --y 511 --radius 21
    python3 tools/ball_annotate.py set --clip-id pexels_6573644 --frame 130 \
        --not-visible --reason "struck, left frame"
    python3 tools/ball_annotate.py summary --clip-id pexels_6573644
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostcaddie.video.ball_annotations import (  # noqa: E402
    annotation_summary, drop_frame, load_annotation, new_annotation,
    save_annotation, set_frame,
)
from ghostcaddie.video.source_binding import sha256_file  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ball_annotations"

_GRID = (60, 220, 60)
_TEXT = (255, 255, 255)
_CROSS = (60, 60, 255)
_HEADER = 26


def _annotation_path(clip_id: str) -> Path:
    return ANNOTATION_DIR / f"{clip_id}.json"


def _open(video: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    return capture


def _read_frames(video: Path, start: int, count: int, step: int) -> list[tuple[int, np.ndarray]]:
    capture = _open(video)
    try:
        wanted = [start + offset * step for offset in range(count)]
        frames: list[tuple[int, np.ndarray]] = []
        index = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if index != start:  # seek is unreliable on some codecs; decode forward
            capture.release()
            capture = _open(video)
            index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index == wanted[0]:
                frames.append((index, frame))
                wanted.pop(0)
            index += 1
        return frames
    finally:
        capture.release()


def cmd_init(args: argparse.Namespace) -> int:
    video = (PROJECT_ROOT / args.video) if not Path(args.video).is_absolute() else Path(args.video)
    relative = video.resolve().relative_to(PROJECT_ROOT)
    capture = _open(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    document = new_annotation(
        clip_id=args.clip_id, video=str(relative), source_sha256=sha256_file(video),
        width=width, height=height, fps=fps, annotator=args.annotator, notes=args.notes,
    )
    path = _annotation_path(args.clip_id)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; pass --force to overwrite and lose labels")
    save_annotation(path, document, project_root=PROJECT_ROOT)
    print(f"{path}: {width}x{height} @ {fps:.4f} fps, sha256 {document['source_sha256'][:12]}…")
    return 0


def _default_centre(document: dict, start: int) -> tuple[float, float] | None:
    previous = [
        (int(key), entry) for key, entry in document["frames"].items()
        if entry["visible"] and int(key) <= start
    ]
    if not previous:
        return None
    _, entry = max(previous, key=lambda item: item[0])
    return entry["x"], entry["y"]


def _draw_tile(frame: np.ndarray, *, frame_index: int, origin: tuple[int, int],
               size: int, scale: float, label: str) -> np.ndarray:
    x0, y0 = origin
    height, width = frame.shape[:2]
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + size), min(height, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    out_size = int(round(size * scale))
    tile = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Grid labels are NATIVE frame coordinates, not crop-local: the annotator
    # reads a number straight off the image and hands it to ``set-tile``, with
    # no origin arithmetic to get wrong.
    step = max(1, int(round(size / 8)))
    for value in range(0, size + 1, step):
        position = min(int(round(value * scale)), out_size - 1)
        cv2.line(tile, (position, 0), (position, out_size - 1), _GRID, 1)
        cv2.line(tile, (0, position), (out_size - 1, position), _GRID, 1)
        cv2.putText(tile, str(x0 + value), (min(position + 2, out_size - 30), 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, _GRID, 1, cv2.LINE_AA)
        if value:
            cv2.putText(tile, str(y0 + value), (2, min(position - 2, out_size - 2)),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, _GRID, 1, cv2.LINE_AA)

    # A crosshair at the crop centre, whose native coordinate is printed in the
    # header. Absolute positions are hard to read consistently across a wide
    # montage -- an offset from a nearby fixed mark is not.
    centre = int(round((size / 2) * scale))
    arm = max(6, int(round(4 * scale)))
    cv2.line(tile, (centre - arm, centre), (centre + arm, centre), _CROSS, 1, cv2.LINE_AA)
    cv2.line(tile, (centre, centre - arm), (centre, centre + arm), _CROSS, 1, cv2.LINE_AA)

    header = np.zeros((_HEADER, out_size, 3), dtype=np.uint8)
    cv2.putText(header, label, (4, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, _TEXT, 1, cv2.LINE_AA)
    return np.vstack([header, tile])


def cmd_montage(args: argparse.Namespace) -> int:
    document = load_annotation(_annotation_path(args.clip_id), project_root=PROJECT_ROOT)
    video = PROJECT_ROOT / document["video"]
    size = args.size or min(document["width"], document["height"])
    if args.cx is not None and args.cy is not None:
        centre = (float(args.cx), float(args.cy))
    else:
        centre = _default_centre(document, args.start) or (
            document["width"] / 2.0, document["height"] / 2.0)
    origin = (int(round(centre[0] - size / 2)), int(round(centre[1] - size / 2)))

    # A fast-moving ball outruns any single crop window. ``--centres`` lets the
    # annotator supply a coarse per-frame guess (read off a wide montage) so the
    # fine pass can crop tightly around each frame separately and still be read
    # at high magnification.
    per_frame: dict[int, tuple[float, float]] = {}
    if args.centres:
        raw = json.loads(Path(args.centres).read_text(encoding="utf-8"))
        rows = raw.items() if isinstance(raw, dict) else ((row[0], row[1:]) for row in raw)
        per_frame = {int(key): (float(value[0]), float(value[1])) for key, value in rows}

    frames = _read_frames(video, args.start, args.count, args.step)
    if not frames:
        raise SystemExit(f"no frames decoded from {video} at {args.start}")

    columns = args.columns
    rows = (len(frames) + columns - 1) // columns
    tiles = []
    sidecar_tiles = []
    for position, (frame_index, frame) in enumerate(frames):
        if frame_index in per_frame:
            spot = per_frame[frame_index]
            origin = (int(round(spot[0] - size / 2)), int(round(spot[1] - size / 2)))
        entry = document["frames"].get(str(frame_index))
        if entry is None:
            state = ""
        elif entry["visible"]:
            state = f" [{entry['x']:.0f},{entry['y']:.0f}]"
        else:
            state = " [not visible]"
        if entry is not None and entry["ambiguous"]:
            state += " ?"
        label = (f"t{position} f{frame_index} +({origin[0] + size // 2},"
                 f"{origin[1] + size // 2}) {int(round(size / 8))}px/cell{state}")
        tiles.append(_draw_tile(frame, frame_index=frame_index, origin=origin,
                                size=size, scale=args.scale, label=label))
        sidecar_tiles.append({"tile": position, "frame_index": frame_index,
                              "origin_x": origin[0], "origin_y": origin[1],
                              "size": size, "scale": args.scale})

    tile_shape = tiles[0].shape
    blank = np.zeros(tile_shape, dtype=np.uint8)
    grid = []
    for row in range(rows):
        row_tiles = tiles[row * columns:(row + 1) * columns]
        row_tiles += [blank] * (columns - len(row_tiles))
        grid.append(np.hstack(row_tiles))
    montage = np.vstack(grid)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "clip_id": args.clip_id, "video": document["video"],
        "source_sha256": document["source_sha256"], "tiles": sidecar_tiles,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{out} ({montage.shape[1]}x{montage.shape[0]}), sidecar {sidecar}")
    print(f"tiles: frames {[index for index, _ in frames]}")
    return 0


def _label_kwargs(args: argparse.Namespace, x: float | None, y: float | None) -> dict:
    if args.not_visible:
        if not args.reason:
            raise SystemExit("--not-visible requires --reason")
        label = {"visible": False, "reason": args.reason}
    else:
        if x is None or y is None:
            raise SystemExit("a visible label requires coordinates")
        label = {"visible": True, "x": float(x), "y": float(y)}
        if args.radius is not None:
            label["radius"] = float(args.radius)
        if args.reason:
            label["reason"] = args.reason
    if args.ambiguous:
        label["ambiguous"] = True
    return label


def cmd_set(args: argparse.Namespace) -> int:
    path = _annotation_path(args.clip_id)
    document = load_annotation(path, project_root=PROJECT_ROOT)
    document = set_frame(document, args.frame, **_label_kwargs(args, args.x, args.y))
    save_annotation(path, document, project_root=PROJECT_ROOT)
    print(f"frame {args.frame}: {document['frames'][str(args.frame)]}")
    return 0


def cmd_set_tile(args: argparse.Namespace) -> int:
    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    path = _annotation_path(args.clip_id)
    document = load_annotation(path, project_root=PROJECT_ROOT)
    if sidecar["source_sha256"] != document["source_sha256"]:
        raise SystemExit("sidecar was built from different source bytes than the annotation")
    tiles = {int(tile["tile"]): tile for tile in sidecar["tiles"]}
    if args.tile not in tiles:
        raise SystemExit(f"tile {args.tile} not in {args.sidecar}")
    tile = tiles[args.tile]
    document = set_frame(document, int(tile["frame_index"]),
                         **_label_kwargs(args, args.x, args.y))
    save_annotation(path, document, project_root=PROJECT_ROOT)
    print(f"frame {tile['frame_index']}: {document['frames'][str(tile['frame_index'])]}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Apply many labels at once from a JSON list, for montage-rate labelling.

    Each item is ``{"tile": N, ...}`` (resolved through the sidecar) or
    ``{"frame": N, ...}`` (native coordinates), plus the same label fields the
    single-frame commands accept.
    """
    items = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("labels file must contain a JSON list")
    path = _annotation_path(args.clip_id)
    document = load_annotation(path, project_root=PROJECT_ROOT)
    tiles: dict[int, dict] = {}
    if args.sidecar:
        sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
        if sidecar["source_sha256"] != document["source_sha256"]:
            raise SystemExit("sidecar was built from different source bytes than the annotation")
        tiles = {int(tile["tile"]): tile for tile in sidecar["tiles"]}

    for item in items:
        if "tile" in item:
            if int(item["tile"]) not in tiles:
                raise SystemExit(f"tile {item['tile']} needs a --sidecar that contains it")
            frame_index = int(tiles[int(item["tile"])]["frame_index"])
        else:
            frame_index = int(item["frame"])
        if item.get("visible", "x" in item):
            label = {"visible": True, "x": float(item["x"]), "y": float(item["y"])}
            if "radius" in item:
                label["radius"] = float(item["radius"])
        else:
            if not item.get("reason"):
                raise SystemExit(f"frame {frame_index}: a not-visible label requires a reason")
            label = {"visible": False, "reason": str(item["reason"])}
        if item.get("ambiguous"):
            label["ambiguous"] = True
        if item.get("reason") and label["visible"]:
            label["reason"] = str(item["reason"])
        document = set_frame(document, frame_index, **label)

    save_annotation(path, document, project_root=PROJECT_ROOT)
    summary = annotation_summary(document)
    print(f"{path}: {summary['labeled_frames']} labelled "
          f"({summary['visible_frames']} visible, {summary['not_visible_frames']} not visible, "
          f"{summary['ambiguous_frames']} ambiguous)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Draw every label back onto its own frame, so it can be checked by eye.

    Recording a centre and trusting it is how a tee ends up in a file as a golf
    ball. This renders a sampled frame from each labelled run with the stored
    circle drawn on it; anything that is not sitting on the ball is a label bug,
    not a tracker bug.
    """
    document = load_annotation(_annotation_path(args.clip_id), project_root=PROJECT_ROOT)
    video = PROJECT_ROOT / document["video"]
    runs = annotation_summary(document)["runs"]
    wanted: list[int] = []
    for run in runs:
        length = run["length"]
        step = max(1, length // max(1, args.per_run))
        wanted.extend(range(run["start"], run["end"] + 1, step))
    wanted = sorted(set(wanted))[:args.columns * 6]

    capture = _open(video)
    tiles = []
    index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index == wanted[0]:
                tiles.append(_verify_tile(frame, index, document, args.crop, args.scale))
                wanted.pop(0)
            index += 1
    finally:
        capture.release()
    if not tiles:
        raise SystemExit("no labelled frames decoded")
    columns = args.columns
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + columns] + [blank] * (columns - len(tiles[i:i + columns])))
            for i in range(0, len(tiles), columns)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)
    return 0


def _verify_tile(frame, index, document, crop, scale):
    entry = document["frames"].get(str(index))
    height, width = frame.shape[:2]
    side = max(64, int(round(crop * scale)))
    if entry and entry["visible"]:
        # Zoom on the recorded centre: a label is only checkable at the scale it
        # was read at.
        x0 = int(round(entry["x"] - crop / 2))
        y0 = int(round(entry["y"] - crop / 2))
        tile = np.zeros((crop, crop, 3), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(width, x0 + crop), min(height, y0 + crop)
        if sx1 > sx0 and sy1 > sy0:
            tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_NEAREST)
        ratio = side / float(crop)
    else:
        # A not-visible claim is about the whole frame, so show the whole frame.
        fit = min(side / width, side / height)
        resized = cv2.resize(frame, (max(1, int(width * fit)), max(1, int(height * fit))))
        tile = np.zeros((side, side, 3), dtype=np.uint8)
        tile[:resized.shape[0], :resized.shape[1]] = resized
        x0 = y0 = 0
        ratio = fit
    if entry and entry["visible"]:
        point = (int(round((entry["x"] - x0) * ratio)), int(round((entry["y"] - y0) * ratio)))
        cv2.circle(tile, point, max(3, int(round(float(entry.get("radius", 8)) * ratio))),
                   (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(tile, point, (0, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        state = f"visible ({entry['x']:.0f},{entry['y']:.0f}) r{entry.get('radius', 0):.0f}"
    elif entry:
        state = "NOT VISIBLE - whole frame"
    else:
        state = "unlabelled"
    if entry and entry["ambiguous"]:
        state += " AMBIGUOUS"
    header = np.zeros((26, tile.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"f{index} {state}", (4, 18), cv2.FONT_HERSHEY_PLAIN, 0.95,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, tile])


def cmd_unset(args: argparse.Namespace) -> int:
    path = _annotation_path(args.clip_id)
    document = drop_frame(load_annotation(path, project_root=PROJECT_ROOT), args.frame)
    save_annotation(path, document, project_root=PROJECT_ROOT)
    print(f"frame {args.frame} removed")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    document = load_annotation(_annotation_path(args.clip_id), project_root=PROJECT_ROOT)
    print(json.dumps(annotation_summary(document), indent=2))
    return 0


def _add_label_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--not-visible", action="store_true")
    parser.add_argument("--ambiguous", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--radius", type=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create an annotation file for a clip")
    init.add_argument("--video", required=True)
    init.add_argument("--clip-id", required=True)
    init.add_argument("--annotator", default="local-annotator")
    init.add_argument("--notes", default="")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    montage = subparsers.add_parser("montage", help="render magnified consecutive frames")
    montage.add_argument("--clip-id", required=True)
    montage.add_argument("--start", type=int, required=True)
    montage.add_argument("--count", type=int, default=12)
    montage.add_argument("--step", type=int, default=1)
    montage.add_argument("--size", type=int)
    montage.add_argument("--scale", type=float, default=3.0)
    montage.add_argument("--columns", type=int, default=4)
    montage.add_argument("--cx", type=float)
    montage.add_argument("--cy", type=float)
    montage.add_argument("--centres", help="JSON {frame: [cx, cy]} of coarse per-frame guesses")
    montage.add_argument("--out", required=True)
    montage.set_defaults(func=cmd_montage)

    setter = subparsers.add_parser("set", help="label one frame in native coordinates")
    setter.add_argument("--clip-id", required=True)
    setter.add_argument("--frame", type=int, required=True)
    setter.add_argument("--x", type=float)
    setter.add_argument("--y", type=float)
    _add_label_flags(setter)
    setter.set_defaults(func=cmd_set)

    tile = subparsers.add_parser("set-tile", help="label one frame in montage-tile coordinates")
    tile.add_argument("--clip-id", required=True)
    tile.add_argument("--sidecar", required=True)
    tile.add_argument("--tile", type=int, required=True)
    tile.add_argument("--x", type=float, help="native frame x, read off the tile grid")
    tile.add_argument("--y", type=float, help="native frame y, read off the tile grid")
    _add_label_flags(tile)
    tile.set_defaults(func=cmd_set_tile)

    batch = subparsers.add_parser("batch", help="apply a JSON list of labels")
    batch.add_argument("--clip-id", required=True)
    batch.add_argument("--labels", required=True)
    batch.add_argument("--sidecar")
    batch.set_defaults(func=cmd_batch)

    verify = subparsers.add_parser("verify", help="draw stored labels back onto their frames")
    verify.add_argument("--clip-id", required=True)
    verify.add_argument("--out", required=True)
    verify.add_argument("--per-run", type=int, default=4)
    verify.add_argument("--crop", type=int, default=260)
    verify.add_argument("--scale", type=float, default=1.5)
    verify.add_argument("--columns", type=int, default=5)
    verify.set_defaults(func=cmd_verify)

    unset = subparsers.add_parser("unset", help="remove one frame label")
    unset.add_argument("--clip-id", required=True)
    unset.add_argument("--frame", type=int, required=True)
    unset.set_defaults(func=cmd_unset)

    summary = subparsers.add_parser("summary", help="print label counts and runs")
    summary.add_argument("--clip-id", required=True)
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
