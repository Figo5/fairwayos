"""Reusable, tested MIL comparison overlay + reconciliation helpers.

Extracted from the untested one-off ``out/render_mil_mp4s.py`` so the
FairwayOS MIL evidence pipeline has verified source code behind it.

Coordinate transform: the one-off resized sources to 1200 px wide and then
drew native-coordinate MIL boxes UNSCALED while baseline points WERE scaled
(spatially inconsistent, wrong on portrait sources). Everything here maps
both boxes and points from native (src_w, src_h) to output (out_w, out_h)
through the SAME per-axis scale factors, landscape or portrait, with rounded
output dims handled.

Timing: the tracker defaulted to 30 fps; clips are 60/25 fps. Helpers emit
three DISTINCT time bases per frame:
- ``timestamp``        = source time      = src_idx / fps
- ``timestamp_window`` = window-relative  = (src_idx - window_start) / fps
- ``timestamp_seed``   = since seed       = (src_idx - seed_frame) / fps
Playback time for an encoded window equals timestamp_window (1 fps == 1 fps).

Rendering (``render_flat_frame``) is truthfulness-first:
- raw tracker output vs AI-reviewed acceptance are separate layers. A frame
  whose tracker state says tracked/reacquired but whose AI-review verdict is
  on_body/on_shaft/on_ground is displayed REJECTED (red box + red border),
  never green.
- no trails, ever: nothing is drawn for unavailable/ended frames and no
  connector can exist between marks because each frame is drawn independently.
- seed frame drawn distinctly (magenta).
"""
from __future__ import annotations

import math
import os
import subprocess
from typing import Optional, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------- flags
RESEARCH_FLAGS = {"research_only": True, "ground_truth": False,
                  "production_eligible": False}

# ------------------------------------------------------------- color palette
# BGR
COLOR_SEED = (255, 0, 255)           # magenta - seed frame is distinct
COLOR_TRACKED = (0, 255, 0)          # green  - raw tracked (pre-review layer)
COLOR_REACQUIRED = (0, 255, 255)     # yellow - raw reacquired
COLOR_REJECTED = (0, 0, 255)         # red    - AI-reviewed as off-head
COLOR_BASELINE = (255, 0, 0)         # blue   - baseline observed point
COLOR_UNAVAILABLE = (160, 160, 160)  # gray   - status text only
COLOR_ENDED = (0, 128, 0)            # dark green - status text only
QA_BORDER_OK = (0, 255, 0)
QA_BORDER_REJECT = (0, 0, 255)

# raw state -> drawn box color (None = draw nothing, status text only)
RAW_STATE_COLOR = {
    "seed": COLOR_SEED,
    "tracked": COLOR_TRACKED,
    "reacquired": COLOR_REACQUIRED,
}

# raw MIL state x AI verdict -> display classification
DISPLAY_STATES = (
    "seed", "on_clubhead", "rejected", "unclear", "unreviewed",
    "unavailable", "ended",
)
_REJECT_VERDICTS = {"on_body", "on_shaft", "on_ground"}
_VISIBLE_RAW = {"tracked", "reacquired"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.55
TEXT_THICK = 1


# ------------------------------------------------------- coordinate transform
def scale_factors(src_w, src_h, out_w, out_h) -> tuple:
    """Per-axis scale factors native -> output."""
    for name, v in (("src_w", src_w), ("src_h", src_h), ("out_w", out_w),
                    ("out_h", out_h)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    return (float(out_w) / float(src_w), float(out_h) / float(src_h))


def transform_box(bbox, src_w, src_h, out_w, out_h, *, round_output=False):
    """Map a native-coordinate (x, y, w, h) box to output coordinates.

    Landscape and portrait treated identically: the same per-axis factors
    apply to the box origin AND size, so the box center lands exactly on the
    transformed center point (same scale as the baseline point transform).
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be (x, y, w, h)")
    vals = tuple(bbox)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in vals):
        raise ValueError("bbox values must be finite numbers")
    x, y, w, h = (float(v) for v in vals)
    if w <= 0 or h <= 0:
        raise ValueError("bbox width/height must be positive")
    sx, sy = scale_factors(src_w, src_h, out_w, out_h)
    ob = (x * sx, y * sy, w * sx, h * sy)
    if round_output:
        ob = (int(round(ob[0])), int(round(ob[1])), int(round(ob[2])),
              int(round(ob[3])))
    return ob


def transform_point(point, src_w, src_h, out_w, out_h, *, round_output=False):
    """Map a native-coordinate (x, y) point to output coordinates."""
    if point is None or len(point) != 2:
        raise ValueError("point must be (x, y)")
    vals = tuple(point)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in vals):
        raise ValueError("point values must be finite numbers")
    if not (0.0 <= float(vals[0]) <= float(src_w)
            and 0.0 <= float(vals[1]) <= float(src_h)):
        raise ValueError("point must lie within the source frame")
    sx, sy = scale_factors(src_w, src_h, out_w, out_h)
    p = (float(point[0]) * sx, float(point[1]) * sy)
    if round_output:
        p = (int(round(p[0])), int(round(p[1])))
    return p


# ------------------------------------------------------------------- timing
def source_timestamp(source_frame_index: int, fps: float) -> float:
    """Source time of a frame: source_frame_index / fps (NOT offset/30)."""
    if isinstance(source_frame_index, bool) or not isinstance(source_frame_index, int) \
            or source_frame_index < 0:
        raise ValueError("source_frame_index must be a non-negative int")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) \
            or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    return source_frame_index / float(fps)


def timestamp_at_source_index(source_frame_index: int, window_start: int,
                              fps: float) -> float:
    """Window-relative timestamp: (src_idx - window_start) / fps."""
    if isinstance(window_start, bool) or not isinstance(window_start, int) \
            or window_start < 0:
        raise ValueError("window_start must be a non-negative int")
    return source_timestamp(source_frame_index - window_start, fps)


def _validate_fps(fps) -> None:
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) \
            or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")


def _timestamp_triplet(source_frame_index: int, window_start: int,
                       seed_frame: Optional[int], fps: float) -> dict:
    """Three DISTINCT time bases. Seed-relative time is legitimately NEGATIVE
    for pre-seed frames (rows before the seed inside the window), so it is
    computed directly (fps is validated via ``source_timestamp`` above)."""
    _validate_fps(fps)
    f = float(fps)
    ts_src = source_timestamp(source_frame_index, fps)
    ts_win = timestamp_at_source_index(source_frame_index, window_start, fps)
    ts_seed = ((source_frame_index - seed_frame) / f
               if seed_frame is not None else None)
    return {"timestamp": ts_src, "timestamp_window": ts_win,
            "timestamp_seed": ts_seed}


def correct_perframe_timestamps(rows: Sequence[dict], *, window_start: int,
                                fps: float, seed_frame: Optional[int] = None) -> list:
    """Return NEW rows with correct timestamps in three distinct bases.

    ``timestamp``        = source time      = src_idx / fps
    ``timestamp_window`` = window-relative  = (src_idx - window_start) / fps
    ``timestamp_seed``   = since seed       = (src_idx - seed_frame) / fps

    Rows must be contiguous (source_frame_index step 1) and start at
    ``window_start`` when the window begins at the seed; rows starting later
    are allowed only when ``seed_frame`` is inside the row range (pre-seed
    window frames have no tracker row). Input rows are not mutated.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    idxs = [r["source_frame_index"] for r in rows]
    if any(b != a + 1 for a, b in zip(idxs, idxs[1:])):
        raise ValueError("per-frame rows must be contiguous")
    if idxs[0] != window_start and seed_frame not in idxs:
        raise ValueError(
            f"rows start at {idxs[0]} but window_start={window_start} and "
            f"seed_frame={seed_frame} is not among rows")
    out = []
    for r in rows:
        nr = dict(r)
        nr.update(_timestamp_triplet(r["source_frame_index"], window_start,
                                     seed_frame, fps))
        out.append(nr)
    return out


# ---------------------------------------------------------------- rendering
def classify_mil_display(raw_state: str, has_bbox: bool,
                         qa_verdict: Optional[str]) -> str:
    """Classify a MIL frame row for display.

    Raw state and AI-reviewed acceptance are separate layers: a tracked frame
    with verdict on_body/on_shaft/on_ground is REJECTED (never accepted just
    because the tracker state says tracked). Unavailable/ended stay put. The
    seed is its own class. A visible frame never AI-reviewed is unreviewed.
    """
    if raw_state in ("unavailable", "ended"):
        return raw_state
    if raw_state == "seed":
        return "seed"
    if not has_bbox:
        return "unavailable"
    if qa_verdict is None:
        return "unreviewed"
    if qa_verdict == "on_clubhead":
        return "on_clubhead"
    if qa_verdict in _REJECT_VERDICTS:
        return "rejected"
    return "unclear"


def _put(img, text, org, color, scale=TEXT_SCALE, thick=TEXT_THICK):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _draw_box(img, bbox_out, color, thick=2):
    x, y, w, h = (int(round(v)) for v in bbox_out)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thick, cv2.LINE_AA)


def render_flat_frame(frame_bgr, *, mil_row, baseline_row, native_w, native_h,
                      out_w, out_h, source_frame_index, window_start, fps,
                      mil_qa_verdict=None, baseline_qa_verdict=None,
                      frame_number=None) -> np.ndarray:
    """Render ONE annotated flat (no trails) output frame.

    ``mil_row`` / ``baseline_row`` are the per-frame records (native coords).
    Unavailable/ended rows draw status text only -- never a mark, never a
    trail. A rejected (AI-reviewed off-head) frame gets a red box + full red
    border. An AI-reviewed on_clubhead frame gets a green border. The seed
    frame is drawn distinctly (magenta box + SEED label). Frames that decode
    short of the native size are upscaled so native coordinates still map
    correctly (short-decode sources).
    """
    fh, fw = frame_bgr.shape[:2]
    # guard: rows are in native coordinates of the SOURCE; if the decoded
    # frame differs from the declared native size (short decode), normalize
    # to the declared native grid first so the transform stays correct.
    if (fw, fh) != (native_w, native_h):
        frame_bgr = cv2.resize(frame_bgr, (native_w, native_h),
                               interpolation=cv2.INTER_AREA)
    img = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    # 1px black border so full-frame QA rectangles stay fully visible
    img[0, :] = (0, 0, 0)
    img[-1, :] = (0, 0, 0)
    img[:, 0] = (0, 0, 0)
    img[:, -1] = (0, 0, 0)

    # ---- MIL layer ------------------------------------------------------
    display = None
    if mil_row is not None:
        state = mil_row.get("state") or "unavailable"
        bbox = mil_row.get("bbox")
        display = classify_mil_display(state, bbox is not None, mil_qa_verdict)
        if display in ("seed", "on_clubhead", "unreviewed", "unclear",
                       "rejected"):
            color = {"seed": COLOR_SEED, "rejected": COLOR_REJECTED}.get(
                display, RAW_STATE_COLOR.get(state, COLOR_TRACKED))
            box_out = transform_box(bbox, native_w, native_h, out_w, out_h)
            _draw_box(img, box_out, color,
                      thick=3 if display == "seed" else 2)
            tag = {"seed": "SEED",
                   "rejected": "REJECTED(off-head)",
                   "on_clubhead": "MIL on-head",
                   "unreviewed": "MIL unreviewed",
                   "unclear": "MIL unclear"}[display]
            bx, by = int(round(box_out[0])), int(round(box_out[1]))
            if by - 14 >= 12:
                _put(img, tag, (bx, by - 8), color)
            else:
                _put(img, tag, (bx, by + int(round(box_out[3])) + 18), color)
        elif display == "unavailable":
            warn = mil_row.get("warning") or ""
            _put(img, f"MIL unavailable ({warn})"[:80], (12, 60),
                 COLOR_UNAVAILABLE)
        elif display == "ended":
            warn = mil_row.get("warning") or "track_ended"
            _put(img, f"MIL ended ({warn})"[:80], (12, 60), COLOR_ENDED)
        # seed corner tag
        if display == "seed":
            _put(img, "SEED FRAME", (12, 90), COLOR_SEED, scale=0.7)

    # ---- baseline layer -------------------------------------------------
    if baseline_row is not None:
        bstate = baseline_row.get("state")
        bpt = baseline_row.get("point")
        if bstate == "observed" and bpt:
            px, py = transform_point(bpt, native_w, native_h, out_w, out_h)
            # raw verdicts on_body/on_shaft/on_ground are REJECTED: the
            # point is drawn red, never accepted blue
            bcolor = QA_BORDER_REJECT if baseline_qa_verdict in _REJECT_VERDICTS \
                else COLOR_BASELINE
            cv2.circle(img, (int(round(px)), int(round(py))), 8,
                       bcolor, 2, cv2.LINE_AA)
        elif bstate == "unavailable":
            _put(img, "baseline unavailable", (12, 78), COLOR_UNAVAILABLE)

    # ---- QA review borders (AI-reviewed layer) --------------------------
    if mil_qa_verdict == "on_clubhead" and display in ("seed", "on_clubhead"):
        cv2.rectangle(img, (1, 1), (out_w - 2, out_h - 2), QA_BORDER_OK, 4)
    elif mil_qa_verdict in ("on_body", "on_shaft", "on_ground") and \
            display == "rejected":
        cv2.rectangle(img, (1, 1), (out_w - 2, out_h - 2), QA_BORDER_REJECT, 4)

    # ---- legend + source frame label ------------------------------------
    _put(img, "MIL box (green=tracked yellow=reacquired magenta=seed red=QA-rejected)",
         (12, 22), (200, 200, 255), scale=0.5)
    _put(img, "baseline point: blue circle", (12, 40), (200, 200, 255),
         scale=0.5)
    ts_src = source_timestamp(source_frame_index, fps)
    ts_win = timestamp_at_source_index(source_frame_index, window_start, fps)
    label = f"src {source_frame_index}  t_src={ts_src:.3f}s  t_win={ts_win:.3f}s"
    if frame_number is not None:
        label = f"#{int(frame_number):03d} {label}"
    _put(img, label, (12, out_h - 14), (255, 255, 255), scale=0.6)
    return img


def render_window(clip_path, window: Sequence[int], *, mil_rows, out_dir,
                  fps, prefix="mil", baseline_rows=None,
                  mil_qa_verdicts=None, baseline_qa_verdicts=None,
                  target_width=1200, round_output=True,
                  seed_frame=None) -> dict:
    """Render the COMPLETE window [start, end] inclusive -- every frame,
    including unavailable/ended ones -- as numbered PNGs.

    Never skips a frame; if the source cannot decode a window frame this
    raises instead of silently truncating. Returns a result dict with
    frame_count (== end-start+1), window, frame_files and per-frame display
    classifications.
    """
    start, end = int(window[0]), int(window[1])
    if end < start:
        raise ValueError("window end must be >= start")
    expected = end - start + 1
    mil_by_idx = {r["source_frame_index"]: r for r in (mil_rows or [])}
    base_by_idx = ({r["source_frame_index"]: r for r in baseline_rows}
                   if baseline_rows else {})
    mil_qa = mil_qa_verdicts or {}
    base_qa = baseline_qa_verdicts or {}

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = int(target_width)
    out_h = int(round(src_h * (out_w / src_w))) if round_output else \
        int(src_h * out_w / src_w)
    # H.264/yuv420p requires even dimensions; losing at most 1 row keeps the
    # box/point transform consistent (it uses the same out_h as the resize).
    if out_h % 2:
        out_h -= 1

    os.makedirs(out_dir, exist_ok=True)
    frame_files = []
    displays = {}
    n = 0
    try:
        for fi in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"source decode failed at frame {fi} (window "
                    f"{start}-{end}); refusing to truncate the window")
            out = render_flat_frame(
                frame, mil_row=mil_by_idx.get(fi),
                baseline_row=base_by_idx.get(fi), native_w=src_w,
                native_h=src_h, out_w=out_w, out_h=out_h,
                source_frame_index=fi, window_start=start, fps=fps,
                mil_qa_verdict=mil_qa.get(str(fi), mil_qa.get(fi)),
                baseline_qa_verdict=base_qa.get(str(fi), base_qa.get(fi)),
                frame_number=n + 1)
            path = os.path.join(out_dir, f"frame_{n + 1:06d}.png")
            if not cv2.imwrite(path, out):
                raise RuntimeError(f"failed writing {path}")
            frame_files.append(path)
            m = mil_by_idx.get(fi)
            if m is not None:
                displays[fi] = classify_mil_display(
                    m.get("state") or "unavailable", m.get("bbox") is not None,
                    mil_qa.get(str(fi), mil_qa.get(fi)))
            n += 1
    finally:
        cap.release()
    if n != expected:
        raise RuntimeError(
            f"rendered {n} frames but window {start}-{end} has {expected}")
    return {"window": [start, end], "frame_count": n, "frame_files": frame_files,
            "out_w": out_w, "out_h": out_h, "fps": fps, "displays": displays,
            "render_flags": dict(RESEARCH_FLAGS)}


# ----------------------------------------------------------------- encoding
def h264_encode_command(frames_dir, out_path, fps) -> list:
    """Established H.264/yuv420p/faststart chain over numbered PNG frames."""
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{float(fps):g}",
        "-i", os.path.join(str(frames_dir), "frame_%06d.png"),
        "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]


def encode_window_mp4(frames_dir, out_path, fps) -> str:
    argv = h264_encode_command(frames_dir, out_path, fps)
    subprocess.run(argv, check=True)
    return str(out_path)


# -------------------------------------------------------------- reconcile
def segment_runs(rows: Sequence[dict]) -> list:
    """Maximal consecutive bbox runs with their segment ids.

    A missing bbox (unavailable/ended) breaks a run; a segment_id change
    also breaks a run, so a renderer can never bridge either.
    """
    runs: list = []
    start: Optional[int] = None
    last: int = 0
    seg: Optional[int] = None
    for r in rows:
        bbox = r.get("bbox")
        sid = r.get("segment_id")
        if bbox is not None:
            if start is None:
                start, last, seg = r["source_frame_index"], \
                    r["source_frame_index"], sid
            elif sid == seg and r["source_frame_index"] == last + 1:
                last = r["source_frame_index"]
            else:
                runs.append({"start": start, "end": last, "segment_id": seg})
                start = last = r["source_frame_index"]
                seg = sid
        else:
            if start is not None:
                runs.append({"start": start, "end": last, "segment_id": seg})
                start = None
    if start is not None:
        runs.append({"start": start, "end": last, "segment_id": seg})
    return runs


def reconcile_summary(mil_rows: Sequence[dict], *, window: Sequence[int],
                      fps: float, clip: str, baseline_rows=None,
                      qa_verdicts=None, seed_frame=None, provenance=None) -> dict:
    """Derive ALL counts from the per-frame records -- never from prose.

    The declared window must match the row span: a summary that says
    [25, 51] while rows span [25, 70] is exactly the defect this rejects.
    Pre-seed window frames (rows starting after window start, e.g. 6541855's
    MIL rows starting at seed 140 inside window 128-156) are counted as
    pre_seed_frames; rows ending before the window end raise.
    """
    start, end = int(window[0]), int(window[1])
    total = end - start + 1
    idxs = [r["source_frame_index"] for r in mil_rows]
    if not idxs:
        raise ValueError(f"{clip}: no MIL rows to reconcile")
    if any(b != a + 1 for a, b in zip(idxs, idxs[1:])):
        raise ValueError(f"{clip}: MIL rows are not contiguous")
    if idxs[-1] != end:
        raise ValueError(
            f"{clip}: declared window ends at {end} but per-frame rows end at "
            f"{idxs[-1]} ({len(idxs)} rows). Window bounds and per-frame "
            f"records must agree.")
    if idxs[0] < start:
        raise ValueError(f"{clip}: rows start {idxs[0]} before window {start}")
    qa = qa_verdicts or {}
    seed = seed_frame if seed_frame is not None else (
        next((r["source_frame_index"] for r in mil_rows
              if r.get("state") == "seed"), None))

    state_counts = {k: 0 for k in
                    ("seed", "tracked", "reacquired", "unavailable", "ended")}
    display_counts = {k: 0 for k in DISPLAY_STATES}
    for r in mil_rows:
        st = r.get("state") or "unavailable"
        if st not in state_counts:
            raise ValueError(f"{clip}: unknown MIL state {st!r}")
        state_counts[st] += 1
        d = classify_mil_display(st, r.get("bbox") is not None,
                                 qa.get(str(r["source_frame_index"]),
                                        qa.get(r["source_frame_index"])))
        display_counts[d] += 1

    runs = segment_runs(mil_rows)
    longest = max(((r["end"] - r["start"] + 1) for r in runs), default=0)
    s = {
        "clip": clip,
        "window_frames": [start, end],
        "total_frames": total,
        "rows_present": len(idxs),
        "pre_seed_frames": max(0, idxs[0] - start),
        "seed_frame": seed,
        "fps": float(fps),
        "timestamp_basis": {
            "source_time": "source_frame_index / fps",
            "window_relative": "(source_frame_index - window_start) / fps",
            "seed_relative": "(source_frame_index - seed_frame) / fps",
            "note": "tracker default 30fps timestamps corrected; playback "
                    "time of the encoded window equals window-relative time",
        },
        "mil_state_counts": state_counts,
        "display_counts": display_counts,
        "segments": len(runs),
        "segment_runs": runs,
        "longest_bbox_run": longest,
        "mil_qa_counts": _qa_counts(qa),
        "render_flags": dict(RESEARCH_FLAGS),
    }
    if provenance:
        s["provenance"] = dict(provenance)
    if baseline_rows is not None:
        bs = {k: 0 for k in ("observed", "unavailable")}
        bidx = [r["source_frame_index"] for r in baseline_rows]
        if any(b != a + 1 for a, b in zip(bidx, bidx[1:])):
            raise ValueError(f"{clip}: baseline rows are not contiguous")
        if bidx and (bidx[0] < start or bidx[-1] != end):
            raise ValueError(
                f"{clip}: baseline rows span {bidx[0]}-{bidx[-1]} but window "
                f"is {start}-{end}")
        for r in baseline_rows:
            st = r.get("state") or "unavailable"
            if st not in bs:
                raise ValueError(f"{clip}: unknown baseline state {st!r}")
            bs[st] += 1
        s["baseline_state_counts"] = bs
        s["baseline_rows_present"] = len(bidx)
    return s


def _qa_counts(qa: dict) -> dict:
    counts: dict = {}
    for v in qa.values():
        counts[v] = counts.get(v, 0) + 1
    return counts