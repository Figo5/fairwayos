"""Separate, genuine image-motion review over existing FairwayOS target points.

This module measures ONLY observed image motion on the source-video timeline. It
never reports physical speed: there is no pixel-to-metre factor, no assumed golfer
height or club length and no generic tour statistics, and none of camera
stabilisation, interpolation, smoothing, target correction or inferred action time
is applied. Automatic provenance describes the saved position estimates; the
target seed or pose ROI may still have been supplied by an operator.

It is pure standard-library Python. The automatic statuses/origins it accepts come
from ``ghostcaddie.demo.tracking`` (imported, never redefined or edited there); a
pair of points is measured only when both positions are recorded as automatic
observations on exactly consecutive source frames. Pose estimates have a separate
local status; existing tracking statuses are unchanged.
"""
from __future__ import annotations

import collections.abc
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tracking import (AUTOMATIC_STATUSES, ORIGIN_AUTOMATIC, ORIGIN_DERIVED,
                       ORIGIN_OPERATOR, STATUS_INTERPOLATED, STATUS_LOST,
                       STATUS_MANUAL, STATUS_MISSING)

PROFILE_NAME = "2d_image_motion_review"
SCOPE = "2d_image_motion_source_timeline"
UNIT = "native_px/video_s"
NORMALISED_UNIT = "source_height/video_s"

REASON = ("No spatial/camera calibration or independently verified action time; only "
          "observed image motion on source-video timeline.")

LIMITATIONS = (
    "Camera pan, zoom and perspective change alter the measured image motion: a value "
    "describes this clip's camera, not the player's real motion.",
    "Pose/target identity errors (the point is the wrong joint, club or target) change "
    "the values; this review does not re-verify identity.",
    "The true impact instant and the true swing peak may be missing from the window, so "
    "the largest measured rate is not the real maximum.",
    "The source-encoded time may be slow-motion, so video seconds are not real-world "
    "seconds.",
    "Cross-clip pixels/second are not comparable because the source geometry differs; a "
    "rate normalised by source height is still not physical speed.",
)

MEAN_NOTE = ("The mean rate is duration-weighted (total paired path over total paired "
             "dt_seconds), not the arithmetic mean of the per-pair rates. Equal "
             "coordinates contribute zero steps: they may reflect a stationary "
             "target, coordinate rounding or tracker freeze, and do not prove "
             "that the source pictures are duplicates.")
PEAK_NOTE = ("`peak_observed_rate` is the largest rate among the pairs actually "
             "measured in this window; it is NOT the real maximum swing velocity. With "
             "incomplete coverage the fastest part of the motion may simply not have "
             "been observed.")
PAIR_RULE = ("A pair is measured only between two AUTOMATIC observations on exactly "
             "consecutive source frames f-1 and f, both inside the window, with no cut "
             "at f and no recorded method ending in '_reacquired' at f. Gaps, operator "
             "anchors, derived/interpolated points and missing frames are never "
             "bridged. Automatic is saved provenance, not independent verification; "
             "target seeds or the pose ROI may be operator inputs.")
NO_PAIR_REASON = ("no pair of consecutive automatic observations inside the window "
                  "could be measured, so no image-motion rate is reported")
RESULT_NOTE = ("Rates are image motion on the source-video timeline (native pixels per "
               "video second), never physical speed. Incomplete coverage can miss the "
               "fast part of the motion.")

#: The declared, machine-readable profile of this review. ``physical_speed_mph`` is
#: pinned to ``None`` and there is no flag anywhere that would let a reader turn
#: these image-motion rates into mph.
PROFILE = {
    "profile": PROFILE_NAME,
    "scope": SCOPE,
    "unit": UNIT,
    "normalised_unit": NORMALISED_UNIT,
    "coordinate_space": "native_source_pixels",
    "time_basis": "source_video_timeline_pts_seconds",
    "physical_speed_mph": None,
    "reason": REASON,
    "limitations": list(LIMITATIONS),
    "pair_rule": PAIR_RULE,
    "excluded_processing": ["camera_stabilisation", "interpolation", "smoothing",
                            "target_correction", "inferred_action_time"],
}

REACQUIRED_SUFFIX = "_reacquired"
STATUS_OBSERVED_POSE = "observed_pose"
OBSERVATION_STATUSES = AUTOMATIC_STATUSES | {STATUS_OBSERVED_POSE}

#: Origins that may appear on an input point.
ALLOWED_ORIGINS = frozenset({ORIGIN_AUTOMATIC, ORIGIN_OPERATOR, ORIGIN_DERIVED})
#: Statuses that may appear on an input point (the automatic ones imported from the
#: tracking policy, plus the non-observations it also defines).
ALLOWED_STATUSES = frozenset(OBSERVATION_STATUSES
                             | {STATUS_MISSING, STATUS_LOST, STATUS_MANUAL,
                                STATUS_INTERPOLATED})

__all__ = ["PROFILE", "STATUS_OBSERVED_POSE", "summarize_motion"]


# --------------------------------------------------------------------------
# validation (pure; never repairs, never mutates the input)

def _is_number(value: Any) -> bool:
    """A finite-capable Python number, never a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_source_size(source_size: Any) -> Tuple[int, int]:
    if not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
        raise ValueError("source_size must be a (width, height) pair")
    width, height = source_size
    if type(width) is not int or type(height) is not int:
        raise ValueError("source_size must be a positive (width, height) integer pair")
    if width <= 0 or height <= 0:
        raise ValueError("source_size must be a positive (width, height) integer pair")
    return int(width), int(height)


def _validate_frames(start_frame: Any, end_frame: Any) -> Tuple[int, int]:
    if type(start_frame) is not int or type(end_frame) is not int:
        raise ValueError("start_frame and end_frame must be exact integer frames")
    if start_frame < 0 or end_frame < 0:
        raise ValueError("start_frame and end_frame must be non-negative frames")
    if start_frame > end_frame:
        raise ValueError("start_frame must not exceed end_frame")
    return int(start_frame), int(end_frame)


def _validate_pts_seconds(pts_seconds: Any, end_frame: int) -> List[float]:
    """The complete, strictly increasing source timeline, in seconds."""
    if not isinstance(pts_seconds, (list, tuple)):
        raise ValueError("pts_seconds must be the complete source timeline list")
    if not pts_seconds:
        raise ValueError("pts_seconds must not be empty")
    seconds: List[float] = []
    previous: Optional[float] = None
    for index, value in enumerate(pts_seconds):
        if not _is_number(value):
            raise ValueError(f"pts_seconds[{index}] must be a finite int or float")
        stamped = float(value)
        if not math.isfinite(stamped):
            raise ValueError(f"pts_seconds[{index}] must be finite")
        if previous is not None and stamped <= previous:
            raise ValueError("pts_seconds must be strictly increasing")
        previous = stamped
        seconds.append(stamped)
    if end_frame > len(seconds) - 1:
        raise ValueError("pts_seconds does not reach end_frame")
    return seconds


def _validate_points(points: Any, *, length: int, width: int, height: int) -> List[dict]:
    """Validate every declared point, inside or outside the window, before use.

    A malformed coordinate (a partial x/y, a non-finite value, a missing coordinate)
    is refused whatever the point's status is -- including ``missing``.
    """
    if not isinstance(points, (list, tuple)):
        raise ValueError("points must be a list of point mappings")
    seen = set()
    records: List[dict] = []
    for index, point in enumerate(points):
        if not isinstance(point, collections.abc.Mapping):
            raise ValueError(f"point {index} must be a mapping")
        if "source_frame" not in point:
            raise ValueError(f"point {index} is missing source_frame")
        frame = point["source_frame"]
        if type(frame) is not int:
            raise ValueError(f"point {index} source_frame must be an exact integer")
        if frame < 0 or frame > length - 1:
            raise ValueError(f"point {index} source_frame is outside the declared "
                             f"source frames")
        if frame in seen:
            raise ValueError(f"duplicate point for source_frame {frame}")
        seen.add(frame)
        if "origin" not in point:
            raise ValueError(f"point {index} is missing origin")
        origin = point["origin"]
        if not isinstance(origin, str) or origin not in ALLOWED_ORIGINS:
            raise ValueError(f"point {index} has unknown origin {origin!r}")
        if "status" not in point:
            raise ValueError(f"point {index} is missing status")
        status = point["status"]
        if not isinstance(status, str) or status not in ALLOWED_STATUSES:
            raise ValueError(f"point {index} has unknown status {status!r}")
        if "x" not in point or "y" not in point:
            raise ValueError(f"point {index} must declare x and y (native pixels or "
                             f"both None)")
        x, y = point["x"], point["y"]
        if (x is None) != (y is None):
            raise ValueError(f"point {index} has a partial coordinate: x/y must both "
                             f"be given or both be None")
        if x is None:
            xf = yf = None
        else:
            if not _is_number(x) or not _is_number(y):
                raise ValueError(f"point {index} coordinate must be a finite number")
            xf, yf = float(x), float(y)
            if not math.isfinite(xf) or not math.isfinite(yf):
                raise ValueError(f"point {index} coordinate must be finite")
            if not (0.0 <= xf < float(width)) or not (0.0 <= yf < float(height)):
                raise ValueError(f"point {index} lies outside the native source bounds")
        method = point.get("method", "")
        if method is None:
            method = ""
        if not isinstance(method, str):
            raise ValueError(f"point {index} method must be a string")
        score = point.get("score", None)
        if score is not None and (not _is_number(score) or not math.isfinite(score)):
            raise ValueError(f"point {index} score must be a finite number or None")
        records.append({"source_frame": int(frame), "origin": origin, "status": status,
                        "x": xf, "y": yf, "method": method})
    return records


def _validate_cuts(cut_frames: Any, *, length: int) -> List[int]:
    if not isinstance(cut_frames, (list, tuple)):
        raise ValueError("cut_frames must be a list of source frames")
    seen = set()
    cuts: List[int] = []
    for value in cut_frames:
        if type(value) is not int:
            raise ValueError("cut_frames must be exact integer source frames")
        if value < 0 or value > length - 1:
            raise ValueError("cut_frames must lie inside the declared source frames")
        if value in seen:
            raise ValueError("cut_frames must be unique")
        seen.add(value)
        cuts.append(int(value))
    return sorted(cuts)


# --------------------------------------------------------------------------
# measurement (pure)

def _is_automatic_observation(record: Mapping[str, Any]) -> bool:
    """A position recorded as automatic with native source coordinates."""
    return (record["origin"] == ORIGIN_AUTOMATIC
            and record["status"] in OBSERVATION_STATUSES
            and record["x"] is not None)


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the same inclusive convention as numpy)."""
    size = len(ordered)
    if size == 1:
        return ordered[0]
    position = q * (size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _pair(previous: Mapping[str, Any], current: Mapping[str, Any],
          seconds: Sequence[float], height: int) -> dict:
    dt_seconds = seconds[current["source_frame"]] - seconds[previous["source_frame"]]
    dx = current["x"] - previous["x"]
    dy = current["y"] - previous["y"]
    step = math.hypot(dx, dy)
    rate = step / dt_seconds
    if not math.isfinite(dt_seconds) or not math.isfinite(rate):
        raise ValueError("source timeline produces a non-finite motion rate")
    return {
        "source_frames": [previous["source_frame"], current["source_frame"]],
        "dt_seconds": dt_seconds,
        "dxdy_native_px": [dx, dy],
        "step_native_px": step,
        "rate_native_px_per_video_second": rate,
        "rate_source_height_per_video_second": rate / height,
        "methods": [previous["method"], current["method"]],
    }


def summarize_motion(points, pts_seconds, *, source_size, start_frame, end_frame,
                     cut_frames=()):
    """Review observed image motion on the source-video timeline.

    Nothing here is physical speed: the result's ``physical_speed_mph`` is always
    ``None`` and no flag enables a mph value. Only pairs of consecutive AUTOMATIC
    observations are measured; a window with no such pair is ``available=False`` with
    a ``summary`` of ``None``.
    """
    width, height = _validate_source_size(source_size)
    start_frame, end_frame = _validate_frames(start_frame, end_frame)
    seconds = _validate_pts_seconds(pts_seconds, end_frame)
    length = len(seconds)
    if start_frame > length - 1:
        raise ValueError("start_frame is outside the declared source frames")
    records = _validate_points(points, length=length, width=width, height=height)
    cuts = _validate_cuts(cut_frames, length=length)
    cut_set = set(cuts)

    ordered = sorted(records, key=lambda record: record["source_frame"])
    eligible = [record for record in ordered
                if _is_automatic_observation(record)
                and start_frame <= record["source_frame"] <= end_frame]

    pairs: List[dict] = []
    previous: Optional[Mapping[str, Any]] = None
    for record in eligible:
        if previous is not None:
            frame = record["source_frame"]
            previous_frame = previous["source_frame"]
            same_segment = (frame == previous_frame + 1
                            and frame not in cut_set
                            and not record["method"].endswith(REACQUIRED_SUFFIX))
            if same_segment:
                pairs.append(_pair(previous, record, seconds, height))
        previous = record

    pair_count = len(pairs)
    denominator = max(0, end_frame - start_frame)
    paired_coverage = (pair_count / denominator) if denominator else 0.0

    if pairs:
        total_path = sum(pair["step_native_px"] for pair in pairs)
        total_dt = sum(pair["dt_seconds"] for pair in pairs)
        rates = [pair["rate_native_px_per_video_second"] for pair in pairs]
        peak = max(rates)
        # pairs are in frame order, so the first index of the peak is the earliest.
        peak_index = rates.index(peak)
        summary = {
            "mean_rate_native_px_per_video_second": total_path / total_dt,
            "p50_rate_native_px_per_video_second": _linear_quantile(sorted(rates), 0.5),
            "p95_rate_native_px_per_video_second": _linear_quantile(sorted(rates), 0.95),
            "peak_observed_rate_native_px_per_video_second": peak,
            "peak_source_pair": list(pairs[peak_index]["source_frames"]),
            "total_paired_path_native_px": total_path,
            "total_paired_dt_seconds": total_dt,
            "mean_note": MEAN_NOTE,
            "peak_note": PEAK_NOTE,
        }
    else:
        summary = None

    return {
        "profile": PROFILE_NAME,
        "scope": SCOPE,
        "unit": UNIT,
        "normalised_unit": NORMALISED_UNIT,
        "physical_speed_mph": None,
        "reason": REASON,
        "limitations": list(LIMITATIONS),
        "available": pair_count > 0,
        "unavailable_reason": None if pair_count else NO_PAIR_REASON,
        "source_size": {"width": width, "height": height},
        "start_frame": start_frame,
        "end_frame": end_frame,
        "cut_frames": list(cuts),
        "frames_in_window": max(0, end_frame - start_frame + 1),
        "eligible_automatic_frames": len(eligible),
        "pair_count": pair_count,
        "zero_step_pair_count": sum(pair["step_native_px"] == 0 for pair in pairs),
        "nonzero_step_pair_count": sum(pair["step_native_px"] > 0 for pair in pairs),
        "paired_coverage": paired_coverage,
        "pairs": pairs,
        "summary": summary,
        "note": RESULT_NOTE,
    }
