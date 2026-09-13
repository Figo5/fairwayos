"""Honest evidence-bounded analytics reporting for Fairway."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

_REQUESTED_METRICS = (
    ("Ball speed", ("calibration", "action_time", "valid_target")),
    ("Clubhead speed", ("calibration", "action_time", "valid_target")),
    ("Carry distance", ("calibration", "valid_target", "landing")),
)
_REASON_TEXT = {
    "calibration": "missing physical calibration",
    "action_time": "missing reliable action time",
    "valid_target": "missing valid tracked target",
    "landing": "missing observed landing",
}
_COCO17_ANGLES = (
    ("Left elbow", 5, 7, 9),
    ("Right elbow", 6, 8, 10),
    ("Left knee", 11, 13, 15),
    ("Right knee", 12, 14, 16),
)
_MIN_CONFIDENCE = 0.5


def build_report(observations: Sequence[Mapping[str, Any]] | None, metadata: Mapping[str, Any]) -> str:
    """Build a human-readable report without speculative physical metrics.

    Signature compatibility is preserved. Optional observations may include
    MoveNet/COCO17 ``keypoints`` as 17 ``[y, x, confidence]`` triples with an
    explicit ``keypoint_space``. Pixel coordinates use ``"pixel"``; normalized
    coordinates use ``"normalized"`` plus ``image_width`` and ``image_height``.
    Angles are image-space 2D summaries only; invalid, nonfinite,
    out-of-contract, and low-confidence points are omitted.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a dictionary-like mapping")

    lines = ["Fairway analytics report", ""]
    lines.extend(_physical_metric_lines(metadata))
    lines.extend(_phase_event_lines(metadata))
    lines.extend([
        "",
        "Swing mechanics: qualitative observations only; 2D video observations are not definitive biomechanics.",
    ])

    angle_lines = _angle_summary_lines(observations)
    lines.extend(angle_lines)

    notes = _format_observations(observations)
    if notes:
        lines.extend(notes)
    elif not angle_lines:
        lines.append("- No supported qualitative observations provided.")

    return "\n".join(lines)


def _physical_metric_lines(metadata: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for metric, requirements in _REQUESTED_METRICS:
        missing = [_REASON_TEXT[key] for key in requirements if not metadata.get(key)]
        if missing:
            lines.append(f"{metric}: unavailable ({'; '.join(missing)}).")
        else:
            lines.append(
                f"{metric}: unavailable (physical metric output requires calibrated observations and reviewed action-time semantics)."
            )
    return lines


def _phase_event_lines(metadata: Mapping[str, Any]) -> list[str]:
    events = []
    for event in metadata.get("phase_events") or ():
        if not isinstance(event, Mapping) or event.get("verified") is not True:
            continue
        label, frame, time_s = event.get("label"), event.get("frame"), event.get("time_s")
        if not label or not _valid_frame(frame) or not _finite_number(time_s):
            continue
        checked_time = float(time_s) if isinstance(time_s, (int, float)) else 0.0
        events.append({"label": str(label), "frame": frame, "time_s": checked_time})
    if not events:
        return []

    lines = ["", "Verified phase/event timing (supplied metadata only; no phase detection):"]
    for event in events:
        lines.append(f"- {event['label']}: frame {event['frame']}, t={event['time_s']:.3f}s")
    for prev, curr in zip(events, events[1:]):
        frame_delta = curr["frame"] - prev["frame"]
        time_delta = curr["time_s"] - prev["time_s"]
        if frame_delta >= 0 and time_delta >= 0:
            lines.append(
                f"- {prev['label']} → {curr['label']}: {frame_delta} frames, {time_delta:.3f}s"
            )
    return lines


def _angle_summary_lines(observations: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if not observations:
        return []
    rows: list[str] = []
    saw_keypoints = False
    for observation in observations:
        if not isinstance(observation, Mapping) or "keypoints" not in observation:
            continue
        saw_keypoints = True
        frame = observation.get("frame")
        frame_text = f"frame {frame}" if _valid_frame(frame) else "unframed observation"
        for label, a_idx, b_idx, c_idx in _COCO17_ANGLES:
            angle = _angle_degrees(observation, a_idx, b_idx, c_idx)
            if angle is not None:
                rows.append(f"- {label}: {frame_text} {angle:.1f}°")
    if rows:
        return ["", "2D body angle summaries (image-space; not 3D biomechanics):", *rows]
    if saw_keypoints:
        return ["", "No supported 2D body angles from confident visible keypoints."]
    return []


def _angle_degrees(observation: Mapping[str, Any], a_idx: int, b_idx: int, c_idx: int) -> float | None:
    keypoints = observation.get("keypoints")
    if not isinstance(keypoints, Sequence) or len(keypoints) < 17:
        return None
    transform = _coordinate_transform(observation)
    if transform is None:
        return None
    a, b, c = (transform(_point(keypoints[index])) for index in (a_idx, b_idx, c_idx))

    if a is None or b is None or c is None:
        return None
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def _point(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 3:
        return None
    y, x, confidence = raw[0], raw[1], raw[2]
    if not (_finite_number(y) and _finite_number(x) and _finite_number(confidence)):
        return None
    if float(confidence) < _MIN_CONFIDENCE or float(confidence) > 1.0:
        return None
    return (float(x), float(y))


def _frame_bounds(observation: Mapping[str, Any]) -> tuple[float, float] | None:
    """Frame dimensions, when the observation states usable ones."""
    width = observation.get("image_width")
    height = observation.get("image_height")
    if not (_finite_number(width) and _finite_number(height)):
        return None
    width_f, height_f = float(cast(int | float, width)), float(cast(int | float, height))
    return (width_f, height_f) if width_f > 0 and height_f > 0 else None


def _inside_pixels(point: tuple[float, float] | None,
                   bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    """Drop a pixel point that cannot exist in the stated frame.

    Out-of-frame pixels are impossible geometry, and an angle computed from them
    looks entirely plausible in the report. The far edge is exclusive: in a
    1280-wide frame the last addressable pixel is 1279. Without stated dimensions
    there is nothing to check against, so the point passes unvalidated.
    """
    if point is None or bounds is None:
        return point
    x, y = point
    return point if 0 <= x < bounds[0] and 0 <= y < bounds[1] else None


def _scale_normalized(point: tuple[float, float] | None,
                      bounds: tuple[float, float]) -> tuple[float, float] | None:
    """Scale a normalized point into pixels, then hold it to the same pixel bounds.

    Normalized 1.0 scales to exactly width, which is one past the last addressable
    pixel, so it is out of frame like any other such coordinate. Bounds are
    half-open everywhere: [0, width) and [0, height).
    """
    if point is None:
        return None
    return _inside_pixels((point[0] * bounds[0], point[1] * bounds[1]), bounds)


def _coordinate_transform(observation: Mapping[str, Any]):
    space = observation.get("keypoint_space")
    bounds = _frame_bounds(observation)
    if space in {"pixel", "native_pixel"}:
        return lambda point: _inside_pixels(point, bounds)
    if space != "normalized" or bounds is None:
        return None
    return lambda point: _scale_normalized(point, bounds)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_frame(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _format_observations(observations: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if not observations:
        return []
    formatted: list[str] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        note = observation.get("note")
        if not note:
            continue
        parts: list[str] = []
        if _valid_frame(observation.get("frame")):
            parts.append(f"frame {observation['frame']}")
        if observation.get("label"):
            parts.append(str(observation["label"]))
        if observation.get("visible") is not None:
            parts.append("visible" if observation["visible"] else "not visible")
        if _finite_number(observation.get("confidence")):
            parts.append(f"confidence {observation['confidence']}")
        prefix = " (" + ", ".join(parts) + ")" if parts else ""
        formatted.append(f"-{prefix}: {note}")
    return formatted
