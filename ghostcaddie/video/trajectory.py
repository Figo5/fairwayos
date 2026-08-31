"""Pixel-space ball trajectory tracing from explicit observations only.

This module is intentionally independent from shot gates, analytics, and course
geometry. It never upgrades an inferred point to an observation. Optional constant-velocity prediction is bounded and remains explicitly marked.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class TrajectoryPoint:
    frame_index: int
    timestamp_seconds: float
    x: float
    y: float
    confidence: float
    uncertainty_radius: float
    provenance: str
    lifetime: int


@dataclass(frozen=True)
class TrajectoryTrace:
    points: tuple[TrajectoryPoint, ...]
    gaps: tuple[tuple[int, int], ...]
    warnings: tuple[str, ...]
    available: bool


def _field(observation: Any, name: str, default=None):
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def _radius(confidence: float) -> float:
    """A display uncertainty radius derived only from explicit confidence."""
    return round(2.0 + (1.0 - confidence) * 18.0, 3)


def _explicit(observation: Any):
    ball = _field(observation, "ball")
    if ball is None:
        return None
    try:
        x, y, confidence = float(ball["x"]), float(ball["y"]), float(ball["confidence"])
        frame = int(_field(observation, "frame_index"))
        timestamp = float(_field(observation, "timestamp_seconds"))
    except (KeyError, TypeError, ValueError):
        raise ValueError("ball observations require frame_index, timestamp_seconds, and ball x/y/confidence")
    if not all(math.isfinite(value) for value in (x, y, confidence, timestamp)):
        raise ValueError("ball observation values must be finite")
    if not 0 <= confidence <= 1:
        raise ValueError("ball confidence must be between 0 and 1")
    return frame, timestamp, x, y, confidence


def trace_ball_observations(
    observations: Iterable[Any], *, max_interpolation_gap: int = 2, fade_frames: int = 3,
    max_step_pixels: Optional[float] = None, predict_missing: bool = False,
    max_prediction_gap: int = 0, optical_flow=None, camera_motion=None,
    max_vertical_speed_pixels_per_second: Optional[float] = None,
    ground_y: Optional[float] = None, ground_tolerance_pixels: float = 0.0
) -> TrajectoryTrace:
    """Build a deterministic pixel trace from explicit ball observations.

    Missing balls remain unavailable. Missing frames are interpolated only when
    the number of missing frames is at most ``max_interpolation_gap``; longer
    losses are represented as gaps and are never bridged.
    """
    if isinstance(max_interpolation_gap, bool) or max_interpolation_gap < 0:
        raise ValueError("max_interpolation_gap must be non-negative")
    if isinstance(fade_frames, bool) or fade_frames < 1:
        raise ValueError("fade_frames must be positive")
    if max_step_pixels is not None and (not math.isfinite(float(max_step_pixels)) or max_step_pixels <= 0):
        raise ValueError("max_step_pixels must be positive when provided")
    if isinstance(max_prediction_gap, bool) or max_prediction_gap < 0:
        raise ValueError("max_prediction_gap must be non-negative")
    if max_vertical_speed_pixels_per_second is not None and (
            not math.isfinite(float(max_vertical_speed_pixels_per_second)) or
            max_vertical_speed_pixels_per_second <= 0):
        raise ValueError("max_vertical_speed_pixels_per_second must be positive when provided")
    if ground_y is not None and not math.isfinite(float(ground_y)):
        raise ValueError("ground_y must be finite when provided")
    if not math.isfinite(float(ground_tolerance_pixels)) or ground_tolerance_pixels < 0:
        raise ValueError("ground_tolerance_pixels must be non-negative")
    if camera_motion is not None and not callable(camera_motion):
        raise TypeError("camera_motion must be callable when provided")
    records = list(observations)
    explicit = [item for observation in records if (item := _explicit(observation)) is not None]
    if not explicit:
        return TrajectoryTrace((), (), ("ball_unavailable",), False)
    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(explicit, explicit[1:])):
        raise ValueError("explicit ball observations must be strictly ordered")
    if camera_motion is not None:
        stabilized = []
        for frame, timestamp, x, y, confidence in explicit:
            try:
                dx, dy = camera_motion(frame, timestamp)
                dx, dy = float(dx), float(dy)
            except (TypeError, ValueError, IndexError):
                raise ValueError("camera_motion must return finite (x, y) displacement")
            if not all(math.isfinite(value) for value in (dx, dy)):
                raise ValueError("camera_motion must return finite (x, y) displacement")
            stabilized.append((frame, timestamp, x - dx, y - dy, confidence))
        explicit = stabilized

    points = []
    gaps = []
    warnings = []
    continuity_broken = False
    for index, current in enumerate(explicit):
        frame, timestamp, x, y, confidence = current
        if index:
            previous = explicit[index - 1]
            missing = frame - previous[0] - 1
            step_distance = math.hypot(x - previous[2], y - previous[3])
            plausible = max_step_pixels is None or step_distance <= float(max_step_pixels) * (frame - previous[0])
            if plausible and max_vertical_speed_pixels_per_second is not None:
                elapsed = timestamp - previous[1]
                if abs(y - previous[3]) / elapsed > float(max_vertical_speed_pixels_per_second):
                    plausible = False
                    warnings.append("implausible_vertical_speed")
            if plausible and ground_y is not None:
                boundary = float(ground_y) + float(ground_tolerance_pixels)
                if y > boundary or previous[3] > boundary:
                    plausible = False
                    warnings.append("ground_constraint")
            if not plausible:
                gaps.append((previous[0] + 1, frame - 1) if missing else (frame, frame))
                if "implausible_vertical_speed" not in warnings and "ground_constraint" not in warnings:
                    warnings.append("implausible_jump")
                continuity_broken = True
            elif missing <= max_interpolation_gap:
                for step in range(1, missing + 1):
                    fraction = step / (missing + 1)
                    points.append(TrajectoryPoint(
                        frame_index=previous[0] + step,
                        timestamp_seconds=previous[1] + fraction * (timestamp - previous[1]),
                        x=previous[2] + fraction * (x - previous[2]),
                        y=previous[3] + fraction * (y - previous[3]),
                        confidence=round(previous[4] + fraction * (confidence - previous[4]), 6),
                        uncertainty_radius=_radius(previous[4] + fraction * (confidence - previous[4])),
                        provenance="interpolated", lifetime=0,
                    ))
            elif missing:
                gaps.append((previous[0] + 1, frame - 1))
                continuity_broken = True
        points.append(TrajectoryPoint(frame, timestamp, x, y, confidence, _radius(confidence), "observed", 0))

    # Trailing missing frames are never predicted by default. Opt-in prediction
    # is bounded, and is marked synthetic so callers cannot mistake it for an
    # explicit candidate observation.
    if predict_missing and max_prediction_gap:
        last = explicit[-1]
        try:
            trailing = [(_field(item, "frame_index"), float(_field(item, "timestamp_seconds")))
                        for item in records if _field(item, "frame_index") is not None
                        and int(_field(item, "frame_index")) > last[0]]
        except (TypeError, ValueError):
            trailing = []
        trailing.sort()
        trailing_all = trailing
        trailing = trailing[:max_prediction_gap]
        if len(trailing_all) > max_prediction_gap:
            gaps.append((trailing_all[0][0], trailing_all[-1][0]))
        elif trailing and (len(explicit) < 2 and optical_flow is None or not predict_missing or not max_prediction_gap):
            gaps.append((trailing[0][0], trailing[-1][0]))
        if (len(trailing_all) <= max_prediction_gap and trailing and
                (len(explicit) >= 2 or optical_flow is not None) and not continuity_broken):
            if len(explicit) >= 2:
                prior = explicit[-2]
                vx = (last[2] - prior[2]) / (last[0] - prior[0])
                vy = (last[3] - prior[3]) / (last[0] - prior[0])
            else:
                vx = vy = 0.0
            for frame_index, timestamp in trailing:
                if optical_flow is not None:
                    proposed = optical_flow(
                        (last[2], last[3]), frame_index, timestamp
                    )
                else:
                    proposed = (last[2] + vx * (frame_index - last[0]),
                                last[3] + vy * (frame_index - last[0]))
                try:
                    px, py = float(proposed[0]), float(proposed[1])
                except (TypeError, ValueError, IndexError):
                    warnings.append("prediction_rejected")
                    break
                distance = math.hypot(px - last[2], py - last[3])
                bound = (float(max_step_pixels) if max_step_pixels is not None else 24.0) * (frame_index - last[0])
                ground_invalid = (ground_y is not None and
                                  py > float(ground_y) + float(ground_tolerance_pixels))
                vertical_invalid = False
                if max_vertical_speed_pixels_per_second is not None:
                    elapsed = timestamp - last[1]
                    vertical_invalid = (elapsed <= 0 or abs(py - last[3]) / elapsed >
                                        float(max_vertical_speed_pixels_per_second))
                if (not all(math.isfinite(value) for value in (px, py)) or distance > bound or
                        ground_invalid or vertical_invalid):
                    if ground_invalid:
                        warnings.append("ground_constraint")
                    elif vertical_invalid:
                        warnings.append("implausible_vertical_speed")
                    warnings.append("prediction_rejected")
                    break
                confidence = max(0.0, last[4] - 0.15 * (frame_index - last[0]))
                points.append(TrajectoryPoint(frame_index, timestamp, px, py, confidence,
                    _radius(confidence) + 4.0 * (frame_index - last[0]), "predicted", 0))
                warnings.append("predicted_track")
                last = (frame_index, timestamp, px, py, confidence)

    # Lifetime is a visual retention/fade value, not a physical estimate.
    total = len(points)
    points = [TrajectoryPoint(
        p.frame_index, p.timestamp_seconds, p.x, p.y, p.confidence,
        p.uncertainty_radius, p.provenance,
        fade_frames if index == 0 else max(1, min(fade_frames - index, total - index))
    ) for index, p in enumerate(points)]
    if gaps:
        warnings.append("ball_track_gap")
    return TrajectoryTrace(tuple(points), tuple(gaps), tuple(dict.fromkeys(warnings)), True)


def build_trajectory_overlay(trace: TrajectoryTrace) -> str:
    """Return a deterministic ffmpeg filter fragment for trace visualization."""
    if not isinstance(trace, TrajectoryTrace):
        raise TypeError("trace must be a TrajectoryTrace")
    if not trace.available:
        return "drawtext=text='trajectory\\: unavailable':x=12:y=12:fontcolor=gray," \
               "drawtext=text='warnings\\: ball_unavailable':x=12:y=36:fontcolor=red"
    filters = []
    for point in trace.points:
        color = "lime" if point.provenance == "observed" else "yellow"
        filters.append(
            f"drawbox=x={point.x - 4:g}:y={point.y - 4:g}:w=8:h=8:color={color}:t=fill"
        )
        filters.append(
            f"drawbox=x={point.x - point.uncertainty_radius:g}:y={point.y - point.uncertainty_radius:g}:"
            f"w={2 * point.uncertainty_radius:g}:h={2 * point.uncertainty_radius:g}:color={color}@0.45:t=1"
        )
        filters.append(
            f"drawtext=text='{point.provenance}\\: confidence={point.confidence:.2f} "
            f"radius={point.uncertainty_radius:g} fade={point.lifetime}':"
            f"x={point.x + 6:g}:y={point.y - 12:g}:fontcolor={color}"
        )
    warning_text = ",".join(trace.warnings) if trace.warnings else "none"
    filters.append(f"drawtext=text='warnings\\: {warning_text}':x=12:y=12:fontcolor=red")
    return ",".join(filters)
