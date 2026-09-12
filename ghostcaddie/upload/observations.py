"""The contract between automatic perception and anything that measures.

Perception emits POINT OBSERVATIONS and nothing else: a target, a frame, a pixel
coordinate, a real timestamp, how it was obtained. Measurement is a separate
consumer that reads frozen observations. Nothing here initialises, crops or fits
from a reference coordinate, and nothing here invents a coordinate.

Two rules come straight from reading the reference implementations:

  * GhostBall's pipeline_integration.py substitutes `[52.0, 34.0]` -- the centre
    of the pitch -- when a ball position is missing, so downstream numbers on
    those frames are computed from a position nobody measured. This module fails
    CLOSED instead: a frame with no detection produces no observation.
  * Soccer's main.py gets its ball from a dedicated trained detector run on
    640x640 tiles, not from a generic tracker. Until an equivalent golf detector
    exists, the honest output for ball and clubhead is "no observations".

Metric speed additionally needs a spatial calibration AND the real capture rate.
A file's playback rate is not its capture rate: the Rory clip declares
30000/1001 playback and carries no capture-rate metadata, so mph is refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


class MetricSpeedUnavailable(RuntimeError):
    """Raised when metric speed is asked for without the inputs it requires."""


def _finite(v, name):
    if v is None or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return float(v)


@dataclass(frozen=True)
class PointObservation:
    target: str
    frame_index: int
    x_px: float
    y_px: float
    t_seconds: float
    method: str
    source_sha256: str
    confidence: Optional[float] = None
    uncertainty_px: Optional[float] = None
    state: str = "observed"

    def __post_init__(self):
        for n in ("x_px", "y_px", "t_seconds"):
            _finite(getattr(self, n), n)
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative int")
        if not self.method or not self.source_sha256:
            raise ValueError("an observation must name its method and its source")

    def to_dict(self) -> dict:
        return {"target": self.target, "frame_index": self.frame_index,
                "x_px": round(self.x_px, 3), "y_px": round(self.y_px, 3),
                "t_seconds": round(self.t_seconds, 6), "method": self.method,
                "source_sha256": self.source_sha256,
                "confidence": self.confidence,
                "uncertainty_px": self.uncertainty_px, "state": self.state,
                "pseudo_label": True, "ground_truth": False,
                "production_eligible": False}


@dataclass(frozen=True)
class Calibration:
    meters_per_pixel: float
    uncertainty: float
    method: str

    def __post_init__(self):
        if not (math.isfinite(self.meters_per_pixel) and self.meters_per_pixel > 0):
            raise ValueError("meters_per_pixel must be finite and positive")
        if not (math.isfinite(self.uncertainty) and self.uncertainty >= 0):
            raise ValueError("uncertainty must be finite and non-negative")
        if not self.method:
            raise ValueError("a calibration must say how it was obtained")


@dataclass(frozen=True)
class Timebase:
    """real_action_elapsed / playback_elapsed. 1.0 means real time, and even
    that must be stated explicitly rather than assumed."""
    slowmo_factor: Optional[float]
    method: str


@dataclass(frozen=True)
class ImageSpeed:
    target: str
    distance_px: float
    dt_seconds: float
    px_per_second: float
    frame_gap: int
    spans_gap: bool


@dataclass(frozen=True)
class MetricSpeed:
    target: str
    meters: float
    real_dt_seconds: float
    meters_per_second: float
    uncertainty_fraction: float
    calibration_method: str
    timebase_method: str


def observations_from_records(records: Sequence[dict], target: str, fps: float,
                              method: str, source_sha256: str,
                              uncertainty_px: Optional[float] = None
                              ) -> List[PointObservation]:
    """Frozen observations from adapter records. Only real detections survive."""
    if not fps or not math.isfinite(fps) or fps <= 0:
        raise ValueError("a positive frame rate is required to timestamp frames")
    out = []
    for r in records or []:
        xy = r.get("point_xy")
        if not r.get("visible") or not xy:
            continue                      # fail closed: no default coordinate
        f = int(r["source_frame"])
        out.append(PointObservation(
            target=target, frame_index=f, x_px=float(xy[0]), y_px=float(xy[1]),
            t_seconds=f / fps, method=method,
            source_sha256=r.get("source_sha256") or source_sha256,
            confidence=r.get("confidence"), uncertainty_px=uncertainty_px,
            state="observed"))
    return out


def image_speed_px_s(a: PointObservation, b: PointObservation) -> ImageSpeed:
    """Pixel speed between two REAL observations of the SAME target."""
    if a.target != b.target:
        raise ValueError(f"cannot measure across targets: {a.target} vs {b.target}")
    dt = b.t_seconds - a.t_seconds
    if dt <= 0:
        raise ValueError("observations must be strictly increasing in time")
    d = math.hypot(b.x_px - a.x_px, b.y_px - a.y_px)
    gap = b.frame_index - a.frame_index
    return ImageSpeed(target=a.target, distance_px=d, dt_seconds=dt,
                      px_per_second=d / dt, frame_gap=gap, spans_gap=gap > 1)


def metric_speed_mps(a: PointObservation, b: PointObservation,
                     calibration: Optional[Calibration],
                     timebase: Timebase) -> MetricSpeed:
    """Metric speed, or a refusal naming exactly what is missing.

    This never estimates, never assumes real time, and never converts pixels to
    metres without a stated calibration.
    """
    if calibration is None:
        raise MetricSpeedUnavailable(
            "no spatial calibration: metres per pixel is unknown for this target "
            "at these frames, so no metric speed can be reported")
    f = timebase.slowmo_factor
    if f is None:
        raise MetricSpeedUnavailable(
            "the real capture rate is unknown: this file declares only its "
            f"playback rate ({timebase.method}). Playback rate is not capture "
            "rate, so no metric speed can be reported")
    f = float(f)
    if not math.isfinite(f) or f <= 0:
        raise MetricSpeedUnavailable(
            f"the slow-motion factor must be finite and positive, got {f!r}")
    img = image_speed_px_s(a, b)
    real_dt = img.dt_seconds / f
    metres = img.distance_px * calibration.meters_per_pixel
    frac = (calibration.uncertainty / calibration.meters_per_pixel
            if calibration.meters_per_pixel else float("inf"))
    return MetricSpeed(target=a.target, meters=metres, real_dt_seconds=real_dt,
                       meters_per_second=metres / real_dt,
                       uncertainty_fraction=frac,
                       calibration_method=calibration.method,
                       timebase_method=timebase.method)
