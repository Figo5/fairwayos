"""Image-space geometry for the demo path. Pure Python, no cv2, no numpy.

Two rules are enforced here rather than remembered:

* **One space, one unit.** Every coordinate that reaches this module is a source
  pixel coordinate. Angles are computed from pixel deltas with the source sample
  aspect ratio applied to x, never from normalised coordinates mixed with pixel
  ones — mixing the two silently changes the units of a degree.
* **A missing point is missing.** Helpers return ``None`` (and the caller records
  why) instead of substituting 0, the previous point, or the frame centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Point2 = Tuple[float, float]


def is_finite_pair(p: Optional[Sequence[float]]) -> bool:
    return (isinstance(p, (list, tuple)) and len(p) >= 2
            and math.isfinite(float(p[0])) and math.isfinite(float(p[1])))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def midpoint(a: Sequence[float], b: Sequence[float]) -> Point2:
    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)


def angle_degrees(a: Sequence[float], b: Sequence[float], c: Sequence[float],
                  *, pixel_aspect: float = 1.0) -> Optional[float]:
    """Interior angle at ``b`` in the triangle a-b-c, in degrees (0..180).

    x is scaled by the sample aspect ratio first, so a non-square-pixel source does
    not report a warped angle. Returns None for a degenerate (zero-length) limb,
    because such an angle is undefined, not zero.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    v1 = ((ax - bx) * pixel_aspect, ay - by)
    v2 = ((cx - bx) * pixel_aspect, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def tilt_degrees(left: Sequence[float], right: Sequence[float],
                 *, pixel_aspect: float = 1.0) -> Optional[float]:
    """Signed tilt of the LINE through two points from the image horizontal, in
    degrees, folded into ``(-90, 90]``.

    The two landmarks describe a line, not an arrow: the same pair given in the
    other order gives the same tilt. Without this fold a shoulder line leaning the
    other way reads as -137 or +170 degrees, which is a direction, not a tilt — and
    a percentile spread over such values is an artifact of the wrap, not a
    measurement.

    The sign is the slope of the endpoint at the larger image x (the image-right
    end): positive means that end sits LOWER in the image (larger y). Image y grows
    downwards, so the convention is stated here and repeated in the report.
    Undefined (None) for a zero-length segment.
    """
    dx = (float(right[0]) - float(left[0])) * pixel_aspect
    dy = float(right[1]) - float(left[1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    degree = math.degrees(math.atan2(dy, dx))
    while degree > 90.0:
        degree -= 180.0
    while degree <= -90.0:
        degree += 180.0
    return degree


def flexion_degrees(interior_angle: Optional[float]) -> Optional[float]:
    """Knee flexion = 180 - interior angle. A straight leg reads 0 degrees."""
    if interior_angle is None:
        return None
    return 180.0 - interior_angle


def path_length(points: Iterable[Sequence[float]]) -> float:
    """Sum of consecutive step lengths. Callers pass only real observations."""
    pts: List[Sequence[float]] = [p for p in points if is_finite_pair(p)]
    return sum(distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def max_step(points: Sequence[Sequence[float]]) -> Optional[float]:
    pts = [p for p in points if is_finite_pair(p)]
    if len(pts) < 2:
        return None
    return max(distance(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def linear_interpolate(a: Sequence[float], b: Sequence[float], t: float) -> Point2:
    """Point at fraction ``t`` along a->b. Never used to replace an observation."""
    return (float(a[0]) + (float(b[0]) - float(a[0])) * t,
            float(a[1]) + (float(b[1]) - float(a[1])) * t)


@dataclass
class MetricValue:
    """One reported quantity, with how much of the clip it covers."""
    name: str
    unit: str
    state: str                 # supported | unavailable
    value: Optional[float] = None
    direction: Optional[str] = None
    frames_used: int = 0
    frames_possible: int = 0
    reason: str = ""
    space: str = "2d_image"
    detail: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "unit": self.unit, "state": self.state,
             "space": self.space, "value": None if self.value is None else round(self.value, 3),
             "frames_used": self.frames_used, "frames_possible": self.frames_possible,
             "reason": self.reason}
        if self.direction:
            d["direction"] = self.direction
        if self.detail:
            d["detail"] = self.detail
        return d
