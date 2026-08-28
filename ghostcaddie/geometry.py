"""Geometry engine: coordinate system, mapping, and pure geometric helpers.

All engine coordinates are top-down "yards, tee-to-pin" with y-up. The
CoordinateSystem is the single declared reference frame that ingestion
adapters reuse, so every raw data source is interpreted in one place.

Two coordinate modes are supported:

- mode="manual": raw points are already engine coordinates (yards), possibly
  offset by an `origin`. `to_engine` subtracts the origin; `from_engine` adds
  it back. This is the unchanged Milestone-1 behavior.
- mode="four_point": raw points are source-image coordinates (typically
  pixels, x right / y down) and are mapped to engine coordinates by a planar
  homography solved from four ordered source<->engine correspondences. The
  reverse mapping is an explicit, separate operation for re-projection only;
  analytics always consume engine coordinates.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class CoordinateSystem:
    """Declared top-down coordinate frame.

    mode="manual": raw points are engine yards relative to `origin`.
    mode="four_point": raw points are source-image coordinates mapped to
    engine coordinates by a planar homography fit to exactly four ordered
    correspondences. `source_points[i]` corresponds to `engine_points[i]`;
    both quadrilaterals must be listed in the same perimeter order (e.g.
    top-left, top-right, bottom-right, bottom-left) and must not be
    reordered. `source_units` names the source-image unit (default "pixels");
    `units` names the engine unit (default "yards").
    """

    mode: str = "manual"
    origin: Point2D = Point2D(0.0, 0.0)
    units: str = "yards"
    source_units: str = "pixels"
    source_points: Tuple[Point2D, ...] = ()
    engine_points: Tuple[Point2D, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("manual", "four_point"):
            raise ValueError(
                f"CoordinateSystem mode must be 'manual' or 'four_point', got {self.mode!r}"
            )
        if self.mode == "four_point":
            if len(self.source_points) != 4 or len(self.engine_points) != 4:
                raise ValueError(
                    "four_point mode requires exactly 4 source_points and 4 "
                    f"engine_points (got {len(self.source_points)} and "
                    f"{len(self.engine_points)})"
                )


class Homography:
    """Planar projective mapping H with H[2][2] == 1.

    Maps source (u, v) to engine (X, Y) via
        X = (a*u + b*v + c) / (g*u + h*v + 1)
        Y = (d*u + e*v + f) / (g*u + h*v + 1)
    """

    def __init__(self, coeffs: Sequence[float]):
        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h = coeffs

    def apply(self, u: float, v: float) -> Tuple[float, float]:
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError("homography input coordinates must be finite")
        denom = self.g * u + self.h * v + 1.0
        if not math.isfinite(denom) or abs(denom) < 1e-12:
            raise ValueError("homography denominator near zero at point")
        x = (self.a * u + self.b * v + self.c) / denom
        y = (self.d * u + self.e * v + self.f) / denom
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("homography produced non-finite coordinates")
        return (x, y)


def _check_general_position(points: Sequence[Point2D], scale: float) -> None:
    """Reject a 4-point set where any three points are near-collinear.

    Uses the triangle area (half the cross-product magnitude) of every triple,
    normalized by scale^2, so the check is scale-aware for image-pixel and
    yard-sized coordinates alike. Duplicate points are a special case (zero
    area) and are rejected here too.
    """
    if len(points) != 4:
        raise ValueError("expected exactly 4 points")
    min_area = float("inf")
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs(
                    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
                )
                min_area = min(min_area, area)
    if min_area < 1e-6 * scale * scale:
        raise ValueError(
            "degenerate point set: points are duplicate, near-collinear, or "
            "otherwise not in general position"
        )


def _solve_homography(src: Sequence[Point2D], dst: Sequence[Point2D]) -> Homography:
    """Solve the 8-unknown homography mapping src -> dst by Gaussian elimination.

    Builds the standard 8x8 linear system for [a, b, c, d, e, f, g, h] with
    H[2][2] == 1, using partial-pivot elimination on plain lists. Both point
    sets must be in general position (no three near-collinear); pivots whose
    absolute value is below a scale-aware tolerance are rejected (near-singular
    systems), as are non-finite inputs/results.
    """
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("homography requires exactly 4 correspondences")
    for p in src + dst:
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            raise ValueError("homography inputs must be finite")
    # Scale-aware thresholds: each point set is validated against its own
    # magnitude, so mixed-unit calibrations (e.g. pixel source, yard engine)
    # are not falsely rejected by one combined scale. The pivot floor guards
    # the whole linear system, so it stays relative to the overall magnitude.
    src_scale = 1.0
    for p in src:
        src_scale = max(src_scale, abs(p.x), abs(p.y))
    dst_scale = 1.0
    for p in dst:
        dst_scale = max(dst_scale, abs(p.x), abs(p.y))
    _check_general_position(src, src_scale)
    _check_general_position(dst, dst_scale)
    pivot_floor = 1e-9 * max(src_scale, dst_scale)

    rows = []
    for s, d in zip(src, dst):
        u, v, X, Y = s.x, s.y, d.x, d.y
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * X, -v * X, X])
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -u * Y, -v * Y, Y])

    n = 8
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < pivot_floor:
            raise ValueError(
                "near-singular homography: correspondences are duplicate, "
                "near-collinear, or otherwise degenerate"
            )
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pv = rows[col][col]
        for r in range(col + 1, n):
            factor = rows[r][col] / pv
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]

    coeffs = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rows[i][n]
        for c in range(i + 1, n):
            s -= rows[i][c] * coeffs[c]
        coeffs[i] = s / rows[i][i]
    if not all(math.isfinite(t) for t in coeffs):
        raise ValueError("homography solve produced non-finite coefficients")
    return Homography(coeffs)


class CoordinateMapper:
    """Translates between raw input coordinates and engine coordinates.

    Raw dicts look like {"x": float, "y": float}. In manual mode "raw" means
    "as the data source declared them, relative to the coordinate system's
    origin". In four_point mode "raw" means source-image coordinates.
    """

    def __init__(self, coordinate_system: CoordinateSystem):
        self.coordinate_system = coordinate_system
        if coordinate_system.mode == "four_point":
            self._to_engine_h = _solve_homography(
                coordinate_system.source_points, coordinate_system.engine_points
            )
            self._from_engine_h = _solve_homography(
                coordinate_system.engine_points, coordinate_system.source_points
            )

    def to_engine(self, raw: dict) -> Point2D:
        if self.coordinate_system.mode == "four_point":
            x, y = self._to_engine_h.apply(raw["x"], raw["y"])
            return Point2D(x, y)
        origin = self.coordinate_system.origin
        return Point2D(raw["x"] - origin.x, raw["y"] - origin.y)

    def from_engine(self, point: Point2D) -> dict:
        if self.coordinate_system.mode == "four_point":
            x, y = self._from_engine_h.apply(point.x, point.y)
            return {"x": x, "y": y}
        origin = self.coordinate_system.origin
        return {"x": point.x + origin.x, "y": point.y + origin.y}


def distance(a: Point2D, b: Point2D) -> float:
    """Euclidean distance in yards."""
    return math.hypot(a.x - b.x, a.y - b.y)


def bearing_deg(a: Point2D, b: Point2D) -> float:
    """Angle (degrees) from point a to point b, 0 = due "east" (+x)."""
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def point_in_polygon(point: Point2D, polygon: List[Point2D]) -> bool:
    """Ray casting: horizontal ray to +infinity in x; odd crossings = inside.

    Points exactly on a vertex may be classified on either side; for this
    application that boundary ambiguity is immaterial because regions are
    authored with a few yards of slack between boundaries.
    """
    inside = False
    x, y = point.x, point.y
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i].x, polygon[i].y
        x2, y2 = polygon[(i + 1) % n].x, polygon[(i + 1) % n].y
        # Count edges that straddle the point's y and cross the ray at x > point.x
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                inside = not inside
    return inside
