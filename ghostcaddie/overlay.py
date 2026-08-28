"""The ONLY rendering module in the project.

Consumes finished CandidateResult / Recommendation / CourseModel / ShotEvent
data and emits an SVG string via pure f-strings — zero image libraries.
Deliberately does NOT import dispersion/hazards/expected_strokes/simulation:
those modules describe how the numbers were made; this module only draws the
final decision. (CandidateResult is duck-typed via a local Protocol instead
of imported, so this module cannot accidentally pull the engine in.)

Only the final decision is drawn — never the raw per-sample Monte Carlo
landing points.
"""

from typing import Dict, List, Protocol

from .geometry import Point2D
from .models import CourseModel, Recommendation, ShotEvent


class _CandidateLike(Protocol):
    label: str
    club: str
    aim_point: Point2D


class _CandidateResultLike(Protocol):
    """Structural shape of a finished simulation candidate result."""

    expected_strokes: float
    hazard_probabilities: Dict[str, float]
    candidate: _CandidateLike


def _polygon_points_str(polygon: List[Point2D]) -> str:
    return " ".join(f"{p.x:.1f},{p.y:.1f}" for p in polygon)


def render_svg(
    course: CourseModel,
    shot: ShotEvent,
    results: List[_CandidateResultLike],
    recommendation: Recommendation,
    width_px: int = 900,
    height_px: int = 600,
) -> str:
    """Render a course/shots SVG overlay (engine y-up -> SVG y-down).

    The y-axis flip (engine y-up vs SVG y-down) is a standard, arbitrary
    screen convention.
    """
    all_points: List[Point2D] = [course.pin_position, shot.start_position,
                                 shot.actual_landing_position, recommendation.recommended_target]
    for result in results:
        all_points.append(result.candidate.aim_point)
    for poly_list in (
        course.fairway,
        course.green,
        course.bunkers,
        course.water_hazards,
        course.out_of_bounds,
    ):
        for poly in poly_list:
            all_points.extend(poly)

    xs = [p.x for p in all_points]
    ys = [p.y for p in all_points]
    padding = 20.0
    x_min, x_max = min(xs) - padding, max(xs) + padding
    y_min, y_max = min(ys) - padding, max(ys) + padding

    x_scale = (width_px - 2 * padding) / max(x_max - x_min, 1e-9)
    y_scale = (height_px - 2 * padding) / max(y_max - y_min, 1e-9)
    scale = min(x_scale, y_scale)  # uniform, preserve aspect ratio
    offset_x = (width_px - (x_max - x_min) * scale) / 2
    offset_y = (height_px - (y_max - y_min) * scale) / 2

    def sx(p: Point2D) -> float:
        return offset_x + (p.x - x_min) * scale

    def sy(p: Point2D) -> float:
        # Flip y: engine up is SVG down.
        return offset_y + (y_max - p.y) * scale

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
    )
    parts.append('<rect width="100%" height="100%" fill="#eef3ea"/>')

    def draw_region(polygons: List[List[Point2D]], fill: str, stroke: str, sw: float) -> None:
        for poly in polygons:
            pts = " ".join(f"{sx(p):.1f},{sy(p):.1f}" for p in poly)
            parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    draw_region(course.fairway, "#d8e8c0", "#a8c080", 1.5)
    draw_region(course.green, "#9fc97f", "#6f9f4f", 1.5)
    draw_region(course.bunkers, "#e8d9a0", "#c0a870", 1.5)
    draw_region(course.water_hazards, "#a8c8e8", "#7fa8d8", 1.5)
    # OB: red outline, no fill.
    for poly in course.out_of_bounds:
        pts = " ".join(f"{sx(p):.1f},{sy(p):.1f}" for p in poly)
        parts.append(f'<polygon points="{pts}" fill="none" stroke="#d03030" stroke-width="2"/>')

    # Pin marker (cross).
    px, py = sx(course.pin_position), sy(course.pin_position)
    parts.append(
        f'<g stroke="#202020" stroke-width="3"><line x1="{px-6:.1f}" y1="{py:.1f}" '
        f'x2="{px+6:.1f}" y2="{py:.1f}"/><line x1="{px:.1f}" y1="{py-6:.1f}" '
        f'x2="{px:.1f}" y2="{py+6:.1f}"/></g>'
    )

    # Actual shot: start -> actual landing.
    ax, ay = sx(shot.start_position), sy(shot.start_position)
    bx, by = sx(shot.actual_landing_position), sy(shot.actual_landing_position)
    parts.append(
        f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="#d03030" stroke-width="2" stroke-dasharray="4,3"/>'
    )
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4" fill="#d03030"/>')
    parts.append(f'<text x="{bx + 6:.1f}" y="{by - 6:.1f}" font-size="12" fill="#d03030">Actual</text>')

    # Recommended target.
    rx, ry = sx(recommendation.recommended_target), sy(recommendation.recommended_target)
    parts.append(
        f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{rx:.1f}" y2="{ry:.1f}" '
        f'stroke="#1a6b2a" stroke-width="2.5" stroke-dasharray="6,3"/>'
    )
    parts.append(f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="5" fill="none" stroke="#1a6b2a" stroke-width="2"/>')
    parts.append(f'<text x="{rx + 6:.1f}" y="{ry - 5:.1f}" font-size="11" fill="#1a6b2a">Recommended</text>')

    # Decision summary text.
    parts.append(
        f'<text x="{width_px - 10}" y="20" text-anchor="end" font-size="13" fill="#202020" '
        f'font-family="sans-serif">'
        f'Recommended: {recommendation.recommended_club} → {recommendation.expected_strokes:.1f} strokes | '
        f'Actual: {shot.club} → {recommendation.actual_expected_strokes:.1f} strokes | '
        f'Δ {recommendation.decision_cost:+.1f}'
        f'</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
