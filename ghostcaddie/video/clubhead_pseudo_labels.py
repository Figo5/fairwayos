"""Research-only pseudo-labels for provisional clubhead experiments.

Every emitted label is explicitly pseudo and must remain outside production.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

PROVENANCE = {
    "pseudo_label": True,
    "ground_truth": False,
    "research_only": True,
    "production_eligible": False,
}


def _point(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(v) for v in point) else None


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _base(frame_index: int, warnings: Sequence[str], confidence: float = 0.0, uncertainty: Optional[float] = None) -> dict:
    return {
        "frame_index": int(frame_index),
        "available": False,
        "clubhead": {"value": None, "visibility": "unavailable", "source": "unavailable"},
        "shaft": {"value": None, "visibility": "unavailable", "source": "unavailable"},
        "confidence": round(float(confidence), 6),
        "uncertainty_px": None if uncertainty is None else round(float(uncertainty), 4),
        "evidence": [],
        "warnings": list(warnings),
        "impact_window": None,
        "ball_club_alignment": None,
        "provenance": dict(PROVENANCE),
    }


def build_pseudo_label(
    *,
    frame_index: int,
    image_size: Tuple[int, int],
    candidate: Mapping[str, Any],
    pose: Mapping[str, Any],
    ball_point: Optional[Sequence[float]],
    flow_vector: Optional[Sequence[float]],
    previous_point: Optional[Sequence[float]],
) -> dict:
    """Build a provisional label or an explicit unavailable rejection."""
    width, height = image_size
    if int(frame_index) != frame_index or frame_index < 0 or width <= 0 or height <= 0:
        raise ValueError("invalid frame or image size")
    point = _point(candidate.get("point"))
    wrist = _point(pose.get("wrist")) if isinstance(pose, Mapping) else None
    ball = _point(ball_point)
    flow = _point(flow_vector)
    previous = _point(previous_point)
    confidence = float(candidate.get("confidence", 0.0))
    uncertainty = candidate.get("uncertainty_px")
    try:
        uncertainty = float(uncertainty)
    except (TypeError, ValueError):
        uncertainty = math.inf
    warnings = []
    if point is None or not (0 <= point[0] <= width and 0 <= point[1] <= height):
        warnings.append("point_invalid_or_out_of_bounds")
    if not math.isfinite(confidence) or confidence < 0.65:
        warnings.append("candidate_confidence_below_pseudo_threshold")
    if not math.isfinite(uncertainty) or uncertainty > 80.0:
        warnings.append("candidate_uncertainty_exceeds_pseudo_threshold")
    if wrist is None:
        warnings.append("pose_wrist_unavailable")
    if ball is None:
        warnings.append("ball_track_unavailable")
    if point is not None and ball is not None and _distance(point, ball) > 250.0:
        warnings.append("ball_club_relation_inconsistent")
    if point is not None and previous is not None and flow is not None:
        actual = (point[0] - previous[0], point[1] - previous[1])
        if _distance(actual, flow) > 45.0:
            warnings.append("temporal_motion_inconsistent")
    if warnings:
        result = _base(frame_index, ["pseudo_label_rejected", *sorted(set(warnings))], confidence, uncertainty)
        result["evidence"] = sorted(set(candidate.get("evidence", ())))
        return result
    relation = max(0.0, 1.0 - _distance(point, ball) / 250.0)
    motion = 1.0
    if previous is not None and flow is not None:
        actual = (point[0] - previous[0], point[1] - previous[1])
        motion = max(0.0, 1.0 - _distance(actual, flow) / 45.0)
    pose_confidence = min(1.0, max(0.0, float(pose.get("confidence", 0.0))))
    final_confidence = min(1.0, confidence * pose_confidence * (0.7 + 0.3 * relation) * (0.7 + 0.3 * motion))
    evidence = set(candidate.get("evidence", ()))
    evidence.update(("pose", "ball_relation", "motion_consistency", "temporal_consistency"))
    result = _base(frame_index, (), final_confidence, uncertainty)
    result.update({
        "available": True,
        "clubhead": {"value": {"x": round(point[0], 4), "y": round(point[1], 4)}, "visibility": "visible", "source": "pseudo_label"},
        "shaft": {"value": {"grip": {"x": round(wrist[0], 4), "y": round(wrist[1], 4)}, "neck": {"x": round(point[0], 4), "y": round(point[1], 4)}}, "visibility": "visible", "source": "pseudo_label"},
        "evidence": sorted(evidence),
        "ball_club_alignment": round(_distance(point, ball), 4),
        "warnings": ["shaft_is_wrist_to_clubhead_proxy", "research_only_pseudo_geometry"],
    })
    return result


def estimate_impact_window(labels: Sequence[Mapping[str, Any]], fps: float, *, neighborhood: int = 2) -> dict:
    """Return a pixel-space proximity bracket; never an exact impact event."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    aligned = [(int(label["frame_index"]), float(label["ball_club_alignment"])) for label in labels if label.get("available") and label.get("ball_club_alignment") is not None]
    if not aligned:
        return {"available": False, "start_frame": None, "end_frame": None, "exact_impact": None, "warning": "no_pseudo_alignment"}
    frame, _ = min(aligned, key=lambda item: item[1])
    return {"available": True, "start_frame": max(0, frame - neighborhood), "end_frame": frame + neighborhood, "exact_impact": None, "fps": float(fps), "warning": "pseudo_proximity_bracket_not_ground_truth"}
