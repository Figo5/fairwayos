"""Truth-aware summaries for research-only ball-track reports."""

from __future__ import annotations

from typing import Any, Mapping
import math


def evaluate_candidate_quality(
    report: Mapping[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    max_step_per_frame: float = 120.0,
) -> dict:
    """Apply bounded sanity checks to research candidate diagnostics.

    This is a rejection/visualization gate, not a detector or ground-truth
    evaluator.  Rejected frames explicitly disable their marker and trail;
    unavailable frames retain their unavailable metadata.  No production
    eligibility is inferred or changed.
    """
    if not math.isfinite(float(max_step_per_frame)) or max_step_per_frame <= 0:
        raise ValueError("max_step_per_frame must be a finite positive number")
    if image_size is not None:
        width, height = image_size
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive integers")
    else:
        width = height = None

    reasons = set()
    rejected = []
    checked = []
    points = []
    for index, observation in enumerate(report.get("observations", ())):
        ball = observation.get("ball", observation) if isinstance(observation, Mapping) else {}
        point = ball.get("point")
        overlay = ball.get("rendered_overlay", {})
        marker = overlay.get("marker") if isinstance(overlay, Mapping) else None
        has_point = isinstance(point, Mapping) and all(k in point for k in ("x", "y"))
        valid_point = False
        if has_point:
            try:
                x, y = float(point["x"]), float(point["y"])
                valid_point = math.isfinite(x) and math.isfinite(y)
                if valid_point and width is not None:
                    valid_point = 0 <= x <= width and 0 <= y <= height
            except (TypeError, ValueError):
                valid_point = False
        frame_reasons = []
        if marker is True and not valid_point:
            frame_reasons.append("marker_misalignment")
        elif marker is False and valid_point:
            frame_reasons.append("marker_misalignment")
        if has_point and not valid_point:
            frame_reasons.append("marker_misalignment")
        if frame_reasons:
            reasons.update(frame_reasons)
            rejected.append(index)
        else:
            checked.append(index)
        # Only observed coordinates participate in temporal checks; predicted
        # coordinates are visualization extrapolations, not new evidence.
        if valid_point and ball.get("state") == "observed":
            try:
                frame = int(observation.get("frame_index", index))
            except (TypeError, ValueError):
                frame = index
            points.append((frame, x, y, index))

    max_step = 0.0
    for previous, current in zip(points, points[1:]):
        frame_delta = current[0] - previous[0]
        if frame_delta <= 0:
            continue
        step = math.hypot(current[1] - previous[1], current[2] - previous[2])
        max_step = max(max_step, step)
        if step > float(max_step_per_frame) * frame_delta:
            reasons.add("low_temporal_plausibility")
            if current[3] not in rejected:
                rejected.append(current[3])

    rejected.sort()
    rejected_set = set(rejected)
    observation_decisions = []
    for index, observation in enumerate(report.get("observations", ())):
        ball = observation.get("ball", observation) if isinstance(observation, Mapping) else {}
        observation_decisions.append({
            "observation_index": index,
            "state": ball.get("state", "unavailable"),
            "rejected": index in rejected_set,
            # Consumers must use these flags rather than rendering a rejected
            # candidate's stale marker or extending its trail.
            "overlay": {"marker": index not in rejected_set, "trail": index not in rejected_set},
        })
    return {
        "passed": not reasons,
        "status": "research_candidate" if not reasons else "rejected",
        "reasons": tuple(sorted(reasons)),
        "metrics": {"observed_point_count": len(points), "max_step_pixels": round(max_step, 6)},
        "rejected_observation_indices": tuple(rejected),
        "observation_decisions": tuple(observation_decisions),
        "ground_truth_available": False,
        "production_eligible": False,
        "research_only": True,
    }


def summarize_track_report(report: Mapping[str, Any]) -> dict:
    """Summarize temporal behavior without fabricating precision or recall."""
    observations = list(report.get("observations", ()))
    frame_count = len(observations)
    accepted = [item for item in observations if item.get("point") is not None]
    termination_count = sum(item.get("state") == "terminated" for item in observations)
    reacquisition_count = sum(item.get("state") == "reacquired" for item in observations)
    post_termination_candidate_rejections = sum(
        item.get("state") == "terminated" and int(item.get("candidate_count", 0)) > 0
        for item in observations
    )
    longest_active_run = 0
    current_run = 0
    for item in observations:
        if item.get("point") is not None:
            current_run += 1
            longest_active_run = max(longest_active_run, current_run)
        else:
            current_run = 0
    return {
        "frame_count": frame_count,
        "accepted_frame_count": len(accepted),
        "coverage": round(len(accepted) / frame_count, 6) if frame_count else 0.0,
        "longest_active_run": longest_active_run,
        "termination_count": termination_count,
        "reacquisition_count": reacquisition_count,
        "post_termination_candidate_rejections": post_termination_candidate_rejections,
        "false_positive_rate": None,
        "ground_truth_available": False,
        "truth_boundary": "no_reviewed_frame_labels",
    }
