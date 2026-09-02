"""Research-only ROI helpers for Luna-guided golf-ball experiments.

This module never produces production observations or ground truth. Luna labels
are guidance for a bounded search region; every selected result remains a
pseudo-label and must be visually reviewed before any use.
"""

import math


RESEARCH_FLAGS = {
    "pseudo_label": True,
    "ground_truth": False,
    "research_only": True,
    "production_eligible": False,
}


def _inside(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _overlap(point, radius, box):
    x, y = point
    nx = min(max(x, box[0]), box[2])
    ny = min(max(y, box[1]), box[3])
    return math.hypot(x - nx, y - ny) <= radius


def validate_pseudo_labels(labels):
    """Validate Luna's fixed research-only provenance on every record."""
    if not isinstance(labels, list):
        raise ValueError("pseudo-label records must be a list")
    for label in labels:
        if not isinstance(label, dict) or any(label.get(k) != v for k, v in RESEARCH_FLAGS.items()):
            raise ValueError("every Luna label must retain research-only flags")
    return labels


def select_temporal_track(candidates, *, roi, golfer_boxes=None, club_points=None,
                          camera_shift=(0.0, 0.0), min_consecutive=2,
                          max_step=100.0):
    """Select only a bounded, moving, consecutive candidate sequence.

    ``camera_shift`` is a measured global translation to subtract before the
    step test. No candidate is promoted when a person/club overlap, size, ROI,
    ambiguity, or continuity gate fails.
    """
    if min_consecutive < 2 or max_step <= 0:
        raise ValueError("invalid temporal limits")
    golfer_boxes = golfer_boxes or []
    club_points = club_points or []
    accepted = []
    warnings = []
    for item in candidates:
        point = item.get("center") if isinstance(item, dict) else None
        radius = item.get("radius_px") if isinstance(item, dict) else None
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            warnings.append("malformed_candidate")
            continue
        try:
            point = (float(point[0]), float(point[1]))
            radius = float(radius)
        except (TypeError, ValueError):
            warnings.append("malformed_candidate")
            continue
        if not (math.isfinite(point[0]) and math.isfinite(point[1]) and math.isfinite(radius)):
            warnings.append("nonfinite_candidate")
            continue
        if radius < 2.0 or radius > 80.0:
            warnings.append("ball_size_out_of_bounds")
            continue
        if not _inside(point, roi):
            warnings.append("outside_luna_roi")
            continue
        if any(_overlap(point, radius, box) for box in golfer_boxes):
            warnings.append("person_or_club_region")
            continue
        if any(math.hypot(point[0] - p[0], point[1] - p[1]) <= max(2.0 * radius, 25.0)
               for p in club_points):
            warnings.append("person_or_club_region")
            continue
        accepted.append({**item, "center": [point[0], point[1]], "radius_px": radius})
    accepted.sort(key=lambda item: item["frame_index"])
    run = []
    best = []
    previous = None
    dx, dy = camera_shift
    for item in accepted:
        if previous is not None:
            frame_gap = item["frame_index"] - previous["frame_index"]
            step = math.hypot(item["center"][0] - previous["center"][0] - dx,
                              item["center"][1] - previous["center"][1] - dy)
            if frame_gap != 1 or step > max_step:
                if len(run) > len(best):
                    best = run
                run = []
        run.append(item)
        previous = item
    if len(run) > len(best):
        best = run
    if len(best) < min_consecutive:
        warnings.append("consecutive_support_required")
    if not best:
        warnings.append("no_candidate_survived_gates")
    return {
        **RESEARCH_FLAGS,
        "state": "unavailable",
        "track": best if len(best) >= min_consecutive else [],
        "warnings": sorted(set(warnings)),
        "uncertainty_px": None,
    }
