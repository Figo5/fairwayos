"""Truth-aware summaries for research-only ball-track reports."""

from __future__ import annotations

from typing import Any, Mapping


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
