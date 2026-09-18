"""Metrics for tracker v2, kept pure so they can be tested without decoding video.

Two rules the previous pass established and this module enforces:

* A frame where the ball is genuinely absent and the tracker says nothing is a
  **success**, not a miss. Silence is only scored as an error when the ball was
  there to be found.
* A prediction on the wrong object is a **false positive**, never a near miss. A
  tracker that reports the shirt where the ball was is worse than one that reports
  nothing, and the numbers have to say so.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

#: score_frame returns a singular outcome; the counters are plural. Mapping them
#: explicitly is cheaper than a KeyError in the one place results are computed.
_COUNT_KEY = {"true_positive": "true_positives", "false_positive": "false_positives",
              "false_negative": "false_negatives", "correct_silence": "correct_silence"}

SYNTHETIC_METRIC_KEYS = ("frames", "frames_with_ball", "frames_without_ball",
                         "true_positives", "false_positives", "false_negatives",
                         "correct_silence", "precision", "recall", "f1",
                         "median_center_error_px", "p90_center_error_px",
                         "false_positive_rate_on_absent")


def match_tolerance(radius: Optional[float], width: int, height: int,
                    radius_multiple: float = 1.5,
                    frame_fraction: float = 0.02) -> float:
    return max(frame_fraction * max(width, height),
               radius_multiple * (radius or 0.0))


def score_frame(prediction: Optional[tuple[float, float]], ground_truth: Optional[tuple[float, float]],
                *, radius: float, width: int, height: int) -> str:
    """Classify one frame: true positive, false positive, false negative or silence."""
    tolerance = match_tolerance(radius, width, height)
    if ground_truth is None:
        return "correct_silence" if prediction is None else "false_positive"
    if prediction is None:
        return "false_negative"
    if math.hypot(prediction[0] - ground_truth[0],
                  prediction[1] - ground_truth[1]) <= tolerance:
        return "true_positive"
    return "false_positive"


def synthetic_sequence_metrics(states: Sequence[Mapping[str, Any]],
                               ground_truth: Sequence[Mapping[str, Any]],
                               *, width: int, height: int) -> dict:
    """Score a tracker result against exact synthetic labels.

    ``states`` are tracker per-frame dicts (with ``state`` and optional ``x``/``y``)
    and ``ground_truth`` are synthetic samples (with ``visibility`` and ``x``/``y``).
    """
    by_index = {int(item["frame_index"]): item for item in states}
    counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0,
              "correct_silence": 0}
    errors: list[float] = []
    frames_with_ball = frames_without_ball = 0
    for truth in ground_truth:
        index = int(truth["frame_index"])
        state = by_index.get(index)
        visible = truth.get("visibility") in ("VISIBLE", "PARTIAL")
        prediction = None
        if state is not None and state.get("state") == "OBSERVED" and \
                state.get("x") is not None:
            prediction = (float(state["x"]), float(state["y"]))
        target = (float(truth["x"]), float(truth["y"])) if visible else None
        outcome = score_frame(prediction, target, radius=float(truth["radius_px"]),
                              width=width, height=height)
        counts[_COUNT_KEY[outcome]] += 1
        if visible:
            frames_with_ball += 1
            if outcome == "true_positive":
                errors.append(math.hypot(prediction[0] - target[0],
                                         prediction[1] - target[1]))
        else:
            frames_without_ball += 1
    precision = counts["true_positives"] / max(
        1, counts["true_positives"] + counts["false_positives"])
    recall = counts["true_positives"] / max(1, frames_with_ball)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {
        "frames": len(ground_truth),
        "frames_with_ball": frames_with_ball,
        "frames_without_ball": frames_without_ball,
        **counts,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_center_error_px": round(float(sorted(errors)[len(errors) // 2]), 2)
        if errors else None,
        "p90_center_error_px": round(float(sorted(errors)[int(0.9 * (len(errors) - 1))]), 2)
        if errors else None,
        "false_positive_rate_on_absent": round(
            counts["false_positives"] / max(1, frames_without_ball), 4),
    }


def aggregate_synthetic_metrics(per_clip: Iterable[Mapping[str, Any]]) -> dict:
    """Pooled metrics, with the per-clip numbers kept alongside them."""
    clips = [clip for clip in per_clip]
    if not clips:
        return {"clips": 0}
    pool = {key: 0 for key in ("frames", "frames_with_ball", "frames_without_ball",
                               "true_positives", "false_positives", "false_negatives",
                               "correct_silence")}
    for clip in clips:
        for key in pool:
            pool[key] += int(clip.get(key) or 0)
    precision = pool["true_positives"] / max(
        1, pool["true_positives"] + pool["false_positives"])
    recall = pool["true_positives"] / max(1, pool["frames_with_ball"])
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    errors = [clip["median_center_error_px"] for clip in clips
              if clip.get("median_center_error_px") is not None]
    return {
        "clips": len(clips),
        **pool,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "median_of_clip_median_center_error_px": round(
            float(sorted(errors)[len(errors) // 2]), 2) if errors else None,
        "false_positive_rate_on_absent": round(
            pool["false_positives"] / max(1, pool["frames_without_ball"]), 4),
        "per_clip": [dict(clip) for clip in clips],
    }


def real_video_diagnostics(summaries: Sequence[Mapping[str, Any]],
                           tracker_results: Sequence[Mapping[str, Any]],
                           *, negatives: Optional[Sequence[Mapping[str, Any]]] = None
                           ) -> dict:
    """Real-video layer: what the tracker does when there is no ground truth.

    These are *not* accuracy numbers and must never be reported as such. They are
    behavioural diagnostics: how often independent methods agree, how often the
    temporal check passes, how often the tracker abstains, and how often it does
    the two things v1 did wrong (body lock, static-scene lock).
    """
    frames = sum(int(summary.get("frames") or 0) for summary in summaries)
    tiers = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "AMBIGUOUS": 0, "UNAVAILABLE": 0}
    diagnostics = {"BODY_LOCK": 0, "CLOTHING_LOCK": 0, "STATIC_SCENE_LOCK": 0,
                   "CLUB_LOCK": 0, "FLAG_LOCK": 0}
    for summary in summaries:
        for tier, count in (summary.get("tiers") or {}).items():
            tiers[tier] = tiers.get(tier, 0) + int(count)
        for kind, count in (summary.get("diagnostics") or {}).items():
            diagnostics[kind] = diagnostics.get(kind, 0) + int(count)

    observed = sum(int(result["confidence"].get("observed_frames") or 0)
                   for result in tracker_results)
    spans = sum(int(result["confidence"].get("span_frames") or 0)
                for result in tracker_results)
    cycles = [result["confidence"].get("cycle_consistency_rate") or 0.0
              for result in tracker_results]
    body = sum(int(result["confidence"].get("body_lock_rate", 0.0)
                   * max(1, int(result["confidence"].get("observed_frames") or 0)))
               for result in tracker_results)
    static = sum(int(result["confidence"].get("static_lock_rate", 0.0)
                     * max(1, int(result["confidence"].get("observed_frames") or 0)))
                 for result in tracker_results)
    accepted = sum(1 for result in tracker_results if result.get("accepted"))
    return {
        "clips": len(tracker_results),
        "frames": frames,
        "consensus_tiers": tiers,
        "consensus_rate": round((tiers["HIGH"] + tiers["MEDIUM"]) / max(1, frames), 4),
        "diagnostics": diagnostics,
        "observed_frames": observed,
        "tracked_spans": spans,
        "coverage": round(observed / max(1, spans), 4),
        "mean_cycle_consistency": round(float(sum(cycles) / max(1, len(cycles))), 4),
        "body_lock_rate": round(body / max(1, observed), 4),
        "static_lock_rate": round(static / max(1, observed), 4),
        "abstention_rate": round(1.0 - accepted / max(1, len(tracker_results)), 4),
        "tracks_accepted": accepted,
        "note": "behavioural diagnostics, not accuracy: these clips have no human "
                "labels, and pseudo-label agreement is not real-world accuracy",
        "negatives": list(negatives or []),
    }


def negative_sequence_report(results: Sequence[Mapping[str, Any]]) -> dict:
    """How often the tracker claims a ball on sequences where none was inserted.

    The measured object here is undeserved confidence. A tracker that accepts
    tracks on empty background is not promotable, whatever its recall elsewhere."""
    fabricated = 0
    observed_frames = 0
    for result in results:
        if result.get("accepted"):
            fabricated += 1
        observed_frames += int(result["confidence"].get("observed_frames") or 0)
    return {
        "sequences": len(results),
        "sequences_with_an_accepted_track": fabricated,
        "fabricated_track_rate": round(fabricated / max(1, len(results)), 4),
        "observed_frames_on_empty_background": observed_frames,
    }
