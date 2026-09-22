"""Scoring a predicted ball track against human ground truth.

The scoring rules were written down in ``docs/ball-tracking-results.md``
before any method was run, and this module implements them literally. Two of
them matter more than the rest:

* A frame where the ball is genuinely not visible and the tracker says nothing
  is a **success**, not a miss. Counting invisible frames as tracking failures
  would reward a tracker that guesses.
* A prediction that lands on the tee, the flagstick, a shoe or a decoy ball
  while the real ball is elsewhere is a **false positive**, not merely an
  imprecise hit. A wrong-object lock is the failure this whole research pass
  exists to detect, so it is never allowed to look like a near miss.

Frames the annotator flagged ``ambiguous`` are excluded from precision, recall
and centre error, and reported separately rather than quietly dropped.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

#: Match tolerance: the larger of a resolution-relative floor and a
#: ball-size-relative term. Declared in the research plan before measurement.
TOLERANCE_FRAME_FRACTION = 0.02
TOLERANCE_RADIUS_MULTIPLE = 1.5

#: A predicted centre that stays inside this radius across consecutive frames is
#: treated as the same physical spot for static-lock accounting.
STATIC_LOCK_PIXELS = 2.0


def match_tolerance(*, width: int, height: int, radius: Optional[float]) -> float:
    """Pixel tolerance for calling a prediction a hit on this clip's scale."""
    floor = TOLERANCE_FRAME_FRACTION * min(width, height)
    if radius is None:
        return floor
    return max(floor, TOLERANCE_RADIUS_MULTIPLE * float(radius))


def _predicted_points(observations: Iterable[Mapping]) -> dict[int, tuple[float, float, str]]:
    """Frame -> (x, y, state) for states that actually assert a position.

    ``interpolated`` is deliberately excluded from scoring: it is not an
    observation, and letting it earn a true positive would be exactly the
    stale-coordinate credit this evaluation is meant to refuse.
    """
    points: dict[int, tuple[float, float, str]] = {}
    for observation in observations:
        state = observation.get("state")
        if state != "observed":
            continue
        x, y = observation.get("x"), observation.get("y")
        if x is None or y is None:
            continue
        points[int(observation["frame_index"])] = (float(x), float(y), state)
    return points


def evaluate_clip(annotation: Mapping, observations: Sequence[Mapping],
                  *, frames: Optional[Sequence[int]] = None) -> dict:
    """Score one clip. ``frames`` restricts scoring to the labelled frames."""
    width, height = int(annotation["width"]), int(annotation["height"])
    labels = annotation["frames"]
    scored = sorted(int(key) for key in labels) if frames is None else sorted(frames)
    predicted = _predicted_points(observations)

    true_positive = false_positive = false_negative = true_negative = 0
    wrong_object = 0
    ambiguous = 0
    errors: list[float] = []
    per_frame: list[dict] = []

    for index in scored:
        label = labels.get(str(index))
        if label is None:
            continue
        point = predicted.get(index)
        if label.get("ambiguous"):
            ambiguous += 1
            per_frame.append({"frame": index, "outcome": "ambiguous",
                              "predicted": bool(point), "visible": bool(label["visible"])})
            continue
        if label["visible"]:
            tolerance = match_tolerance(width=width, height=height, radius=label.get("radius"))
            if point is None:
                false_negative += 1
                per_frame.append({"frame": index, "outcome": "false_negative"})
                continue
            error = math.hypot(point[0] - label["x"], point[1] - label["y"])
            if error <= tolerance:
                true_positive += 1
                errors.append(error)
                per_frame.append({"frame": index, "outcome": "true_positive", "error_px": error})
            else:
                false_positive += 1
                wrong_object += 1
                per_frame.append({"frame": index, "outcome": "wrong_object", "error_px": error})
        else:
            if point is None:
                true_negative += 1
                per_frame.append({"frame": index, "outcome": "true_negative"})
            else:
                false_positive += 1
                per_frame.append({"frame": index, "outcome": "false_positive_on_absent"})

    visible = true_positive + false_negative + wrong_object
    not_visible = true_negative + (false_positive - wrong_object)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / visible if visible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "clip_id": annotation["clip_id"],
        "width": width, "height": height,
        "scored_frames": len(per_frame),
        "ambiguous_frames": ambiguous,
        "visible_frames": visible,
        "not_visible_frames": not_visible,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_error_px": _percentile(errors, 50),
        "p90_error_px": _percentile(errors, 90),
        "fp_rate_on_absent": ((false_positive - wrong_object) / not_visible) if not_visible else 0.0,
        "wrong_object_rate_on_visible": (wrong_object / visible) if visible else 0.0,
        "unavailable_rate": 1.0 - (len(predicted) / len(per_frame)) if per_frame else 1.0,
        **_track_metrics(labels, predicted, scored, width=width, height=height),
        "per_frame": per_frame,
    }


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (percentile / 100.0) * (len(ordered) - 1)
    low, high = int(math.floor(position)), int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _track_metrics(labels: Mapping, predicted: Mapping[int, tuple[float, float, str]],
                   scored: Sequence[int], *, width: int, height: int) -> dict:
    """Continuity, identity switches and static-lock length."""
    correct_run = longest_correct = 0
    switches = 0
    previous_correct: Optional[bool] = None
    lock_run = longest_lock = 0
    lock_anchor: Optional[tuple[float, float]] = None

    for index in scored:
        label = labels.get(str(index))
        if label is None or label.get("ambiguous"):
            correct_run = 0
            continue
        point = predicted.get(index)
        if label["visible"]:
            tolerance = match_tolerance(width=width, height=height, radius=label.get("radius"))
            correct = point is not None and math.hypot(point[0] - label["x"], point[1] - label["y"]) <= tolerance
        else:
            correct = point is None

        correct_run = correct_run + 1 if correct else 0
        longest_correct = max(longest_correct, correct_run)
        if point is not None and previous_correct is not None and previous_correct != correct:
            switches += 1
        if point is not None:
            previous_correct = correct

        # A long run of predictions pinned to one spot while the truth says the
        # ball is absent or has moved is the signature of a wrong-object lock,
        # and is reported as its own number because one long lock is worse than
        # the same count of scattered errors.
        if point is not None and not correct:
            if lock_anchor is not None and math.hypot(point[0] - lock_anchor[0],
                                                      point[1] - lock_anchor[1]) <= STATIC_LOCK_PIXELS:
                lock_run += 1
            else:
                lock_run = 1
            lock_anchor = (point[0], point[1])
            longest_lock = max(longest_lock, lock_run)
        else:
            lock_run, lock_anchor = 0, None

    return {
        "longest_correct_run": longest_correct,
        "identity_switches": switches,
        "longest_static_lock": longest_lock,
    }


def aggregate(results: Sequence[Mapping]) -> dict:
    """Pool per-clip counts. Per-clip numbers are always reported alongside."""
    keys = ("true_positive", "false_positive", "false_negative", "true_negative",
            "visible_frames", "not_visible_frames", "ambiguous_frames", "scored_frames")
    totals = {key: sum(int(result[key]) for result in results) for key in keys}
    wrong_object = sum(int(round(result["wrong_object_rate_on_visible"] * result["visible_frames"]))
                       for result in results)
    precision_denominator = totals["true_positive"] + totals["false_positive"]
    precision = totals["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = totals["true_positive"] / totals["visible_frames"] if totals["visible_frames"] else 0.0
    errors = [value for result in results for value in _clip_errors(result)]
    return {
        **totals,
        "clips": len(results),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "median_error_px": _percentile(errors, 50),
        "p90_error_px": _percentile(errors, 90),
        "wrong_object_frames": wrong_object,
        "fp_frames_on_absent": totals["false_positive"] - wrong_object,
        "fp_rate_on_absent": ((totals["false_positive"] - wrong_object) / totals["not_visible_frames"])
                             if totals["not_visible_frames"] else 0.0,
        "wrong_object_rate_on_visible": (wrong_object / totals["visible_frames"])
                                        if totals["visible_frames"] else 0.0,
        "longest_static_lock": max((int(result["longest_static_lock"]) for result in results), default=0),
        "identity_switches": sum(int(result["identity_switches"]) for result in results),
    }


def _clip_errors(result: Mapping) -> list[float]:
    return [float(row["error_px"]) for row in result.get("per_frame", ())
            if row.get("outcome") == "true_positive"]


__all__ = ["match_tolerance", "evaluate_clip", "aggregate",
           "TOLERANCE_FRAME_FRACTION", "TOLERANCE_RADIUS_MULTIPLE", "STATIC_LOCK_PIXELS"]
