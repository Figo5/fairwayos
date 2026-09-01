"""Coordinate normalization for local research model outputs."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

_CONFIDENCE_SEMANTICS = "detection_quality_not_identity"
_MAX_UNCERTAINTY_WINDOW = 32
# A centroid radius above this fixed research-only threshold is reported as
# wide (potentially unstable); it is not an identity or tracking decision.
_WIDE_SPREAD_THRESHOLD_PX = 50.0
_TEMPORAL_UNCERTAINTY_SCHEMA_VERSION = 1
_SPREAD_STATUS_CODES = {
    "unavailable": 0,
    "bounded": 1,
    "wide": 2,
}
_SPREAD_STATUS_DESCRIPTIONS = {
    "bounded": "centroid radius at or below threshold",
    "wide": "centroid radius above threshold",
    "unavailable": "no finite points available",
}


def validate_spread_status_consistency(spread_status, spread_status_code) -> bool:
    """Return whether a status/code pair matches the research enumeration."""
    return (
        isinstance(spread_status, str)
        and type(spread_status_code) is int
        and _SPREAD_STATUS_CODES.get(spread_status) == spread_status_code
    )


def aggregate_temporal_uncertainty(observations: Iterable[dict], *, window: int = 5):
    """Summarize recent research observations without asserting identity.

    Only observations with finite two-dimensional points contribute.  The
    window is capped so a long history cannot turn this diagnostic into an
    unbounded aggregate.  A centroid radius at or below the documented
    ``50.0`` pixel threshold is ``bounded``; a larger radius is ``wide``
    (potentially unstable).  If no point is available, the summary remains
    explicitly unavailable rather than synthesizing a location or confidence.
    """
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= _MAX_UNCERTAINTY_WINDOW:
        raise ValueError(f"window must be an integer between 1 and {_MAX_UNCERTAINTY_WINDOW}")
    recent = list(observations)[-window:]
    usable = []
    for observation in recent:
        if not isinstance(observation, dict):
            continue
        point = observation.get("point")
        if not isinstance(point, dict):
            continue
        try:
            x, y = float(point["x"]), float(point["y"])
            confidence = float(observation.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            continue
        usable.append((observation, (x, y), max(0.0, min(1.0, confidence))))
    base = {
        "schema_version": _TEMPORAL_UNCERTAINTY_SCHEMA_VERSION,
        "sample_count": len(usable),
        "observed_count": sum(item[0].get("state") == "observed" for item in usable),
        "predicted_count": sum(item[0].get("state") == "predicted" for item in usable),
        "confidence_range": None,
        "spatial_radius_px": None,
        "confidence_semantics": _CONFIDENCE_SEMANTICS,
        "identity": "unavailable",
        "window_limit": window,
        "window_size": len(recent),
        "window_bounded": bool(usable),
        "spread_status": "unavailable",
        "spread_status_code": _SPREAD_STATUS_CODES["unavailable"],
        "spread_status_code_values": sorted(_SPREAD_STATUS_CODES.values()),
        "spread_status_consistent": validate_spread_status_consistency(
            "unavailable", _SPREAD_STATUS_CODES["unavailable"]
        ),
        "spread_status_description": _SPREAD_STATUS_DESCRIPTIONS["unavailable"],
        "spread_threshold_px": _WIDE_SPREAD_THRESHOLD_PX,
        "spread_available": bool(usable),
    }
    if not usable:
        return {**base, "state": "unavailable", "provenance": "unavailable"}
    points = [item[1] for item in usable]
    confidences = [item[2] for item in usable]
    centroid = (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
    spatial_radius = max(math.hypot(point[0] - centroid[0], point[1] - centroid[1]) for point in points)
    spread_status = "wide" if spatial_radius > _WIDE_SPREAD_THRESHOLD_PX else "bounded"
    base.update({
        "state": "available",
        "provenance": "research_temporal_aggregation",
        "confidence_range": (min(confidences), max(confidences)),
        "spatial_radius_px": spatial_radius,
        "spread_status": spread_status,
        "spread_status_code": _SPREAD_STATUS_CODES[spread_status],
        "spread_status_consistent": validate_spread_status_consistency(
            spread_status, _SPREAD_STATUS_CODES[spread_status]
        ),
        "spread_status_description": _SPREAD_STATUS_DESCRIPTIONS[spread_status],
    })
    return base


class ResearchBallTrack:
    """Bounded, research-only temporal gate for model ball candidates."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.35,
        max_step: float = 110.0,
        max_misses: int = 2,
        confidence_decay: float = 0.2,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not math.isfinite(float(max_step)) or max_step <= 0 or max_misses < 1 or not 0.0 <= confidence_decay <= 1.0:
            raise ValueError("invalid track limits")
        self.min_confidence = float(min_confidence)
        self.max_step = float(max_step)
        self.max_misses = int(max_misses)
        self.confidence_decay = float(confidence_decay)
        self._active = False
        self._terminated = False
        self._point: Tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._confidence = 0.0
        self._misses = 0

    @staticmethod
    def _result(state, point, confidence, warning=None):
        return {
            "state": state,
            "point": None if point is None else {"x": float(point[0]), "y": float(point[1])},
            "confidence": float(confidence),
            "confidence_semantics": _CONFIDENCE_SEMANTICS,
            "warning": warning,
        }

    def update(self, candidates: Iterable[dict]):
        candidates = list(candidates)
        if self._terminated:
            return self._result("terminated", None, 0.0, "track_terminated")
        valid = _valid_candidates(candidates, self.min_confidence)
        if self._point is None:
            if not valid:
                return self._result("unavailable", None, 0.0, _candidate_status_warning(candidates))
            chosen = max(valid, key=lambda item: float(item["confidence"]))
            self._point = tuple(float(value) for value in chosen["center"])
            self._confidence = float(chosen["confidence"])
            self._active = True
            self._misses = 0
            return self._result("observed", self._point, self._confidence)

        predicted = (self._point[0] + self._velocity[0], self._point[1] + self._velocity[1])
        allowed = self.max_step + min(self.max_step, math.hypot(*self._velocity))
        ranked = []
        for candidate in valid:
            center = tuple(float(value) for value in candidate["center"])
            distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
            ranked.append((distance, -float(candidate["confidence"]), candidate, center))
        if ranked:
            distance, _, chosen, center = min(ranked)
            if distance <= allowed:
                delta = (center[0] - self._point[0], center[1] - self._point[1])
                self._velocity = ((self._velocity[0] + delta[0]) / 2.0, (self._velocity[1] + delta[1]) / 2.0)
                self._point = center
                self._confidence = float(chosen["confidence"])
                self._misses = 0
                return self._result("observed", self._point, self._confidence)
            warning = "motion_constraint_rejected"
        else:
            warning = "no_valid_candidate"

        self._misses += 1
        self._confidence = max(0.0, self._confidence - self.confidence_decay)
        if self._misses >= self.max_misses:
            self._terminated = True
            self._active = False
            return self._result("terminated", None, 0.0, "track_terminated")
        self._point = predicted
        return self._result("predicted", self._point, self._confidence, "confidence_decayed" if warning == "no_valid_candidate" else warning)

class ResearchBallMultiHypothesisTrack:
    """Research-only tracker with bounded ambiguity and guarded reacquisition."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.35,
        reacquire_confidence: float = 0.75,
        max_step: float = 80.0,
        max_misses: int = 2,
        max_hypotheses: int = 3,
        ambiguity_margin: float = 0.05,
    ) -> None:
        if not 0.0 <= min_confidence <= reacquire_confidence <= 1.0:
            raise ValueError("invalid confidence thresholds")
        if (not math.isfinite(float(max_step)) or max_step <= 0 or max_misses < 1 or
                max_hypotheses < 1 or not math.isfinite(float(ambiguity_margin)) or
                not 0.0 <= ambiguity_margin <= 1.0):
            raise ValueError("invalid track limits")
        self.min_confidence = float(min_confidence)
        self.reacquire_confidence = float(reacquire_confidence)
        self.max_step = float(max_step)
        self.max_misses = int(max_misses)
        self.max_hypotheses = int(max_hypotheses)
        self.ambiguity_margin = float(ambiguity_margin)
        self._hypotheses = []
        self._terminated = False
        self._pending_reacquisition = None

    @staticmethod
    def _result(state, point, confidence, warning=None, hypothesis_count=0):
        return {
            "state": state,
            "point": None if point is None else {"x": float(point[0]), "y": float(point[1])},
            "confidence": float(confidence),
            "confidence_semantics": _CONFIDENCE_SEMANTICS,
            "warning": warning,
            "hypothesis_count": int(hypothesis_count),
        }

    @staticmethod
    def _center(candidate):
        return tuple(float(value) for value in candidate["center"])

    def _reacquire(self, candidates):
        strong = [candidate for candidate in candidates if float(candidate.get("confidence", 0.0)) >= self.reacquire_confidence]
        if self._pending_reacquisition is not None and len(strong) == 1:
            center = self._center(strong[0])
            distance = math.hypot(center[0] - self._pending_reacquisition[0], center[1] - self._pending_reacquisition[1])
            if distance <= self.max_step:
                self._hypotheses = [{"point": center, "velocity": (0.0, 0.0), "confidence": float(strong[0]["confidence"]), "score": float(strong[0]["confidence"]), "misses": 0}]
                self._pending_reacquisition = None
                self._terminated = False
                return self._result("reacquired", center, strong[0]["confidence"], hypothesis_count=1)
        if len(strong) == 1:
            self._pending_reacquisition = self._center(strong[0])
            return self._result("terminated", None, 0.0, "reacquisition_pending")
        self._pending_reacquisition = None
        warning = "reacquisition_ambiguous" if len(strong) > 1 else "reacquisition_insufficient_evidence"
        return self._result("terminated", None, 0.0, warning)

    def update(self, candidates: Iterable[dict]):
        candidates = list(candidates)
        valid = _valid_candidates(candidates, self.min_confidence)
        if self._terminated:
            return self._reacquire(valid)
        if not self._hypotheses:
            if not valid:
                return self._result("unavailable", None, 0.0, _candidate_status_warning(candidates))
            self._hypotheses = [
                {"point": self._center(candidate), "velocity": (0.0, 0.0), "confidence": float(candidate["confidence"]), "score": float(candidate["confidence"]), "misses": 0}
                for candidate in sorted(valid, key=lambda item: float(item["confidence"]), reverse=True)[: self.max_hypotheses]
            ]
            best = max(self._hypotheses, key=lambda item: item["score"])
            return self._result("observed", best["point"], best["confidence"], hypothesis_count=len(self._hypotheses))

        branches = []
        for hypothesis in self._hypotheses:
            predicted = (hypothesis["point"][0] + hypothesis["velocity"][0], hypothesis["point"][1] + hypothesis["velocity"][1])
            allowed = self.max_step + min(self.max_step, math.hypot(*hypothesis["velocity"]))
            matched = False
            for candidate in valid:
                center = self._center(candidate)
                distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
                if distance <= allowed:
                    confidence = float(candidate["confidence"])
                    delta = (center[0] - hypothesis["point"][0], center[1] - hypothesis["point"][1])
                    branches.append({"point": center, "velocity": ((hypothesis["velocity"][0] + delta[0]) / 2.0, (hypothesis["velocity"][1] + delta[1]) / 2.0), "confidence": confidence, "score": hypothesis["score"] * 0.7 + confidence * 0.3 - distance / max(self.max_step, 1.0) * 0.05, "misses": 0})
                    matched = True
            if not matched and hypothesis["misses"] + 1 < self.max_misses:
                branches.append({"point": predicted, "velocity": hypothesis["velocity"], "confidence": max(0.0, hypothesis["confidence"] - 0.15), "score": hypothesis["score"] * 0.7, "misses": hypothesis["misses"] + 1})
        existing_points = [branch["point"] for branch in branches]
        for candidate in valid:
            center = self._center(candidate)
            if not any(math.hypot(center[0] - point[0], center[1] - point[1]) <= self.max_step / 2.0 for point in existing_points):
                confidence = float(candidate["confidence"])
                branches.append({"point": center, "velocity": (0.0, 0.0), "confidence": confidence, "score": confidence * 0.5, "misses": 0})
        self._hypotheses = sorted(branches, key=lambda item: item["score"], reverse=True)[: self.max_hypotheses]
        if not self._hypotheses:
            self._terminated = True
            return self._result("terminated", None, 0.0, "track_terminated")
        if (len(self._hypotheses) > 1 and
                self._hypotheses[0]["score"] - self._hypotheses[1]["score"] <= self.ambiguity_margin):
            return self._result("unavailable", None, 0.0, "ambiguous_candidates", hypothesis_count=len(self._hypotheses))
        best = self._hypotheses[0]
        return self._result("predicted" if best["misses"] else "observed", best["point"], best["confidence"], "confidence_decayed" if best["misses"] else None, hypothesis_count=len(self._hypotheses))



def _dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")


def _candidate_status_warning(candidates: Iterable[dict]) -> str:
    """Keep explicit upstream rejection/unavailability visible to the track."""
    statuses = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("status") == "rejected" or candidate.get("accepted") is False:
            statuses.append("candidate_rejected")
        elif candidate.get("status") == "unavailable" or candidate.get("available") is False:
            statuses.append("candidate_unavailable")
    return "candidate_rejected" if "candidate_rejected" in statuses else (
        "candidate_unavailable" if statuses else "no_valid_candidate"
    )


def _valid_candidates(candidates: Iterable[dict], minimum: float):
    """Return finite, usable candidates without raising on bad input."""
    valid = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if (candidate.get("status") in {"rejected", "unavailable"} or
                candidate.get("accepted") is False or candidate.get("available") is False):
            continue
        try:
            confidence = float(candidate.get("confidence", 0.0))
            center = tuple(float(value) for value in candidate["center"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(center) != 2 or not math.isfinite(confidence) or not all(math.isfinite(value) for value in center):
            continue
        if confidence >= minimum:
            valid.append(candidate)
    return valid


def _values(values: Iterable[float], count: int) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be numeric") from exc
    if len(result) != count:
        raise ValueError("unexpected coordinate shape")
    return result


def normalize_point(point: Iterable[float], width: int, height: int) -> Tuple[float, float]:
    """Convert normalized or pixel-space point coordinates to pixels."""
    _dimensions(width, height)
    x, y = _values(point, 2)
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("coordinates must be finite")
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def normalize_box(box: Iterable[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert normalized or pixel-space ``x1,y1,x2,y2`` coordinates to pixels.

    A golf ball cannot plausibly fill a quarter of any frame dimension or 5%
    of the frame area. Boxes that large are background hallucinations (the
    local ball model emits full-frame boxes on some clips), so their
    coordinates are rejected instead of seeding a phantom track.
    """
    _dimensions(width, height)
    x1, y1, x2, y2 = _values(box, 4)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("coordinates must be finite")
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError("box must have positive extent")
    if box_width > width / 4.0 or box_height > height / 4.0:
        raise ValueError("box implausibly large for a golf ball")
    if box_width * box_height > 0.05 * width * height:
        raise ValueError("box implausibly large for a golf ball")
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("box must be within the image frame")
    return x1, y1, x2, y2
