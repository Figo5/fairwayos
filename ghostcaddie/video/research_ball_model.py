"""Coordinate normalization for local research model outputs."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


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
            "warning": warning,
        }

    def update(self, candidates: Iterable[dict]):
        if self._terminated:
            return self._result("terminated", None, 0.0, "track_terminated")
        valid = _valid_candidates(candidates, self.min_confidence)
        if self._point is None:
            if not valid:
                return self._result("unavailable", None, 0.0, "no_valid_candidate")
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
    ) -> None:
        if not 0.0 <= min_confidence <= reacquire_confidence <= 1.0:
            raise ValueError("invalid confidence thresholds")
        if not math.isfinite(float(max_step)) or max_step <= 0 or max_misses < 1 or max_hypotheses < 1:
            raise ValueError("invalid track limits")
        self.min_confidence = float(min_confidence)
        self.reacquire_confidence = float(reacquire_confidence)
        self.max_step = float(max_step)
        self.max_misses = int(max_misses)
        self.max_hypotheses = int(max_hypotheses)
        self._hypotheses = []
        self._terminated = False
        self._pending_reacquisition = None

    @staticmethod
    def _result(state, point, confidence, warning=None, hypothesis_count=0):
        return {
            "state": state,
            "point": None if point is None else {"x": float(point[0]), "y": float(point[1])},
            "confidence": float(confidence),
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
        valid = _valid_candidates(candidates, self.min_confidence)
        if self._terminated:
            return self._reacquire(valid)
        if not self._hypotheses:
            if not valid:
                return self._result("unavailable", None, 0.0, "no_valid_candidate")
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
        best = self._hypotheses[0]
        return self._result("predicted" if best["misses"] else "observed", best["point"], best["confidence"], "confidence_decayed" if best["misses"] else None, hypothesis_count=len(self._hypotheses))



def _dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")


def _valid_candidates(candidates: Iterable[dict], minimum: float):
    """Return finite, two-dimensional candidates without raising on bad input."""
    valid = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
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
    return x1, y1, x2, y2
