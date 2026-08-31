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
        if max_step <= 0 or max_misses < 1 or not 0.0 <= confidence_decay <= 1.0:
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
        valid = [candidate for candidate in candidates if float(candidate.get("confidence", 0.0)) >= self.min_confidence]
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

def _dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")


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
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def normalize_box(box: Iterable[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert normalized or pixel-space ``x1,y1,x2,y2`` coordinates to pixels."""
    _dimensions(width, height)
    x1, y1, x2, y2 = _values(box, 4)
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return x1 * width, y1 * height, x2 * width, y2 * height
    return x1, y1, x2, y2
