"""Fail-closed temporal gates for research-only pixel candidates.

This module does not detect golf balls or clubheads. It only filters proposals
from an independent detector and prevents isolated/ambiguous candidates from
becoming rendered tracks.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Real
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    frame_index: int
    x: float
    y: float
    radius: float
    residual_motion: float
    appearance_score: float


def _normalize_candidate(candidate: Candidate) -> Optional[Candidate]:
    try:
        if type(candidate.frame_index) is not int or candidate.frame_index < 0:
            return None
        values = (candidate.x, candidate.y, candidate.radius,
                  candidate.residual_motion, candidate.appearance_score)
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            return None
        normalized = tuple(float(value) for value in values)
        if any(not isfinite(value) for value in normalized):
            return None
        normalized_candidate = Candidate(candidate.frame_index, *normalized)
        if not (2.0 <= normalized_candidate.radius <= 30.0):
            return None
        if normalized_candidate.residual_motion <= 0.0:
            return None
        if not (0.0 <= normalized_candidate.appearance_score <= 1.0):
            return None
        return normalized_candidate
    except Exception:
        return None


def _overlaps(candidate: Candidate, box: Tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = box
    nx = min(max(candidate.x, left), right)
    ny = min(max(candidate.y, top), bottom)
    return hypot(candidate.x - nx, candidate.y - ny) <= candidate.radius


def _normalize_person_box(
    box: Tuple[float, float, float, float], *, width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    try:
        if len(box) != 4 or any(not isinstance(value, Real) or isinstance(value, bool)
                                for value in box):
            return None
        normalized = tuple(float(value) for value in box)
        left, top, right, bottom = normalized
        if (any(not isfinite(value) for value in normalized) or
                not (0.0 <= left <= right <= float(width) and
                     0.0 <= top <= bottom <= float(height))):
            return None
        return (left, top, right, bottom)
    except Exception:
        return None


def filter_candidate(
    candidate: Candidate,
    *,
    width: int,
    height: int,
    person_boxes: Sequence[Tuple[float, float, float, float]] = (),
) -> Optional[Candidate]:
    """Return a normalized safe candidate, or ``None`` when evidence is insufficient."""
    try:
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            return None
        safe = _normalize_candidate(candidate)
        if safe is None:
            return None
        if not (safe.radius <= safe.x < width - safe.radius and
                safe.radius <= safe.y < height - safe.radius):
            return None
        boxes = tuple(person_boxes)
        safe_boxes = tuple(
            _normalize_person_box(box, width=width, height=height) for box in boxes
        )
        if any(box is None for box in safe_boxes):
            return None
        if any(_overlaps(safe, box) for box in safe_boxes if box is not None):
            return None
        return safe
    except Exception:
        return None


def _displacement(left: Candidate, right: Candidate) -> Optional[float]:
    try:
        value = hypot(right.x - left.x, right.y - left.y)
        return value if isfinite(value) else None
    except Exception:
        return None


def accept_candidate_run(
    candidates: Iterable[Candidate],
    *,
    min_consecutive: int = 2,
    max_step: float = 100.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[Candidate]:
    """Select the longest unambiguous, consecutive, motion-supported run."""
    if type(min_consecutive) is not int or min_consecutive < 2:
        raise ValueError("invalid temporal limits")
    if not isinstance(max_step, Real) or isinstance(max_step, bool):
        raise ValueError("invalid temporal limits")
    try:
        step_value = float(max_step)
    except Exception:
        raise ValueError("invalid temporal limits")
    if not isfinite(step_value) or step_value <= 0.0:
        raise ValueError("invalid temporal limits")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("invalid frame dimensions")

    by_frame = {}
    try:
        for candidate in candidates:
            frame_index = getattr(candidate, "frame_index", None)
            if type(frame_index) is not int or frame_index < 0:
                continue
            if frame_index in by_frame:
                by_frame[frame_index] = None
                continue
            safe = filter_candidate(candidate, width=width, height=height)
            if safe is None:
                by_frame[frame_index] = None
                continue
            by_frame[frame_index] = safe
    except Exception:
        return []

    usable = [candidate for candidate in by_frame.values() if candidate is not None]
    usable.sort(key=lambda candidate: candidate.frame_index)
    best: List[Candidate] = []
    run: List[Candidate] = []
    for candidate in usable:
        if run:
            previous = run[-1]
            displacement = _displacement(previous, candidate)
            if (displacement is None or
                    candidate.frame_index != previous.frame_index + 1 or
                    displacement <= 0.0 or displacement > step_value):
                if len(run) > len(best):
                    best = run
                run = []
        run.append(candidate)
    if len(run) > len(best):
        best = run
    return best if len(best) >= min_consecutive else []
