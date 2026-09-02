"""Fail-closed temporal gates for research-only pixel candidates.

This module does not detect golf balls or clubheads. It only filters proposals
from an independent detector and prevents isolated/ambiguous candidates from
becoming rendered tracks.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    frame_index: int
    x: float
    y: float
    radius: float
    residual_motion: float
    appearance_score: float


def _overlaps(candidate: Candidate, box: Tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = box
    nx = min(max(candidate.x, left), right)
    ny = min(max(candidate.y, top), bottom)
    return hypot(candidate.x - nx, candidate.y - ny) <= candidate.radius


def filter_candidate(
    candidate: Candidate,
    *,
    width: int,
    height: int,
    person_boxes: Sequence[Tuple[float, float, float, float]] = (),
) -> Optional[Candidate]:
    """Return a safe candidate, or ``None`` when evidence is insufficient."""
    values = (candidate.x, candidate.y, candidate.radius,
              candidate.residual_motion, candidate.appearance_score)
    if candidate.frame_index < 0 or any(not isfinite(float(v)) for v in values):
        return None
    if width <= 0 or height <= 0:
        return None
    if not (0.0 <= candidate.x < width and 0.0 <= candidate.y < height):
        return None
    if not (2.0 <= candidate.radius <= 30.0):
        return None
    if candidate.residual_motion <= 0.0:
        return None
    if not (0.0 <= candidate.appearance_score <= 1.0):
        return None
    if any(_overlaps(candidate, box) for box in person_boxes):
        return None
    return candidate


def accept_candidate_run(
    candidates: Iterable[Candidate],
    *,
    min_consecutive: int = 2,
    max_step: float = 100.0,
) -> List[Candidate]:
    """Select the longest unambiguous, consecutive, motion-supported run."""
    if min_consecutive < 2 or isinstance(max_step, bool) or not isinstance(max_step, (int, float)):
        raise ValueError("invalid temporal limits")
    if not isfinite(float(max_step)) or max_step <= 0.0:
        raise ValueError("invalid temporal limits")
    by_frame = {}
    for candidate in candidates:
        values = (candidate.frame_index, candidate.x, candidate.y,
                  candidate.radius, candidate.residual_motion,
                  candidate.appearance_score)
        if (not isinstance(candidate.frame_index, int) or candidate.frame_index < 0 or
                any(not isfinite(float(value)) for value in values[1:]) or
                not (2.0 <= candidate.radius <= 30.0) or
                candidate.residual_motion <= 0.0 or
                not (0.0 <= candidate.appearance_score <= 1.0)):
            continue
        if candidate.frame_index in by_frame:
            by_frame[candidate.frame_index] = None
        else:
            by_frame[candidate.frame_index] = candidate
    usable = [c for c in by_frame.values() if c is not None]
    usable.sort(key=lambda c: c.frame_index)
    best: List[Candidate] = []
    run: List[Candidate] = []
    for candidate in usable:
        if run:
            previous = run[-1]
            if (candidate.frame_index != previous.frame_index + 1 or
                    hypot(candidate.x - previous.x, candidate.y - previous.y) > max_step):
                if len(run) > len(best):
                    best = run
                run = []
        run.append(candidate)
    if len(run) > len(best):
        best = run
    return best if len(best) >= min_consecutive else []
