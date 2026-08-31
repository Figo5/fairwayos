"""Bounded, deterministic research helpers for selecting motion windows.

This module does not run a detector (and never runs Hough). Callers provide one
scalar score per decoded frame, then receive a small, auditable set of windows.
JSONL persistence keeps each clip's result independently recoverable.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence, Union


@dataclass(frozen=True)
class WindowCandidate:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float
    mean_score: float

    def as_dict(self) -> dict:
        return asdict(self)


def select_bounded_windows(
    scores: Sequence[float], *, radius: int, max_windows: int,
    min_peak_score: float = 0.0,
) -> tuple[WindowCandidate, ...]:
    """Select at most ``max_windows`` non-overlapping windows around peaks.

    Ranking is peak score, mean score, then earliest peak. Window endpoints are
    clipped to the available score sequence. This is deliberately a proposal
    mechanism: it makes no claim that a peak is a golf event.
    """
    if isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be non-negative")
    if isinstance(max_windows, bool) or max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if not math.isfinite(float(min_peak_score)):
        raise ValueError("min_peak_score must be finite")
    values = tuple(float(score) for score in scores)
    if any(not math.isfinite(score) for score in values):
        raise ValueError("scores must be finite")
    if not values:
        return ()

    candidates = []
    for peak_frame, peak_score in enumerate(values):
        if peak_score < min_peak_score:
            continue
        start = max(0, peak_frame - radius)
        end = min(len(values) - 1, peak_frame + radius)
        # Keep only local maxima; ties resolve to the earliest frame. This
        # prevents a plateau from consuming the window budget.
        left = values[peak_frame - 1] if peak_frame else -math.inf
        right = values[peak_frame + 1] if peak_frame + 1 < len(values) else -math.inf
        if peak_score < left or peak_score < right:
            continue
        if peak_score == left or peak_score == right:
            if peak_frame and peak_score == left:
                continue
        candidates.append(WindowCandidate(
            start, end, peak_frame, peak_score,
            sum(values[start:end + 1]) / (end - start + 1),
        ))

    ranked = sorted(candidates, key=lambda item: (-item.peak_score, -item.mean_score, item.peak_frame))
    selected = []
    for candidate in ranked:
        if any(candidate.start_frame <= current.end_frame and
               current.start_frame <= candidate.end_frame for current in selected):
            continue
        selected.append(candidate)
        if len(selected) == max_windows:
            break
    return tuple(sorted(selected, key=lambda item: item.peak_frame))


def append_clip_window_record(path: Union[str, os.PathLike], record: dict) -> None:
    """Append one complete clip record as a durable JSONL line."""
    if not isinstance(record, dict) or not record.get("clip_id"):
        raise ValueError("record must be a dict with a non-empty clip_id")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
