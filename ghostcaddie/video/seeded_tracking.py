"""Research-only seeded pixel tracking with fail-closed termination.

This module never creates ground truth or domain events. It tracks only points
explicitly supplied by a human and records uncertainty/state transitions.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple

@dataclass(frozen=True)
class SeedPoint:
    frame_index: int
    point: Tuple[float, float]
    label: str

@dataclass(frozen=True)
class SeededTrackPoint:
    frame_index: int
    point: Optional[Tuple[float, float]]
    state: str
    confidence: float
    uncertainty: Optional[float]
    warning: Optional[str] = None

class SeededPointTracker:
    """Bounded Lucas-Kanade tracker for one explicitly seeded point."""
    def __init__(self, *, max_prediction_frames: int = 2, max_step: float = 80.0,
                 min_confidence: float = 0.15):
        if max_prediction_frames < 0 or max_step <= 0 or not 0 <= min_confidence <= 1:
            raise ValueError("invalid tracker limits")
        self.max_prediction_frames = max_prediction_frames
        self.max_step = float(max_step)
        self.min_confidence = float(min_confidence)

    def track(self, frames, seed: SeedPoint) -> list[SeededTrackPoint]:
        if not frames or seed.frame_index < 0 or seed.frame_index >= len(frames):
            raise ValueError("seed frame is outside supplied frames")
        first = frames[seed.frame_index]
        if first is None or len(first.shape) < 2:
            raise ValueError("frames must contain image arrays")
        h, w = first.shape[:2]
        x, y = seed.point
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)) or not (0 <= x < w and 0 <= y < h):
            raise ValueError("seed point must be finite and inside the image")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV and NumPy are required for seeded tracking") from exc
        result = [SeededTrackPoint(i, None, "unavailable", 0.0, None, "before_seed") for i in range(seed.frame_index)]
        result.append(SeededTrackPoint(seed.frame_index, (float(x), float(y)), "observed", 1.0, 0.0))
        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY) if len(first.shape) == 3 else first
        prev = np.array([[[x, y]]], dtype=np.float32)
        lost = 0
        for index in range(seed.frame_index + 1, len(frames)):
            frame = frames[index]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            nxt, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, winSize=(21, 21), maxLevel=3)
            valid = nxt is not None and status is not None and int(status[0][0]) == 1
            if valid:
                nx, ny = map(float, nxt[0][0])
                step = math.hypot(nx - float(prev[0][0][0]), ny - float(prev[0][0][1]))
                err = float(error[0][0]) if error is not None else float("inf")
                valid = all(math.isfinite(v) for v in (nx, ny, step, err)) and step <= self.max_step and 0 <= nx < w and 0 <= ny < h
            if valid:
                confidence = max(0.0, min(1.0, 1.0 / (1.0 + err / 10.0)))
                if confidence < self.min_confidence:
                    valid = False
            if valid:
                lost = 0
                point = (nx, ny)
                result.append(SeededTrackPoint(index, point, "observed", confidence, err))
                prev = np.array([[[nx, ny]]], dtype=np.float32)
            else:
                lost += 1
                if lost <= self.max_prediction_frames:
                    result.append(SeededTrackPoint(index, None, "predicted", max(0.0, 1.0 - lost / (self.max_prediction_frames + 1)), None, "flow_unreliable"))
                else:
                    result.append(SeededTrackPoint(index, None, "unavailable", 0.0, None, "track_terminated"))
                    break
            prev_gray = gray
        return result
