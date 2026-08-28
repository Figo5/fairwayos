"""Optional local YOLO pose adapter for generic pixel-space evidence.

The core package stays free of CV dependencies.  This module is imported only
when explicitly selected through GHOSTCADDIE_AUTO_DETECTOR.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from .observations import VideoObservations

try:  # Optional dependency, available only in the isolated AI environment.
    import cv2
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised by the unavailable path.
    cv2 = None
    YOLO = None


_SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
             (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16))
_FRAME_RE = re.compile(r"frame_(\d+)")


class YoloPoseDetector:
    """Convert highest-confidence generic person/pose results to pixel contract."""

    def __init__(self, model_name: str | None = None, sample_fps: float | None = None):
        if cv2 is None or YOLO is None:
            raise RuntimeError("YOLO pose runtime is unavailable")
        self.model_name = model_name or os.environ.get("GHOSTCADDIE_AUTO_MODEL", "yolo11n-pose.pt")
        self.sample_fps = float(sample_fps or os.environ.get("GHOSTCADDIE_AUTO_FPS", "2"))
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        self.model = YOLO(self.model_name)

    def detect(self, frame_paths: Iterable[str]) -> VideoObservations:
        paths = [Path(path) for path in frame_paths]
        if not paths:
            raise RuntimeError("no frames supplied to YOLO pose detector")
        first = cv2.imread(str(paths[0]))
        if first is None:
            raise RuntimeError("unable to decode detector frame")
        height, width = first.shape[:2]
        records = []
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                continue
            result = self.model(frame, verbose=False)[0]
            person = self._person(result)
            if person is None:
                continue
            x1, y1, x2, y2, confidence, index = person
            keypoints = self._keypoints(result, index)
            ankles = [keypoints[i] for i in (15, 16) if i in keypoints]
            anchor = (sum(p[0] for p in ankles) / len(ankles),
                      sum(p[1] for p in ankles) / len(ankles)) if ankles else ((x1 + x2) / 2, y2)
            match = _FRAME_RE.search(path.stem)
            sampled_index = int(match.group(1)) - 1 if match else len(records)
            warnings = ["ball_missing"]
            if confidence < 0.5:
                warnings.append("low_confidence")
            records.append({
                "frame_index": sampled_index,
                "timestamp_seconds": round(sampled_index / self.sample_fps, 6),
                "golfer": {"bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                           "anchor": {"x": anchor[0], "y": anchor[1]}, "confidence": confidence},
                "club": None, "clubhead": None, "ball": None, "phase": "unknown",
                "contact": None, "intended_direction": None, "landing": None,
                "warnings": sorted(set(warnings)),
            })
        if not records:
            raise RuntimeError("YOLO pose produced no person observations")
        records.sort(key=lambda item: (item["frame_index"], item["timestamp_seconds"]))
        return VideoObservations.from_dict({
            "schema_version": "video-observations.v1",
            "image": {"width": width, "height": height},
            "observations": records,
        })

    @staticmethod
    def _person(result: Any):
        if result.boxes is None or len(result.boxes) == 0:
            return None
        best = None
        for index, cls in enumerate(result.boxes.cls.tolist()):
            if int(cls) != 0:
                continue
            confidence = float(result.boxes.conf[index])
            coords = [float(value) for value in result.boxes.xyxy[index].tolist()]
            candidate = (*coords, confidence, index)
            if best is None or confidence > best[4]:
                best = candidate
        return best

    @staticmethod
    def _keypoints(result: Any, index: int):
        if result.keypoints is None or result.keypoints.conf is None:
            return {}
        xy = result.keypoints.xy[index].cpu().numpy()
        confidence = result.keypoints.conf[index].cpu().numpy()
        return {i: (float(point[0]), float(point[1])) for i, point in enumerate(xy) if float(confidence[i]) >= 0.25}


def create_detector():
    """Factory used by the explicit GHOSTCADDIE_AUTO_DETECTOR loader."""
    return YoloPoseDetector()
