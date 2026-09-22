"""Shared helpers for the demo-mode tests.

Synthetic fixtures only, always in a temporary directory, and always generated
from code: no source footage, no model output and no research media is used here.
Every helper skips cleanly when the interpreter cannot decode or encode video, so
the demo contract tests still run on a bare interpreter.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from ghostcaddie.demo import DEMO_CONFIG_SCHEMA
from ghostcaddie.demo.probe import SourceProbe

try:  # pragma: no cover - import guard
    import cv2  # noqa: F401
    import numpy as np

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    CV2_AVAILABLE = False
    np = None


def synthetic_probe(width: int = 640, height: int = 480, fps: float = 25.0,
                    duration: float = 8.0, frame_count: Optional[int] = 200,
                    *, variable_cadence: bool = False,
                    pixel_aspect_ratio: float = 1.0) -> SourceProbe:
    """A SourceProbe for a clip that does not need to exist to be described."""
    frames = frame_count if frame_count is not None else int(duration * fps)
    return SourceProbe(
        path="/tmp/synthetic.mp4", display_name="synthetic.mp4", width=width,
        height=height, nominal_fps=fps, average_fps=fps, duration_seconds=duration,
        frame_count=frames, pixel_aspect_ratio=pixel_aspect_ratio,
        container_format="mp4", codec="h264", variable_cadence=variable_cadence,
        cadence_note=(f"container cadence {fps:g} fps; playback rate, not measured "
                      f"action time"))


def base_config(**overrides) -> dict:
    """A minimal valid config payload; override any top-level key."""
    payload = {
        "schema_version": DEMO_CONFIG_SCHEMA,
        "interval": {"start_seconds": 0.0, "end_seconds": 4.0, "max_frames": 200},
        "roi": [200, 80, 240, 320],
        "seeds": {"ball": {"frame": 10, "x": 320.0, "y": 300.0, "note": "test"}},
        "keyframes": {"address": 20, "top": 60, "impact": 90, "finish": 96},
        "targets": {"ball": {"mode": "template", "template_size": 21,
                             "search_radius": 30, "min_score": 0.4,
                             "max_jump_px": 40.0, "loss_frames": 3}},
        "corrections": {},
        "interpolation": {"enabled": True, "max_gap_frames": 2},
    }
    payload.update(overrides)
    return payload


def write_config(directory: Path, payload: dict) -> Path:
    path = Path(directory) / "demo_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def make_synthetic_clip(directory: Path, *, width: int = 320, height: int = 240,
                        fps: float = 25.0, frames: int = 60,
                        ball_motion: str = "slide",
                        club_visible: bool = True,
                        background: int = 35) -> Path:
    """A tiny clip with a bright 'ball' and a dark 'club' to track.

    ``ball_motion``: ``slide`` moves it one pixel per frame (comfortably trackable),
    ``late_jump`` keeps it still and then throws it away (a tracker must lose it),
    ``none`` leaves it static.
    """
    import cv2
    import numpy as np

    path = Path(directory) / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), background, np.uint8)
        cv2.rectangle(frame, (0, int(height * 0.6)), (width, height), (90, 110, 80), -1)
        if ball_motion == "slide":
            bx, by = 60 + index, 200
        elif ball_motion == "late_jump":
            bx, by = 60, 200
            if index >= frames - 6:
                bx, by = 60 + (index - (frames - 6)) * 12, 190
        else:
            bx, by = 60, 200
        cv2.circle(frame, (int(bx), int(by)), 5, (250, 250, 250), -1)
        if club_visible:
            cx, cy = 240 - index * 2, 80 + index
            cv2.circle(frame, (int(cx), int(cy)), 10, (10, 10, 10), -1)
        writer.write(frame)
    writer.release()
    return path


def make_identity_clip(directory: Path, *, width: int = 192, height: int = 144,
                       fps: float = 20.0, frames: int = 12) -> Path:
    """A clip with no golfer in it at all, for 'body must not be invented' tests."""
    import cv2
    import numpy as np

    path = Path(directory) / "empty.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), 40 + index, np.uint8)
        writer.write(frame)
    writer.release()
    return path


def body_record(frame: int, *, points: dict, available: bool = True,
                reason: str = "") -> dict:
    """One worker-shaped body record with named visible keypoints."""
    names = ("nose", "eye_l", "eye_r", "ear_l", "ear_r", "sh_l", "sh_r", "el_l",
             "el_r", "wr_l", "wr_r", "hip_l", "hip_r", "kn_l", "kn_r", "ank_l", "ank_r")
    keypoints = []
    for name in names:
        xy = points.get(name)
        keypoints.append({"name": name, "x": (xy or (0, 0))[0], "y": (xy or (0, 0))[1],
                          "score": 0.9 if xy else 0.05, "visible": xy is not None})
    return {"source_frame": frame, "available": available, "reason": reason,
            "visible_keypoint_count": sum(1 for k in keypoints if k["visible"]),
            "keypoints": keypoints}


UPRIGHT_POINTS = {
    "nose": (110, 60), "sh_l": (95, 90), "sh_r": (125, 90),
    "el_l": (85, 120), "el_r": (135, 120), "wr_l": (90, 145), "wr_r": (130, 145),
    "hip_l": (100, 150), "hip_r": (120, 150), "kn_l": (100, 185), "kn_r": (120, 185),
    "ank_l": (100, 220), "ank_r": (120, 220),
}


def shift_points(points: dict, dx: float, dy: float = 0.0) -> dict:
    return {name: (x + dx, y + dy) for name, (x, y) in points.items()}


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="fairwayos-demo-test-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()


def require_cv2(case: unittest.TestCase) -> None:
    if not CV2_AVAILABLE:
        case.skipTest("this interpreter cannot decode/encode video (no cv2)")


def workdir_sizes(directory: Path) -> dict:
    """{relative path: bytes} for everything under a directory."""
    return {str(p.relative_to(directory)): p.stat().st_size
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}
