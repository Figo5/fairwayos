"""Body-relative features for v3 candidates, from the cached pose pass.

The v2 pass implemented body rejection, unit-tested it, and then never ran a pose
model on real footage, so every ``body_lock_rate`` it reported is zero by absence
of data rather than by measurement. This module makes body real, and uses it the
way the v3 brief requires: **as features, never as an exclusion region.**

That distinction is the whole point. At address and through impact the ball is
legitimately close to the golfer's hands and feet -- a tracker that rejects
everything near the body deletes the correct answer on exactly the frames that
matter most. What separates a ball from a shirt highlight is not proximity, it is
what happens over time: a shirt highlight stays welded to the torso, and a struck
ball leaves. So proximity is handed to the model as a number, and the temporal
question is answered by the tracklet stage.

Poses come from the cache written by ``tools/ball_v3_body_cache.py``. They are
whole-frame single-pose MoveNet and are **not** an identity claim about who the
golfer is; nothing here calls them "the golfer".
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

#: MoveNet keypoint score above which a point is worth using. Matches the
#: canonical worker's threshold so the two passes agree about what "visible" is.
SCORE_THRESHOLD = 0.30
TORSO = ("sh_l", "sh_r", "hip_l", "hip_r")
HANDS = ("wr_l", "wr_r")
FEET = ("ank_l", "ank_r")
#: Features returned per candidate, in order. Named so an ablation can drop them
#: by name rather than by index.
FEATURE_NAMES = (
    "body_available",
    "inside_torso",
    "inside_body_box",
    "distance_to_hands_over_radius",
    "distance_to_feet_over_radius",
    "distance_to_body_centre_over_body_height",
    "body_confidence",
)


class BodyCache:
    """Per-clip pose lookup. Absent pose is a state, not an error."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._clips: dict[str, Optional[dict]] = {}

    def frames_for(self, clip_id: str) -> Optional[dict]:
        if clip_id not in self._clips:
            path = self.directory / f"{clip_id}.json"
            self._clips[clip_id] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
        return self._clips[clip_id]

    def pose(self, clip_id: str, frame_index: int) -> Optional[dict]:
        payload = self.frames_for(clip_id)
        if not payload:
            return None
        return payload.get("frames", {}).get(str(frame_index))


def visible_points(pose: Optional[Mapping]) -> dict[str, tuple[float, float]]:
    if not pose:
        return {}
    points = {}
    for name, value in (pose.get("keypoints") or {}).items():
        x, y, score = value
        if score >= SCORE_THRESHOLD:
            points[name] = (float(x), float(y))
    return points


def _polygon_contains(polygon: Sequence[tuple[float, float]],
                      x: float, y: float) -> bool:
    """Ray casting. Four points, so a library dependency would be silly."""
    inside = False
    count = len(polygon)
    for index in range(count):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % count]
        if (y0 > y) != (y1 > y):
            crossing = x0 + (y - y0) * (x1 - x0) / (y1 - y0 if y1 != y0 else 1e-9)
            if crossing > x:
                inside = not inside
    return inside


def torso_polygon(points: Mapping[str, tuple[float, float]]
                  ) -> Optional[list[tuple[float, float]]]:
    """Shoulders and hips as a quad, ordered so the polygon does not self-cross."""
    if not all(name in points for name in TORSO):
        return None
    return [points["sh_l"], points["sh_r"], points["hip_r"], points["hip_l"]]


def body_box(points: Mapping[str, tuple[float, float]]
             ) -> Optional[tuple[float, float, float, float]]:
    if len(points) < 4:
        return None
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    return min(xs), min(ys), max(xs), max(ys)


def features(pose: Optional[Mapping], x: float, y: float, radius: float
             ) -> list[float]:
    """Body-relative description of one candidate. Always ``len(FEATURE_NAMES)``.

    Every feature is zero when there is no usable pose, and ``body_available``
    says so explicitly, so the model can tell "not near the body" apart from
    "nobody looked".
    """
    points = visible_points(pose)
    if len(points) < 4:
        return [0.0] * len(FEATURE_NAMES)

    radius = max(1.0, float(radius))
    polygon = torso_polygon(points)
    inside_torso = 1.0 if (polygon and _polygon_contains(polygon, x, y)) else 0.0

    box = body_box(points)
    inside_box = 0.0
    centre_distance = 0.0
    if box:
        x0, y0, x1, y1 = box
        inside_box = 1.0 if (x0 <= x <= x1 and y0 <= y <= y1) else 0.0
        height = max(1.0, y1 - y0)
        centre_distance = math.hypot(x - (x0 + x1) / 2.0,
                                     y - (y0 + y1) / 2.0) / height

    def nearest(names: Sequence[str]) -> float:
        available = [points[n] for n in names if n in points]
        if not available:
            return 0.0
        return min(math.hypot(x - px, y - py) for px, py in available) / radius

    return [
        1.0,
        inside_torso,
        inside_box,
        # Capped: "400 radii away" and "4000 radii away" are the same fact, and an
        # uncapped ratio would dominate the input scale on a 4K wide shot.
        min(nearest(HANDS), 200.0) / 200.0,
        min(nearest(FEET), 200.0) / 200.0,
        min(centre_distance, 10.0) / 10.0,
        len(points) / 17.0,
    ]


def torso_centre(pose: Optional[Mapping]) -> Optional[tuple[float, float]]:
    """Where the torso is, for measuring whether a candidate moves *with* it."""
    points = visible_points(pose)
    available = [points[name] for name in TORSO if name in points]
    if len(available) < 2:
        return None
    return (sum(p[0] for p in available) / len(available),
            sum(p[1] for p in available) / len(available))
