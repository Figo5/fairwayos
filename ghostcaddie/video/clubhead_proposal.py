"""Research-only pixel-space club/clubhead proposal fusion.

This module never emits a production observation, impact event, calibration,
landing, ShotEvent, analytics, or recommendation. Inputs are proposals from
independent pose, line, contour, motion, and ROI stages; the output is only a
candidate for visual review.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "clubhead-proposal.v1"


@dataclass(frozen=True)
class ClubheadCandidate:
    frame_index: int
    point: Optional[Tuple[float, float]]
    confidence: float
    uncertainty_px: Optional[float]
    state: str
    evidence: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    impact: None = None
    impact_frame: None = None

    @property
    def available(self) -> bool:
        return self.point is not None and self.state != "unavailable"


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an x/y pair")
    return (_finite(value[0], f"{name}.x"), _finite(value[1], f"{name}.y"))


def _inside(point: Tuple[float, float], roi: Tuple[float, float, float, float]) -> bool:
    x, y = point
    rx, ry, rw, rh = roi
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def _overlaps(point: Tuple[float, float], bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    if bbox is None:
        return False
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def _proposal_items(items: Iterable[Mapping[str, Any]], key: str) -> list[tuple[Tuple[float, float], float]]:
    result = []
    for item in items or ():
        if not isinstance(item, Mapping) or key not in item or "score" not in item:
            continue
        point = _point(item[key], key)
        score = max(0.0, min(1.0, _finite(item["score"], f"{key}.score")))
        if score:
            result.append((point, score))
    return result


def _dispersion(points: Sequence[Tuple[Tuple[float, float], float]], center: Tuple[float, float]) -> float:
    total = sum(weight for _, weight in points)
    if not total:
        return 0.0
    return sum(math.hypot(point[0] - center[0], point[1] - center[1]) * weight
               for point, weight in points) / total


def _pose_compatible(point: Tuple[float, float], pose: Optional[Mapping[str, Any]]) -> bool:
    if not pose or "wrist" not in pose or "elbow" not in pose:
        return True
    wrist = _point(pose["wrist"], "pose.wrist")
    elbow = _point(pose["elbow"], "pose.elbow")
    vx, vy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    arm = math.hypot(vx, vy)
    if arm < 1.0:
        return False
    px, py = point[0] - wrist[0], point[1] - wrist[1]
    distance = math.hypot(px, py)
    forward = (px * vx + py * vy) / arm
    perpendicular = abs(vx * py - vy * px) / arm
    return 0.35 * arm <= forward <= 3.5 * arm and perpendicular <= 0.9 * arm and distance >= 0.35 * arm


def build_clubhead_observation(
    *, frame_index: int, image_size: Tuple[int, int],
    roi: Tuple[float, float, float, float],
    pose: Optional[Mapping[str, Any]] = None,
    line_candidates: Iterable[Mapping[str, Any]] = (),
    contour_candidates: Iterable[Mapping[str, Any]] = (),
    motion_candidates: Iterable[Mapping[str, Any]] = (),
    golfer_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> ClubheadCandidate:
    """Fuse bounded proposal families into one visual-review candidate."""
    if int(frame_index) != frame_index or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    roi = tuple(_finite(value, "roi") for value in roi)
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("roi dimensions must be positive")
    families = [
        ("line", _proposal_items(line_candidates, "endpoint")),
        ("contour", _proposal_items(contour_candidates, "center")),
        ("motion", _proposal_items(motion_candidates, "point")),
    ]
    usable: list[tuple[Tuple[float, float], float, str]] = []
    warnings = []
    for family, items in families:
        for point, score in items:
            if not _inside(point, roi) or _overlaps(point, golfer_bbox) or not _pose_compatible(point, pose):
                continue
            if 0 <= point[0] <= width and 0 <= point[1] <= height:
                usable.append((point, score, family))
    if not usable:
        return ClubheadCandidate(int(frame_index), None, 0.0, None, "unavailable",
                                 (), ("roi_or_golfer_exclusion",))
    total = sum(score for _, score, _ in usable)
    center = (sum(point[0] * score for point, score, _ in usable) / total,
              sum(point[1] * score for point, score, _ in usable) / total)
    pose_confidence = 1.0
    evidence = {"roi"}
    if pose:
        pose_confidence = max(0.0, min(1.0, _finite(pose.get("confidence", 0.0), "pose.confidence")))
        if "wrist" in pose or "elbow" in pose:
            evidence.add("pose")
    evidence.update(family for _, _, family in usable)
    family_count = len(evidence - {"roi", "pose"})
    if family_count < 2:
        return ClubheadCandidate(int(frame_index), None, 0.0, None, "unavailable",
                                 tuple(sorted(evidence)), ("insufficient_independent_families",))
    confidence = min(1.0, (sum(score for _, score, _ in usable) / total) *
                     pose_confidence * (0.75 + 0.25 * min(1.0, family_count / 3.0)))
    uncertainty = _dispersion([(point, score) for point, score, _ in usable], center)
    uncertainty += 2.0 / max(confidence, 0.1)
    if uncertainty > 100.0:
        return ClubheadCandidate(int(frame_index), None, 0.0, round(uncertainty, 4), "unavailable",
                                 tuple(sorted(evidence)), ("proposal_disagreement_exceeds_100px",))
    return ClubheadCandidate(int(frame_index), tuple(round(value, 4) for value in center),
                             round(confidence, 6), round(uncertainty, 4), "observed",
                             tuple(sorted(evidence)), tuple(warnings))


def serialize_clubhead_report(report: Mapping[str, Any]) -> str:
    """Serialize a deterministic, explicitly non-production report."""
    def plain(value: Any) -> Any:
        if isinstance(value, ClubheadCandidate):
            return {key: plain(item) for key, item in asdict(value).items()}
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("report contains non-finite value")
        return value
    payload = plain(report)
    if not isinstance(payload, Mapping):
        raise ValueError("report must be a mapping")
    if payload.get("production_eligible") is not False:
        raise ValueError("clubhead proposal report must be research-only")
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("impact", None)
    payload.setdefault("landing", None)
    payload.setdefault("calibration", None)
    payload.setdefault("shot_event", None)
    payload.setdefault("analytics", None)
    payload.setdefault("recommendation", None)
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
