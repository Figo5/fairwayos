"""Versioned camera-pixel observation contracts for video perception."""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .errors import VideoContractError

OBSERVATIONS_SCHEMA_VERSION = "video-observations.v1"
# Canonical temporal phases for one fixed-camera golf shot.  These names are
# serialized; aliases are accepted only when their meaning is unambiguous.
CANONICAL_PHASES = frozenset({
    "unknown", "address", "backswing", "top", "downswing", "contact",
    "follow_through", "ball_flight", "landing", "rolling", "finish",
})
_PHASE_ALIASES = {
    "setup": "address",
    "setup/address": "address",
    "impact": "contact",
    "follow-through": "follow_through",
    "follow through": "follow_through",
    "flight": "ball_flight",
    "ball flight": "ball_flight",
}
_VALID_WARNINGS = {"occlusion", "blur", "lighting", "camera_motion", "ball_missing", "anchor_missing", "low_confidence"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VideoContractError(name + " must be a finite number")
    return float(value)


def _object(value: Any, name: str, keys) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) - set(keys):
        raise VideoContractError(name + " has malformed or unknown fields")
    return value


def _point(value: Any, name: str, bounds=None, method=False, confidence=True) -> Dict[str, Any]:
    keys = {"x", "y"} | ({"confidence"} if confidence else set()) | ({"method"} if method else set())
    item = _object(value, name, keys)
    if set(item) != keys:
        raise VideoContractError(name + " must contain explicit coordinates" + (" and confidence" if confidence else ""))
    x, y = _number(item["x"], name + ".x"), _number(item["y"], name + ".y")
    result = {"x": item["x"], "y": item["y"]}
    if confidence:
        confidence_value = _number(item["confidence"], name + ".confidence")
        if not 0 <= confidence_value <= 1:
            raise VideoContractError(name + ".confidence must be between 0 and 1")
        result["confidence"] = item["confidence"]
    if bounds is not None and not (0 <= x <= bounds[0] and 0 <= y <= bounds[1]):
        raise VideoContractError(name + " is outside image bounds")
    if method and (not isinstance(item["method"], str) or not item["method"].strip()):
        raise VideoContractError(name + ".method must be a non-empty string")
    if method: result["method"] = item["method"]
    return result


@dataclass(frozen=True)
class PixelBBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_dict(cls, value, bounds):
        item = _object(value, "golfer.bbox", {"x", "y", "width", "height"})
        if set(item) != {"x", "y", "width", "height"}:
            raise VideoContractError("golfer.bbox requires x, y, width, and height")
        x, y = _number(item["x"], "golfer.bbox.x"), _number(item["y"], "golfer.bbox.y")
        width, height = _number(item["width"], "golfer.bbox.width"), _number(item["height"], "golfer.bbox.height")
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > bounds[0] or y + height > bounds[1]:
            raise VideoContractError("golfer.bbox is outside image bounds")
        return cls(item["x"], item["y"], item["width"], item["height"])

    def to_dict(self):
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class GolferObservation:
    bbox: PixelBBox
    anchor: Optional[Dict[str, Any]]
    confidence: float

    @classmethod
    def from_dict(cls, value, bounds):
        item = _object(value, "golfer", {"bbox", "anchor", "confidence"})
        if set(item) != {"bbox", "anchor", "confidence"}:
            raise VideoContractError("golfer requires bbox, anchor, and confidence")
        confidence = _number(item["confidence"], "golfer.confidence")
        if not 0 <= confidence <= 1:
            raise VideoContractError("golfer.confidence must be between 0 and 1")
        anchor = None if item["anchor"] is None else _point(item["anchor"], "golfer.anchor", bounds, confidence=False)
        return cls(PixelBBox.from_dict(item["bbox"], bounds), anchor, confidence)

    def to_dict(self):
        return {"bbox": self.bbox.to_dict(), "anchor": None if self.anchor is None else dict(self.anchor), "confidence": self.confidence}


@dataclass(frozen=True)
class PixelObservation:
    frame_index: int
    timestamp_seconds: float
    golfer: GolferObservation
    club: Optional[Dict[str, Any]]
    clubhead: Optional[Dict[str, Any]]
    ball: Optional[Dict[str, Any]]
    phase: str
    contact: Optional[Dict[str, Any]]
    intended_direction: Optional[Dict[str, Any]]
    landing: Optional[Dict[str, Any]]
    warnings: List[str]

    @classmethod
    def from_dict(cls, value, bounds):
        keys = {"frame_index", "timestamp_seconds", "golfer", "club", "clubhead", "ball", "phase", "contact", "intended_direction", "landing", "warnings"}
        item = _object(value, "observation", keys)
        if set(item) != keys:
            raise VideoContractError("observation must include every field, using null for unknowns")
        frame = item["frame_index"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise VideoContractError("frame_index must be a non-negative integer")
        timestamp = _number(item["timestamp_seconds"], "timestamp_seconds")
        if timestamp < 0:
            raise VideoContractError("timestamp_seconds must be non-negative")
        phase = item["phase"]
        if not isinstance(phase, str):
            raise VideoContractError("phase must be a string")
        phase = _PHASE_ALIASES.get(phase.strip().lower(), phase.strip().lower())
        if phase not in CANONICAL_PHASES:
            raise VideoContractError("invalid or ambiguous phase")
        warnings = item["warnings"]
        if not isinstance(warnings, list) or any(w not in _VALID_WARNINGS for w in warnings):
            raise VideoContractError("warnings must use the defined warning codes")
        if (item["golfer"].get("anchor") is None) != ("anchor_missing" in warnings):
            raise VideoContractError("missing golfer anchor requires exactly anchor_missing warning")
        if item["ball"] is None and "ball_missing" not in warnings:
            raise VideoContractError("missing ball requires ball_missing warning")
        if item["ball"] is not None and "ball_missing" in warnings:
            raise VideoContractError("ball_missing warning requires an unknown ball")
        club = item["club"]
        if club is not None:
            club = _object(club, "club", {"name", "confidence"})
            if set(club) != {"name", "confidence"} or not isinstance(club["name"], str) or not club["name"].strip():
                raise VideoContractError("club requires name and confidence")
            c = _number(club["confidence"], "club.confidence")
            if not 0 <= c <= 1: raise VideoContractError("club.confidence must be between 0 and 1")
        def pixel(name, method=False, bounded=True):
            return None if item[name] is None else _point(item[name], name, bounds if bounded else None, method)
        parsed = cls(frame, timestamp, GolferObservation.from_dict(item["golfer"], bounds), club,
                     pixel("clubhead"), pixel("ball"), phase, pixel("contact", True),
                     pixel("intended_direction", bounded=False), pixel("landing", True), list(warnings))
        quantities = [parsed.golfer.confidence]
        quantities.extend(v["confidence"] for v in (parsed.club, parsed.clubhead, parsed.ball, parsed.contact, parsed.intended_direction, parsed.landing) if v is not None and "confidence" in v)
        low = any(v < 0.5 for v in quantities)
        if low != ("low_confidence" in warnings):
            raise VideoContractError("low-confidence quantities require exactly low_confidence warning")
        return parsed

    def to_dict(self):
        return {"frame_index": self.frame_index, "timestamp_seconds": self.timestamp_seconds,
                "golfer": self.golfer.to_dict(), "club": self.club, "clubhead": self.clubhead,
                "ball": self.ball, "phase": self.phase, "contact": self.contact,
                "intended_direction": self.intended_direction, "landing": self.landing,
                "warnings": list(self.warnings)}


@dataclass(frozen=True)
class VideoObservations:
    image_width: int
    image_height: int
    items: List[PixelObservation]
    schema_version: str = OBSERVATIONS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "image", "observations"}:
            raise VideoContractError("observations payload has invalid top-level fields")
        if payload["schema_version"] != OBSERVATIONS_SCHEMA_VERSION:
            raise VideoContractError("unsupported observations schema version")
        image = _object(payload["image"], "image", {"width", "height"})
        width, height = image.get("width"), image.get("height")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise VideoContractError("image dimensions must be positive integers")
        raw_items = payload["observations"]
        if not isinstance(raw_items, list) or not raw_items:
            raise VideoContractError("observations must be a non-empty list for one shot")
        items = [PixelObservation.from_dict(item, (width, height)) for item in raw_items]
        pairs = [(x.frame_index, x.timestamp_seconds) for x in items]
        if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(pairs, pairs[1:])):
            raise VideoContractError("frames and timestamps must be unique and strictly increasing")
        return cls(width, height, items)

    def to_dict(self):
        return {"schema_version": self.schema_version, "image": {"width": self.image_width, "height": self.image_height}, "observations": [x.to_dict() for x in self.items]}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def load_fixture_observations(resource, project_boundary):
    """Compatibility entry point for the deterministic fixture loader."""
    from .perception import load_fixture_observations as loader
    return loader(resource, project_boundary)
