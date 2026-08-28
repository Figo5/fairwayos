"""Strict, versioned human-in-the-loop video annotation contract.

This module is deliberately independent of analytics and the existing fixture
video contracts.  Documents are plain JSON-compatible mappings and never carry
source paths.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import VideoContractError

SCHEMA_VERSION = "video-human-annotations.v1"
SOURCE_VALUES = frozenset({"user_supplied", "user_confirmed", "observed", "inferred", "unavailable"})
PHASES = frozenset({"address", "backswing", "top", "downswing", "contact", "follow_through", "landing"})
TOP_LEVEL = frozenset({"schema_version", "status", "explicit_submit", "video", "calibration_points", "engine_points", "golfer_anchor", "ball", "clubhead", "contact", "target_intended_direction", "landing", "club_selection", "context", "warnings"})
VIDEO_FIELDS = frozenset({"width", "height", "frame_count", "duration_seconds"})
POINT_FIELDS = frozenset({"x", "y", "frame_index", "timestamp_seconds", "confidence", "phase"})
CALIBRATION_FIELDS = POINT_FIELDS | {"source"}
ENGINE_POINT_FIELDS = frozenset({"x", "y"})
AMBIGUOUS = re.compile(r"\b(?:ambiguous|unclear|unknown|maybe|possibly|unsure|unverified)\b", re.I)


def _error(message: str) -> None:
    raise VideoContractError(message)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _error(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _error(f"{name} must be an integer")
    return value


def _check_path_strings(value: Any, path: str = "document") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _error(f"non-string field at {path}")
            if "path" in key.lower() or key.lower() in {"file", "filename", "source_identifier"}:
                _error(f"unsafe path field at {path}.{key}")
            _check_path_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_path_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if os.path.isabs(value) or value.startswith(("~/", "~\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            _error(f"absolute path at {path}")


def _point(value: Any, name: str, width: int, height: int, frame_count: int | None, duration: float, *, phase_required=False, allowed_fields=POINT_FIELDS) -> None:
    if not isinstance(value, Mapping):
        _error(f"{name} must be an object")
    extra = set(value) - allowed_fields
    if extra:
        _error(f"fabricated field in {name}: {sorted(extra)[0]}")
    for axis, bound in (("x", width), ("y", height)):
        if axis not in value:
            _error(f"missing required field {name}.{axis}")
        number = _number(value[axis], f"{name}.{axis}")
        if not 0 <= number <= bound:
            _error(f"{name}.{axis} out of bounds")
    if "frame_index" in value:
        frame = _integer(value["frame_index"], f"{name}.frame_index")
        if frame < 0 or (frame_count is not None and frame >= frame_count):
            _error(f"{name}.frame_index out of bounds")
    if "timestamp_seconds" in value:
        timestamp = _number(value["timestamp_seconds"], f"{name}.timestamp_seconds")
        if timestamp < 0 or timestamp > duration:
            _error(f"{name}.timestamp_seconds out of bounds")
    if "frame_index" in value and "timestamp_seconds" in value and frame_count and duration:
        expected = value["frame_index"] * duration / max(frame_count - 1, 1)
        if abs(value["timestamp_seconds"] - expected) > max(0.1, 1.0 / 30):
            _error(f"{name} frame/timestamp inconsistent")
    if "confidence" in value and not 0 <= _number(value["confidence"], f"{name}.confidence") <= 1:
        _error(f"{name}.confidence out of bounds")
    if phase_required or "phase" in value:
        phase = value.get("phase")
        if phase not in PHASES or AMBIGUOUS.search(phase):
            _error(f"{name}.phase is ambiguous or invalid")


def _event(value: Any, name: str, frame_count: int, duration: float) -> None:
    if not isinstance(value, Mapping) or set(value) - POINT_FIELDS:
        _error(f"fabricated field in {name}")
    if "frame_index" not in value or "timestamp_seconds" not in value or "confidence" not in value or "phase" not in value:
        _error(f"{name} requires frame_index, timestamp_seconds, confidence, and phase")
    frame = _integer(value["frame_index"], f"{name}.frame_index")
    timestamp = _number(value["timestamp_seconds"], f"{name}.timestamp_seconds")
    if not 0 <= frame < frame_count or not 0 <= timestamp <= duration:
        _error(f"{name} frame/timestamp out of bounds")
    expected = frame * duration / max(frame_count - 1, 1)
    if abs(timestamp - expected) > max(0.1, 1.0 / 30):
        _error(f"{name} frame/timestamp inconsistent")
    if not 0 <= _number(value["confidence"], f"{name}.confidence") <= 1:
        _error(f"{name}.confidence out of bounds")
    if value["phase"] not in PHASES or AMBIGUOUS.search(value["phase"]):
        _error(f"{name}.phase is ambiguous or invalid")


def _annotation(value: Any, name: str, width: int, height: int, frame_count: int | None, duration: float, *, point=True, phase_required=False) -> None:
    if not isinstance(value, Mapping) or set(value) != {"value", "source"}:
        _error(f"{name} must contain exactly value and source")
    source = value["source"]
    if source not in SOURCE_VALUES:
        _error(f"invalid source label for {name}")
    if source == "unavailable":
        if value["value"] is not None:
            _error(f"{name} unavailable value must be null")
        return
    if value["value"] is None:
        _error(f"{name} requires a value unless unavailable")
    if point:
        _point(value["value"], name, width, height, frame_count, duration, phase_required=phase_required)
    elif not isinstance(value["value"], (str, Mapping)):
        _error(f"{name}.value must be a string or object")


def _validated(payload: Mapping[str, Any], strict: bool = True) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _error("document must be an object")
    if strict and set(payload) != TOP_LEVEL:
        missing, extra = TOP_LEVEL - set(payload), set(payload) - TOP_LEVEL
        if missing:
            _error(f"missing required field: {sorted(missing)[0]}")
        _error(f"fabricated field: {sorted(extra)[0]}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        _error("invalid schema_version")
    if payload.get("status") not in {"draft", "submitted"}:
        _error("status must be draft or submitted")
    if not isinstance(payload.get("explicit_submit"), bool):
        _error("explicit_submit must be boolean")
    if payload["status"] == "submitted" and not payload["explicit_submit"]:
        _error("submitted_without_explicit_submit")
    if payload["status"] == "draft" and payload["explicit_submit"]:
        _error("draft cannot be explicitly submitted")
    video = payload.get("video")
    if not isinstance(video, Mapping) or set(video) != VIDEO_FIELDS:
        _error("video must contain exactly width, height, frame_count, duration_seconds")
    width, height = _integer(video["width"], "video.width"), _integer(video["height"], "video.height")
    frame_count, duration = _integer(video["frame_count"], "video.frame_count"), _number(video["duration_seconds"], "video.duration_seconds")
    if width <= 0 or height <= 0 or frame_count <= 0 or duration < 0:
        _error("video dimensions, frame_count, and duration must be positive")
    points = payload["calibration_points"]
    engine_points = payload["engine_points"]
    if not isinstance(points, list) or len(points) != 4:
        _error("calibration_points must contain exactly four points")
    if not isinstance(engine_points, list) or len(engine_points) != 4:
        _error("engine_points must contain exactly four points")
    for i, item in enumerate(points):
        if not isinstance(item, Mapping) or item.get("source") not in SOURCE_VALUES or item.get("source") == "unavailable":
            _error(f"calibration_points[{i}] requires a valid available source")
        _point(item, f"calibration_points[{i}]", width, height, frame_count, duration, allowed_fields=CALIBRATION_FIELDS)
    for i, item in enumerate(engine_points):
        if not isinstance(item, Mapping) or set(item) != ENGINE_POINT_FIELDS:
            _error(f"engine_points[{i}] requires exactly x and y")
        _number(item["x"], f"engine_points[{i}].x")
        _number(item["y"], f"engine_points[{i}].y")
    for field in ("golfer_anchor", "ball", "clubhead", "target_intended_direction", "landing"):
        _annotation(payload[field], field, width, height, frame_count, duration)
    _annotation(payload["contact"], "contact", width, height, frame_count, duration, point=False)
    if payload["contact"]["source"] != "unavailable":
        _event(payload["contact"]["value"], "contact", frame_count, duration)
    for field in ("club_selection", "context"):
        _annotation(payload[field], field, width, height, frame_count, duration, point=False)
    warnings = payload["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(w, str) or not w.strip() or AMBIGUOUS.search(w) for w in warnings):
        _error("warnings must be non-empty, unambiguous strings")
    _check_path_strings(payload)
    return copy.deepcopy(dict(payload))


@dataclass(frozen=True)
class HumanAnnotationDocument:
    payload: Mapping[str, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _validated(self.payload, self.strict))

    @property
    def status(self) -> str:
        return self.payload["status"]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))

    def to_json(self) -> str:
        return serialize_human_annotations(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], strict: bool = True) -> "HumanAnnotationDocument":
        return cls(payload, strict=strict)

    def submit(self, *, explicit_submit: bool = False) -> "HumanAnnotationDocument":
        if not explicit_submit:
            raise VideoContractError("submitted_without_explicit_submit")
        payload = self.to_dict()
        payload["status"] = "submitted"
        payload["explicit_submit"] = True
        return type(self)(payload, strict=self.strict)

    @classmethod
    def from_json(cls, text: str, strict: bool = True) -> "HumanAnnotationDocument":
        return cls(deserialize_human_annotations(text, strict=strict), strict=strict)


def validate_human_annotations(payload: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    return _validated(payload, strict)


def serialize_human_annotations(document: HumanAnnotationDocument | Mapping[str, Any]) -> str:
    payload = document.to_dict() if isinstance(document, HumanAnnotationDocument) else _validated(document)
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def deserialize_human_annotations(text: str, *, strict: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoContractError("malformed annotation JSON") from exc
    return _validated(payload, strict)
