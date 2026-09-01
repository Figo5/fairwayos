"""Research-only frame-level shaft and clubhead annotation contract."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import VideoContractError

SCHEMA_VERSION = "golf-research-clubhead-annotations.v1"
_TOP_LEVEL = frozenset({
    "schema_version", "status", "explicit_submit", "video", "split", "rights",
    "provenance", "frames", "warnings",
})
_VIDEO_FIELDS = frozenset({"clip_id", "width", "height", "frame_count", "frame_rate"})
_PROVENANCE_FIELDS = frozenset({"label_type", "pseudo_label", "ground_truth", "research_only", "production_eligible"})
_FRAME_FIELDS = frozenset({"frame_index", "source_frame_index", "timestamp_seconds", "clubhead", "shaft", "notes"})
_LABEL_FIELDS = frozenset({"value", "visibility", "source"})
_POINT_FIELDS = frozenset({"x", "y"})
_SHAFT_FIELDS = frozenset({"grip", "neck"})
_VISIBILITY = frozenset({"visible", "occluded", "ambiguous", "unavailable"})
_SOURCES = frozenset({"human_ground_truth", "human_confirmed", "model_prediction", "pseudo_label", "unavailable"})
_SPLITS = frozenset({"train", "validation", "test", "unassigned"})
_AMBIGUOUS = re.compile(r"\b(?:ambiguous|unclear|unknown|maybe|possibly|unsure|unverified)\b", re.I)


def _fail(message: str) -> None:
    raise VideoContractError(message)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    return value


def _safe_strings(value: Any, path: str = "dataset") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"non-string field at {path}")
            if "path" in key.lower() or key.lower() in {"file", "filename", "source_identifier"}:
                _fail(f"unsafe path field at {path}.{key}")
            _safe_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _safe_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if os.path.isabs(value) or value.startswith(("~/", "~\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            _fail(f"absolute path at {path}")


def _point(value: Any, name: str, width: int, height: int) -> None:
    if not isinstance(value, Mapping) or set(value) != _POINT_FIELDS:
        _fail(f"{name} must contain exactly x and y")
    x, y = _finite(value["x"], f"{name}.x"), _finite(value["y"], f"{name}.y")
    if not 0 <= x <= width or not 0 <= y <= height:
        _fail(f"{name} out of bounds")


def _label(value: Any, name: str, width: int, height: int, *, shaft: bool = False) -> None:
    if not isinstance(value, Mapping) or set(value) != _LABEL_FIELDS:
        _fail(f"{name} must contain exactly value, visibility, and source")
    visibility, source = value["visibility"], value["source"]
    if visibility not in _VISIBILITY:
        _fail(f"invalid visibility for {name}")
    if source not in _SOURCES:
        _fail(f"invalid source for {name}")
    point_value = value["value"]
    if visibility == "visible":
        if point_value is None:
            _fail(f"visible {name} requires a value")
        if shaft:
            if not isinstance(point_value, Mapping) or set(point_value) != _SHAFT_FIELDS:
                _fail(f"{name} must contain exactly grip and neck")
            _point(point_value["grip"], f"{name}.grip", width, height)
            _point(point_value["neck"], f"{name}.neck", width, height)
        else:
            _point(point_value, name, width, height)
        if source in {"unavailable", "pseudo_label"}:
            _fail(f"visible {name} has invalid source")
    else:
        if point_value is not None or source != "unavailable":
            _fail(f"{name} non-visible state must have null value and unavailable source")


def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _fail("dataset must be an object")
    if set(payload) != _TOP_LEVEL:
        missing, extra = _TOP_LEVEL - set(payload), set(payload) - _TOP_LEVEL
        if missing:
            _fail(f"missing required field: {sorted(missing)[0]}")
        _fail(f"fabricated field: {sorted(extra)[0]}")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("invalid schema_version")
    if payload["status"] not in {"draft", "submitted"} or not isinstance(payload["explicit_submit"], bool):
        _fail("invalid status or explicit_submit")
    if payload["status"] == "submitted" and not payload["explicit_submit"]:
        _fail("submitted_without_explicit_submit")
    if payload["status"] == "draft" and payload["explicit_submit"]:
        _fail("draft cannot be explicitly submitted")
    video = payload["video"]
    if not isinstance(video, Mapping) or set(video) != _VIDEO_FIELDS:
        _fail("video has invalid fields")
    clip_id = video["clip_id"]
    if not isinstance(clip_id, str) or not clip_id.strip():
        _fail("video.clip_id must be non-empty")
    width, height = _integer(video["width"], "video.width"), _integer(video["height"], "video.height")
    count, rate = _integer(video["frame_count"], "video.frame_count"), _finite(video["frame_rate"], "video.frame_rate")
    if width <= 0 or height <= 0 or count <= 0 or rate <= 0:
        _fail("video dimensions, frame_count, and frame_rate must be positive")
    if payload["split"] not in _SPLITS:
        _fail("invalid split")
    if not isinstance(payload["rights"], str) or not payload["rights"].strip():
        _fail("rights must be non-empty")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_FIELDS:
        _fail("provenance has invalid fields")
    if not isinstance(provenance["label_type"], str) or provenance["label_type"] not in {"human", "pseudo", "model"}:
        _fail("invalid provenance.label_type")
    for field in ("pseudo_label", "ground_truth", "research_only", "production_eligible"):
        if not isinstance(provenance[field], bool):
            _fail(f"provenance.{field} must be boolean")
    if provenance["pseudo_label"] and (provenance["ground_truth"] or not provenance["research_only"] or provenance["production_eligible"]):
        _fail("pseudo labels require ground_truth=false, research_only=true, production_eligible=false")
    if provenance["production_eligible"]:
        _fail("production_eligible must remain false for research annotations")
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        _fail("frames must be a non-empty list")
    if count != len(frames):
        _fail("video.frame_count must match the number of annotation frames")
    previous_source = None
    for expected, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or set(frame) != _FRAME_FIELDS:
            _fail(f"frames[{expected}] has invalid fields")
        if _integer(frame["frame_index"], f"frames[{expected}].frame_index") != expected:
            _fail("frame_index values must be consecutive and zero-based")
        source_index = _integer(frame["source_frame_index"], f"frames[{expected}].source_frame_index")
        if source_index < 0 or (previous_source is not None and source_index <= previous_source):
            _fail("source_frame_index values must be strictly increasing")
        previous_source = source_index
        timestamp = _finite(frame["timestamp_seconds"], f"frames[{expected}].timestamp_seconds")
        if timestamp < 0:
            _fail("timestamp_seconds must not be negative")
        _label(frame["clubhead"], f"frames[{expected}].clubhead", width, height)
        _label(frame["shaft"], f"frames[{expected}].shaft", width, height, shaft=True)
        notes = frame["notes"]
        if not isinstance(notes, list) or any(not isinstance(note, str) or not note.strip() or _AMBIGUOUS.search(note) for note in notes):
            _fail("frame notes must be non-empty, unambiguous strings")
    warnings = payload["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(warning, str) or not warning.strip() or _AMBIGUOUS.search(warning) for warning in warnings):
        _fail("warnings must be non-empty, unambiguous strings")
    _safe_strings(payload)
    return copy.deepcopy(dict(payload))


@dataclass(frozen=True)
class ClubheadAnnotationDataset:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _validate(self.payload))

    @property
    def frames(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.payload["frames"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))

    @classmethod
    def from_json(cls, text: str) -> "ClubheadAnnotationDataset":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise VideoContractError("malformed annotation dataset JSON") from exc
        return cls(payload)


def validate_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _validate(payload)


def serialize_dataset(dataset: ClubheadAnnotationDataset | Mapping[str, Any]) -> str:
    payload = dataset.to_dict() if isinstance(dataset, ClubheadAnnotationDataset) else _validate(dataset)
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
