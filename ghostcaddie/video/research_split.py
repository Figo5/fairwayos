"""Frozen, research-only split manifest validation.

This contract checks dataset partition integrity without opening media or making
any claim about annotation quality. It is intentionally disconnected from all
inference and production analytics paths.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any, Mapping

from .errors import VideoContractError

SCHEMA_VERSION = "golf-research-split.v1"
RESEARCH_SPLIT_SCHEMA_VERSION = SCHEMA_VERSION
_TOP_LEVEL = frozenset({"schema_version", "dataset_id", "status", "clips", "warnings"})
_CLIP_FIELDS = frozenset({"clip_id", "source_id", "subject_id", "sequence_id", "sha256", "split"})
_SPLITS = frozenset({"train", "validation", "held_out"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _fail(message: str) -> None:
    raise VideoContractError(message)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _safe_strings(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"non-string field at {path}")
            if "path" in key.lower() or key.lower() in {"file", "filename"}:
                _fail(f"unsafe path field at {path}.{key}")
            _safe_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _safe_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if os.path.isabs(value) or value.startswith(("~/", "~\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            _fail(f"absolute path at {path}")


def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        _fail("manifest must be an object")
    keys = set(payload)
    if keys != _TOP_LEVEL:
        missing, extra = _TOP_LEVEL - keys, keys - _TOP_LEVEL
        if missing:
            _fail(f"missing required field: {sorted(missing)[0]}")
        _fail(f"fabricated field: {sorted(extra)[0]}")
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail("invalid schema_version")
    _text(payload["dataset_id"], "dataset_id")
    if payload["status"] != "frozen":
        _fail("split manifest must have frozen status")
    clips = payload["clips"]
    if not isinstance(clips, list) or not clips:
        _fail("clips must be a non-empty list")
    warnings = payload["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) or not item.strip() for item in warnings):
        _fail("warnings must be a list of non-empty strings")

    seen_clip, seen_hash = set(), set()
    memberships = {field: {} for field in ("source_id", "subject_id", "sequence_id")}
    seen_splits = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, Mapping) or set(clip) != _CLIP_FIELDS:
            _fail(f"clips[{index}] has invalid fields")
        values = {field: _text(clip[field], f"clips[{index}].{field}") for field in _CLIP_FIELDS - {"split"}}
        split = clip["split"]
        if split not in _SPLITS:
            _fail(f"invalid split at clips[{index}]")
        if not _SHA256.fullmatch(values["sha256"]):
            _fail(f"clips[{index}].sha256 must be a lowercase SHA-256 hex digest")
        if values["clip_id"] in seen_clip:
            _fail("duplicate clip_id")
        if values["sha256"] in seen_hash:
            _fail("duplicate sha256; duplicate or re-encoded clips cannot cross evaluation units")
        seen_clip.add(values["clip_id"])
        seen_hash.add(values["sha256"])
        seen_splits.add(split)
        for field in memberships:
            prior = memberships[field].get(values[field])
            if prior is not None and prior != split:
                _fail(f"{field} leakage across splits")
            memberships[field][values[field]] = split
    missing = _SPLITS - seen_splits
    if missing:
        _fail(f"missing required split: {sorted(missing)[0]}")
    _safe_strings(payload)
    return copy.deepcopy(dict(payload))


def validate_split_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a frozen golfer/source/sequence-disjoint manifest."""
    return _validate(payload)


def serialize_split_manifest(payload: Mapping[str, Any]) -> str:
    """Serialize a validated manifest deterministically and reject non-finite JSON."""
    return json.dumps(_validate(payload), allow_nan=False, sort_keys=True, separators=(",", ":"))


__all__ = ["SCHEMA_VERSION", "RESEARCH_SPLIT_SCHEMA_VERSION", "validate_split_manifest", "serialize_split_manifest"]
