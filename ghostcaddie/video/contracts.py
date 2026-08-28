"""Versioned, serialization-safe contracts for video diagnostics."""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from .errors import VideoContractError

SCHEMA_VERSION = "video-diagnostics.v1"
_SENSITIVE = re.compile(r"(?:secret|password|passwd|api[_ -]?key|token|prompt|environment|env(?:ironment)?[_ -]?var)", re.I)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VideoContractError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class VideoMetadata:
    container_format: str
    codec: str
    width: int
    height: int
    frame_rate: float
    duration_seconds: float
    frame_count: Optional[int] = None
    source_identifier: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.container_format, str) or not self.container_format.strip():
            raise VideoContractError("container_format must be a non-empty string")
        if not isinstance(self.codec, str) or not self.codec.strip():
            raise VideoContractError("codec must be a non-empty string")
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VideoContractError(f"{name} must be a positive integer")
        rate = _finite_number(self.frame_rate, "frame_rate")
        duration = _finite_number(self.duration_seconds, "duration_seconds")
        if rate <= 0 or duration < 0:
            raise VideoContractError("frame_rate must be positive and duration_seconds non-negative")
        if self.frame_count is not None and (
            isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count < 0
        ):
            raise VideoContractError("frame_count must be a non-negative integer when available")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "container_format": self.container_format,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "duration_seconds": self.duration_seconds,
        }
        if self.frame_count is not None:
            result["frame_count"] = self.frame_count
        return result


def _safe(value: Any, path: str = "root") -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if not isinstance(key, str) or _SENSITIVE.search(key):
                raise VideoContractError(f"unsafe diagnostic field at {path}")
            clean[key] = _safe(item, f"{path}.{key}")
        return clean
    if isinstance(value, (list, tuple)):
        return [_safe(item, f"{path}[]") for item in value]
    if isinstance(value, str):
        windows_absolute = bool(re.match(r"^[A-Za-z]:[\\\\/]", value))
        if os.path.isabs(value) or value.startswith(("~/", "~\\\\")) or windows_absolute:
            raise VideoContractError(f"absolute path in diagnostic payload at {path}")
        if _SENSITIVE.search(value) or "=" in value and _SENSITIVE.search(value.split("=", 1)[0]):
            raise VideoContractError(f"unsafe diagnostic text at {path}")
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise VideoContractError(f"non-finite diagnostic value at {path}")
        return value
    raise VideoContractError(f"unsupported diagnostic value at {path}")


@dataclass
class VideoDiagnostics:
    status: str = "pending"
    video_metadata: Optional[Any] = None
    artifact_references: List[str] = field(default_factory=list)
    frame_observations: List[Dict[str, Any]] = field(default_factory=list)
    contact: Optional[Dict[str, Any]] = None
    landing: Optional[Dict[str, Any]] = None
    normalized_shot: Optional[Dict[str, Any]] = None
    analytics_result: Optional[Dict[str, Any]] = None
    confidence_values: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    model_provider_provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"pending", "complete", "failed"}:
            raise VideoContractError("status must be pending, complete, or failed")
        for ref in self.artifact_references:
            parts = PurePosixPath(ref).parts if isinstance(ref, str) else ()
            if not isinstance(ref, str) or not ref or os.path.isabs(ref) or "\\" in ref or ".." in parts:
                raise VideoContractError("artifact references must be relative names")
        for key, value in self.confidence_values.items():
            number = _finite_number(value, f"confidence_values.{key}")
            if not 0 <= number <= 1:
                raise VideoContractError("confidence values must be between 0 and 1")
        # Validate all free-form diagnostic payloads at construction time.
        _safe(self.frame_observations, "frame_observations")
        _safe(self.contact, "contact")
        _safe(self.landing, "landing")
        _safe(self.normalized_shot, "normalized_shot")
        _safe(self.analytics_result, "analytics_result")
        _safe(self.warnings, "warnings")
        _safe(self.model_provider_provenance, "model_provider_provenance")

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.video_metadata, VideoMetadata):
            metadata = self.video_metadata.to_dict()
        elif isinstance(self.video_metadata, dict):
            metadata = {key: value for key, value in self.video_metadata.items() if key != "source_identifier"}
            metadata = _safe(metadata, "video_metadata")
        else:
            metadata = _safe(self.video_metadata, "video_metadata")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "video_metadata": metadata,
            "artifact_references": list(self.artifact_references),
            "frame_observations": self.frame_observations,
            "contact": self.contact,
            "landing": self.landing,
            "normalized_shot": self.normalized_shot,
            "analytics_result": self.analytics_result,
            "confidence_values": dict(self.confidence_values),
            "warnings": list(self.warnings),
            "model_provider_provenance": self.model_provider_provenance,
        }
        _safe(payload)
        json.dumps(payload, allow_nan=False)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)
