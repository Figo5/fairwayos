"""Shared validation and provenance for synthetic provider shot adapters."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


class ProviderAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderDiagnostics:
    provider: str
    schema_version: str
    source_record_id: str
    unknown_fields: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "schema_version": self.schema_version,
            "source_record_id": self.source_record_id,
            "unknown_fields": list(self.unknown_fields),
        }


def require_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ProviderAdapterError(f"{name} must be a dict")
    return value


def require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderAdapterError(f"{name} must be a non-empty string")
    return value


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProviderAdapterError(f"{name} must be finite")
    return result


def positive(value, name: str) -> float:
    result = number(value, name)
    if result <= 0:
        raise ProviderAdapterError(f"{name} must be positive")
    return result


def integer(value, name: str) -> int:
    result = positive(value, name)
    if result != int(result):
        raise ProviderAdapterError(f"{name} must be an integer")
    return int(result)


def envelope_metadata(raw: dict, provider: str, schema_version: str,
                      required: Iterable[str], allowed: Iterable[str], strict: bool):
    require_dict(raw, "provider payload")
    if raw.get("provider") != provider:
        raise ProviderAdapterError(f"provider must be {provider!r}")
    if raw.get("schema_version") != schema_version:
        raise ProviderAdapterError(f"schema_version must be {schema_version!r}")
    for field in required:
        if field not in raw:
            raise ProviderAdapterError(f"missing required field {field!r}")
    unknown = tuple(sorted(set(raw) - set(allowed)))
    if strict and unknown:
        raise ProviderAdapterError("unknown fields: " + ", ".join(unknown))
    source_id = require_str(raw["source_record_id"], "source_record_id")
    return ProviderDiagnostics(provider, schema_version, source_id, unknown)


def context_point(context: dict, name: str):
    from ..geometry import Point2D
    value = require_dict(context.get(name), f"course_context.{name}")
    return Point2D(number(value.get("x"), f"course_context.{name}.x"),
                   number(value.get("y"), f"course_context.{name}.y"))


def map_point(mapper, point):
    """Apply the existing mapper's dict boundary exactly once."""
    from ..geometry import Point2D
    mapped = mapper.to_engine({"x": point.x, "y": point.y})
    if isinstance(mapped, Point2D):
        return mapped
    return Point2D(number(mapped.get("x"), "mapped.x"), number(mapped.get("y"), "mapped.y"))


def ensure_context(course_context, player_id, name):
    require_dict(course_context, "course_context")
    if not course_context:
        raise ProviderAdapterError(f"{name} requires course context")
    if not player_id:
        raise ProviderAdapterError(f"{name} requires player context")


def provenance(diag: ProviderDiagnostics, transform: str) -> Dict[str, object]:
    return {
        "provider": diag.provider,
        "schema_version": diag.schema_version,
        "source_record_id": diag.source_record_id,
        "coordinate_transform": transform,
        "unknown_fields": list(diag.unknown_fields),
        "data_disclaimer": "Synthetic/mock provider payload only; no live integration.",
    }
