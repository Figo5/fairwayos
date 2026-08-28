"""Deterministic, safe reports and rendering boundary for automatic perception.

This module deliberately does not run models or decide whether a shot is valid.
It only turns already-produced evidence into a stable, reviewable artifact and
reuses the existing pixel annotation renderer for visual output.
"""

import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Optional

from .automatic_perception import (
    AUTOMATIC_PERCEPTION_SCHEMA_VERSION, ConfidenceMetrics, ContinuityMetrics,
    GateDecision, Thresholds,
)
from .errors import VideoContractError

_SCHEMA_VERSION = "automatic-render.v1"
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://", re.I)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "x") and hasattr(value, "y"):
        return {"x": value.x, "y": value.y}
    return value


def _safe(value: Any, where: str = "root") -> Any:
    value = _plain(value)
    if isinstance(value, Mapping):
        return {k: _safe(v, f"{where}.{k}") for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_safe(v, f"{where}[]") for v in value]
    if isinstance(value, str):
        if os.path.isabs(value) or value.startswith(("~/", "~\\")) or _URL.match(value):
            raise ValueError(f"absolute path or URL is not allowed at {where}")
        if re.match(r"^[A-Za-z]:[\\/]", value) or "\\" in value:
            raise ValueError(f"unsafe path is not allowed at {where}")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {where}")
    return value


def _relative_refs(values: Iterable[str], field: str) -> list[str]:
    refs = []
    for ref in values or ():
        if not isinstance(ref, str) or not ref or os.path.isabs(ref) or _URL.match(ref):
            raise ValueError(f"{field} must contain relative artifact names")
        if ref.startswith(("~/", "~\\")) or "\\" in ref or ".." in PurePosixPath(ref).parts:
            raise ValueError(f"{field} must contain safe relative artifact names")
        refs.append(ref)
    return sorted(set(refs))


def _metric(value: Any, reason: Optional[str] = None) -> Any:
    return None if value is None else _safe(value)


def build_automatic_report(
    frame_results: Iterable[Any], *, gate_decision: Optional[GateDecision] = None,
    thresholds: Optional[Thresholds] = None,
    confidence_metrics: Optional[ConfidenceMetrics] = None,
    continuity_metrics: Optional[ContinuityMetrics] = None,
    artifact_references: Iterable[str] = (), visual_references: Iterable[str] = (),
    gate_failures: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a JSON-safe automatic-perception report without source locations."""
    gate = _plain(gate_decision) if gate_decision is not None else {
        "status": "unavailable", "passed": False, "blocking_reasons": []
    }
    if gate_decision is not None:
        gate["blocking_reasons"] = sorted(gate.get("blocking_reasons", ()))
    else:
        gate["blocking_reasons"] = sorted(set(gate_failures))
    frames = [_safe(item, "frames") for item in frame_results]
    frames.sort(key=lambda item: (item.get("frame_index", 0), json.dumps(item, sort_keys=True)))
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "perception_schema_version": AUTOMATIC_PERCEPTION_SCHEMA_VERSION,
        "frames": frames,
        "gate": _safe(gate),
        "thresholds": _safe(thresholds if thresholds is not None else {"provisional": True}),
        "confidence": _safe(confidence_metrics) if confidence_metrics is not None else None,
        "continuity": _safe(continuity_metrics) if continuity_metrics is not None else None,
        "artifact_references": _relative_refs(artifact_references, "artifact_references"),
        "visual_references": _relative_refs(visual_references, "visual_references"),
    }
    return _safe(payload)


def serialize_automatic_report(report: Mapping[str, Any]) -> str:
    """Serialize an automatic report canonically (stable key ordering)."""
    return json.dumps(_safe(report), allow_nan=False, sort_keys=True, separators=(",", ":"))


def build_evaluation_report(
    *, track_continuity=None, anchor_error=None, impact_error=None,
    ball_precision_recall=None, clubhead_precision_recall=None, landing_error=None,
    false_positives=None, runtime=None, unavailable_reasons: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Build the fixed evaluation schema; unavailable values remain JSON null."""
    metrics = {
        "track_continuity": _metric(track_continuity),
        "anchor_error": _metric(anchor_error),
        "impact_error": _metric(impact_error),
        "ball_precision_recall": _metric(ball_precision_recall),
        "clubhead_precision_recall": _metric(clubhead_precision_recall),
        "landing_error": _metric(landing_error),
        "false_positives": _metric(false_positives),
        "runtime": _metric(runtime),
    }
    reasons = {str(k): str(v) for k, v in (unavailable_reasons or {}).items() if metrics.get(k) is None}
    for key, value in metrics.items():
        if value is None and key not in reasons:
            reasons[key] = "evidence unavailable"
    return _safe({"schema_version": "automatic-evaluation.v1", "metrics": metrics,
                  "unavailable_reasons": reasons})


def serialize_evaluation_report(report: Mapping[str, Any]) -> str:
    return json.dumps(_safe(report), allow_nan=False, sort_keys=True, separators=(",", ":"))


def render_automatic_frame(frame_path, output_path, observation, calibration=None, *, ffmpeg="ffmpeg"):
    """Render a validated automatic observation through the safe existing renderer."""
    from .observations import PixelObservation
    from .annotations import annotate_frame
    if not isinstance(observation, PixelObservation):
        raise TypeError("automatic renderer requires a validated PixelObservation")
    return annotate_frame(frame_path, output_path, observation, calibration, ffmpeg=ffmpeg)


# Friendly aliases for callers integrating the boundary without CLI wiring.
build_report = build_automatic_report
serialize_report = serialize_automatic_report
build_evaluation = build_evaluation_report
