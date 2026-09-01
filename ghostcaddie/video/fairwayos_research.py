"""Research-only FairwayOS sidecar for the shared ball candidate track.

This boundary serializes diagnostic evidence for inspection and handoff. It does
not import or construct production observations, shot events, analytics, or
automatic-perception gates. Missing candidates remain unavailable and human
review remains the fallback.
"""

from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from .research_ball import BallTrackItem, BallTrackResult

FAIRWAYOS_BALL_RESEARCH_SCHEMA_VERSION = "fairwayos-ball-research.v1"


def _json_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _safe_source(source: Optional[str]) -> Optional[str]:
    if source is None:
        return None
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError("source must be a relative identifier")
    text = os.fspath(source)
    if not isinstance(text, str) or Path(text).is_absolute():
        raise ValueError("source must be a relative identifier")
    if text.startswith(("~/", "~\\")):
        raise ValueError("source must be a project-relative identifier")
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        raise ValueError("source must be a project-relative identifier")
    normalized = text.replace("\\", "/")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError("source must not contain traversal")
    return text


def build_fairwayos_sidecar(
    result: BallTrackResult,
    *,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Convert a shared research track into a stable, non-production sidecar."""
    if not isinstance(result, BallTrackResult):
        raise TypeError("result must be a BallTrackResult")
    items = []
    for item in result.items:
        confidence = _json_number(item.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        center = None
        if item.center is not None:
            center = [_json_number(item.center[0], "center.x"),
                      _json_number(item.center[1], "center.y")]
        items.append({
            "frame_index": item.frame_index,
            "center": center,
            "confidence": confidence,
            "provenance": item.provenance,
            "warnings": list(item.warnings),
        })
    observed = sum(item["center"] is not None for item in items)
    return {
        "schema_version": FAIRWAYOS_BALL_RESEARCH_SCHEMA_VERSION,
        "source": _safe_source(source),
        "production_eligible": False,
        "human_fallback": {
            "status": "available",
            "required_for_domain_use": True,
            "reason": "shared research candidates are not validated golf-ball observations",
        },
        "track": {
            "track_id": result.track_id,
            "observed_frames": observed,
            "frame_count": len(items),
            "longest_gap": result.longest_gap,
            "items": items,
        },
    }


def sidecar_from_mapping(payload: Mapping[str, Any], *, source: Optional[str] = None) -> dict[str, Any]:
    """Load the track JSON emitted by a research runner, without domain promotion."""
    if not isinstance(payload, Mapping):
        raise ValueError("track input must be a JSON object")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("track input requires an items list")
    items = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("each track item must be an object")
        frame_value = raw.get("frame_index", raw.get("frame"))
        if frame_value is None:
            raise ValueError("each track item requires frame_index or frame")
        if isinstance(frame_value, bool):
            raise ValueError("frame_index must be an integer")
        center = raw.get("center")
        if center is None and raw.get("x") is not None and raw.get("y") is not None:
            center = (raw["x"], raw["y"])
        if center is not None:
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError("center must be null or a two-item array")
            center = (center[0], center[1])
        confidence_value = raw.get("confidence", 0.0)
        confidence = _json_number(confidence_value, "confidence")
        items.append(BallTrackItem(
            int(frame_value), center, confidence,
            str(raw.get("provenance", raw.get("state", "unavailable"))),
            tuple(raw.get("warnings", ()))
        ))
    result = BallTrackResult(str(payload.get("track_id", "ball-0")), tuple(items),
                             int(payload.get("longest_gap", 0)))
    return build_fairwayos_sidecar(result, source=source)


def write_fairwayos_sidecar(path: str | Path, sidecar: Mapping[str, Any]) -> None:
    import json
    if not isinstance(sidecar, Mapping) or sidecar.get("production_eligible") is not False:
        raise ValueError("FairwayOS research sidecars must set production_eligible to false")
    Path(path).write_text(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n")
