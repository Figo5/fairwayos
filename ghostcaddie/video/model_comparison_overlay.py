"""Research-only visual comparison of local model candidates.

This module renders diagnostic provenance, not golf-ball identity. Candidate
markers are suppressed for unavailable/rejected states; every visible marker is
labelled with its backend family.
"""
from __future__ import annotations

import math
from typing import Mapping, Any

_BACKENDS = ("PT", "ONNX", "GENERIC")
_ACTIVE = {"candidate", "observed"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def build_comparison_overlay(*, frame_index: int, width: int, height: int,
                             candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build a safe per-frame render plan from already-produced local outputs."""
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be positive integer")
    if set(candidates) - set(_BACKENDS):
        raise ValueError("candidate labels must be PT, ONNX, or GENERIC")
    markers, unavailable = [], []
    for label in _BACKENDS:
        raw = candidates.get(label, {"state": "unavailable"})
        state = raw.get("state", "unavailable")
        if state in ("unavailable", "rejected"):
            unavailable.append(label)
            continue
        if state not in _ACTIVE:
            raise ValueError(f"unsupported state for {label}: {state}")
        x, y = _number(raw.get("x"), f"{label}.x"), _number(raw.get("y"), f"{label}.y")
        confidence = _number(raw.get("confidence"), f"{label}.confidence")
        if not (0 <= x < width and 0 <= y < height and 0 <= confidence <= 1):
            raise ValueError(f"{label} candidate is out of bounds")
        markers.append({"label": label, "state": state, "x": x, "y": y, "confidence": confidence})
    return {
        "frame_index": frame_index,
        "markers": markers,
        "unavailable": unavailable,
        "identity": "unavailable",
        "production_eligible": False,
        "research_only": True,
    }


def comparison_filter(*, frame_index: int, width: int, height: int,
                      candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Return an FFmpeg filter graph with labelled markers and diagnostic bars."""
    plan = build_comparison_overlay(frame_index=frame_index, width=width, height=height, candidates=candidates)
    filters = [
        f"drawbox=x=0:y=0:w={width}:h=5:color=blue:t=fill",
        f"drawbox=x=0:y={height-5}:w={width}:h=5:color=red:t=fill",
        f"drawtext=text='RESEARCH ONLY | NOT GOLF-BALL IDENTITY | IDENTITY UNAVAILABLE':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65",
    ]
    y = 32
    for label in _BACKENDS:
        if label in plan["unavailable"]:
            text, color = f"{label}: UNAVAILABLE", "gray"
        else:
            item = next(m for m in plan["markers"] if m["label"] == label)
            text, color = f"{label}: CANDIDATE {item['confidence']:.2f}", {"PT": "yellow", "ONNX": "cyan", "GENERIC": "magenta"}[label]
            filters.append(f"drawbox=x={item['x']-10:g}:y={item['y']-10:g}:w=20:h=20:color={color}:t=2:enable='eq(n\\,{frame_index})'")
        filters.append(f"drawtext=text='{text}':x=8:y={y}:fontsize=15:fontcolor={color}:box=1:boxcolor=black@0.5")
        y += 22
    return ",".join(filters)
