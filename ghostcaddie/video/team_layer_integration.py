"""Fail-closed adapter for independent FairwayOS team layer handoffs.

Local research-only integration boundary. This module validates producer handoffs
before any renderer can consume them; it does not promote labels to ground truth
or call calibration/domain analytics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

VALID_STATES = {"visible", "observed", "unavailable", "unresolvable", "offscreen", "ambiguous", "occluded", "cut", "rejected"}
REQUIRED_FLAGS = {
    "pseudo_label": True,
    "ground_truth": False,
    "research_only": True,
    "production_eligible": False,
}


class LayerContractError(ValueError):
    """Raised when a layer handoff is unsafe to render."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerContractError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise LayerContractError(f"{field} must be finite")
    return out


def _validate_confidence(value: object, where: str) -> Optional[float]:
    if value is None:
        return None
    c = _finite_number(value, f"{where}.confidence")
    if c < 0.0 or c > 1.0:
        raise LayerContractError(f"{where}.confidence must be in [0,1]")
    return c


def _validate_geometry(geometry: object, state: str, where: str) -> Optional[dict]:
    if geometry is None:
        if state in {"visible", "observed"}:
            raise LayerContractError(f"{where}.geometry required when visible")
        return None
    if not isinstance(geometry, Mapping):
        raise LayerContractError(f"{where}.geometry must be object or null")
    gtype = geometry.get("type")
    if gtype == "point":
        return {
            "type": "point",
            "x": _finite_number(geometry.get("x"), f"{where}.geometry.x"),
            "y": _finite_number(geometry.get("y"), f"{where}.geometry.y"),
        }
    if gtype == "box":
        x1 = _finite_number(geometry.get("x1"), f"{where}.geometry.x1")
        y1 = _finite_number(geometry.get("y1"), f"{where}.geometry.y1")
        x2 = _finite_number(geometry.get("x2"), f"{where}.geometry.x2")
        y2 = _finite_number(geometry.get("y2"), f"{where}.geometry.y2")
        if x2 <= x1 or y2 <= y1:
            raise LayerContractError(f"{where}.geometry box must have positive area")
        return {"type": "box", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if gtype == "box_keypoints":
        box = geometry.get("box")
        if not isinstance(box, Mapping):
            raise LayerContractError(f"{where}.geometry.box must be object")
        x1 = _finite_number(box.get("x1"), f"{where}.geometry.box.x1")
        y1 = _finite_number(box.get("y1"), f"{where}.geometry.box.y1")
        x2 = _finite_number(box.get("x2"), f"{where}.geometry.box.x2")
        y2 = _finite_number(box.get("y2"), f"{where}.geometry.box.y2")
        if x2 <= x1 or y2 <= y1:
            raise LayerContractError(f"{where}.geometry box must have positive area")
        keypoints = geometry.get("keypoints", [])
        if not isinstance(keypoints, list):
            raise LayerContractError(f"{where}.geometry.keypoints must be list")
        clean_points = []
        for i, point in enumerate(keypoints):
            if not isinstance(point, Mapping):
                raise LayerContractError(f"{where}.geometry.keypoints[{i}] must be object")
            clean = {
                "x": _finite_number(point.get("x"), f"{where}.geometry.keypoints[{i}].x"),
                "y": _finite_number(point.get("y"), f"{where}.geometry.keypoints[{i}].y"),
            }
            if "confidence" in point and point.get("confidence") is not None:
                clean["confidence"] = _validate_confidence(point.get("confidence"), f"{where}.geometry.keypoints[{i}]")
            if isinstance(point.get("name"), str):
                clean["name"] = point["name"]
            clean_points.append(clean)
        return {"type": "box_keypoints", "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "keypoints": clean_points}
    if gtype == "keypoints":
        points = geometry.get("points")
        if not isinstance(points, list):
            raise LayerContractError(f"{where}.geometry.points must be a list")
        clean = []
        for i, point in enumerate(points):
            if not isinstance(point, Mapping):
                raise LayerContractError(f"{where}.geometry.points[{i}] must be object")
            clean.append({
                "x": _finite_number(point.get("x"), f"{where}.geometry.points[{i}].x"),
                "y": _finite_number(point.get("y"), f"{where}.geometry.points[{i}].y"),
            })
        return {"type": "keypoints", "points": clean}
    raise LayerContractError(f"{where}.geometry unsupported type {gtype!r}")


def _validate_observation(raw: object, layer_name: str, frame_size: Optional[Tuple[int, int]]) -> dict:
    if not isinstance(raw, Mapping):
        raise LayerContractError(f"{layer_name}.observations entry must be object")
    where = f"{layer_name}.observations[{raw.get('frame_index', '?')}]"
    frame = raw.get("frame_index")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise LayerContractError(f"{where}.frame_index must be non-negative integer")
    state = raw.get("state")
    if state not in VALID_STATES:
        raise LayerContractError(f"{where}.state unsupported")
    geometry = _validate_geometry(raw.get("geometry"), str(state), where)
    if geometry is not None and frame_size is not None:
        w, h = frame_size
        xs = []
        ys = []
        if geometry["type"] == "point":
            xs.append(geometry["x"]); ys.append(geometry["y"])
        elif geometry["type"] == "box":
            xs.extend([geometry["x1"], geometry["x2"]])
            ys.extend([geometry["y1"], geometry["y2"]])
        elif geometry["type"] == "box_keypoints":
            xs.extend([geometry["box"]["x1"], geometry["box"]["x2"]])
            ys.extend([geometry["box"]["y1"], geometry["box"]["y2"]])
            xs.extend(p["x"] for p in geometry["keypoints"])
            ys.extend(p["y"] for p in geometry["keypoints"])
        elif geometry["type"] == "keypoints":
            xs.extend(p["x"] for p in geometry["points"]); ys.extend(p["y"] for p in geometry["points"])
        if any(x < 0 or x > w for x in xs) or any(y < 0 or y > h for y in ys):
            raise LayerContractError(f"{where}.geometry out of source frame bounds")
    evidence_method = raw.get("evidence_method")
    if not isinstance(evidence_method, str) or not evidence_method.strip():
        raise LayerContractError(f"{where}.evidence_method required")
    manual = raw.get("manual_assistance")
    if not isinstance(manual, (bool, str)) or manual == "":
        raise LayerContractError(f"{where}.manual_assistance must be boolean or non-empty string")
    paths = raw.get("evidence_paths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise LayerContractError(f"{where}.evidence_paths must be string list")
    return {
        "frame_index": frame,
        "state": state,
        "visible": state in {"visible", "observed"},
        "geometry": geometry,
        "evidence_method": evidence_method,
        "confidence": _validate_confidence(raw.get("confidence"), where),
        "manual_assistance": manual,
        "evidence_paths": list(paths),
    }


def validate_layer_payload(payload: object, expected_layer: str, source_sha256: str,
                           frame_size: Optional[Tuple[int, int]] = None) -> dict:
    if not isinstance(payload, Mapping):
        raise LayerContractError("layer payload must be object")
    if payload.get("source_sha256") != source_sha256:
        raise LayerContractError(f"{expected_layer}.source_sha256 mismatch")
    layer_value = payload.get("layer")
    if layer_value is None and isinstance(payload.get("layer_name"), str):
        layer_value = expected_layer if expected_layer in str(payload.get("layer_name")) else None
    if layer_value != expected_layer:
        raise LayerContractError(f"layer name mismatch for {expected_layer}")
    for key, expected in REQUIRED_FLAGS.items():
        if payload.get(key) is not expected:
            raise LayerContractError(f"{expected_layer}.{key} must be {expected!r}")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise LayerContractError(f"{expected_layer}.observations must be list")
    clean = [_validate_observation(obs, expected_layer, frame_size) for obs in observations]
    seen = set()
    for obs in clean:
        frame = obs["frame_index"]
        if frame in seen:
            raise LayerContractError(f"{expected_layer}.observations duplicate frame {frame}")
        seen.add(frame)
    return {
        "source_sha256": source_sha256,
        "layer": expected_layer,
        **REQUIRED_FLAGS,
        "observations": clean,
    }


def _read_ready(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser text not important
        raise LayerContractError(f"invalid READY.json at {path}: {exc}") from exc


def _ready_status(data: Mapping) -> str:
    if data.get("ready") is True or data.get("status") == "ready" or data.get("readiness") == "usable":
        return "usable"
    if data.get("readiness") == "blocked" and data.get("ready_written_last") is True:
        return "blocked"
    return "missing"


def load_completed_layers(root: Path, expected_layers: Sequence[str], source_sha256: str,
                          frame_size: Optional[Tuple[int, int]] = None) -> Dict[str, dict]:
    layers: Dict[str, dict] = {}
    for layer in expected_layers:
        layer_dir = Path(root) / layer
        layer_path = layer_dir / "layer.json"
        ready_path = layer_dir / "READY.json"
        ready_data = _read_ready(ready_path)
        readiness = _ready_status(ready_data)
        if not layer_path.exists() or readiness == "missing":
            raise LayerContractError(f"missing completed layer {layer}")
        try:
            payload = json.loads(layer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LayerContractError(f"invalid JSON for layer {layer}: {exc}") from exc
        validated = validate_layer_payload(payload, layer, source_sha256, frame_size=frame_size)
        observed_count = sum(1 for obs in validated["observations"] if obs["visible"])
        if readiness == "blocked" and observed_count != 0:
            raise LayerContractError(f"blocked layer {layer} cannot contain observed geometry")
        validated["readiness"] = readiness
        validated["observed_count"] = observed_count
        layers[layer] = validated
    return layers


def _empty_state(layer: str, frame: int, reason: str) -> dict:
    return {"layer": layer, "frame_index": frame, "state": reason, "visible": False,
            "geometry": None, "history_len": 0}


def render_timeline_states(observations_by_layer: Mapping[str, Sequence[Mapping]],
                           frame_range: Iterable[int], cuts: Sequence[int] = ()) -> Dict[int, Dict[str, dict]]:
    """Build renderer-ready states with independent histories and hard cut resets."""
    by_layer: Dict[str, Dict[int, Mapping]] = {
        layer: {int(obs["frame_index"]): obs for obs in observations}
        for layer, observations in observations_by_layer.items()
    }
    histories: MutableMapping[str, list] = {layer: [] for layer in by_layer}
    cut_set = {int(c) for c in cuts}
    timeline: Dict[int, Dict[str, dict]] = {}
    for frame in frame_range:
        frame = int(frame)
        timeline[frame] = {}
        if frame in cut_set:
            for layer in by_layer:
                histories[layer] = []
                timeline[frame][layer] = _empty_state(layer, frame, "cut")
            continue
        for layer, obs_by_frame in by_layer.items():
            obs = obs_by_frame.get(frame)
            if obs is None:
                obs = {"frame_index": frame, "state": "unavailable", "visible": False, "geometry": None}
            visible = obs.get("state") in {"visible", "observed"} or obs.get("visible") is True
            if visible:
                histories[layer].append(obs.get("geometry"))
            else:
                histories[layer] = []
            state = dict(obs)
            state["layer"] = layer
            state["frame_index"] = frame
            state["visible"] = bool(visible)
            state["history_len"] = len(histories[layer])
            timeline[frame][layer] = state
    return timeline
