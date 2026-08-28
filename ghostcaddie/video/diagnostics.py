"""Deterministic assembly of the complete video diagnostics report."""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

from .contracts import VideoDiagnostics
from .observations import VideoObservations


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if hasattr(value, "x") and hasattr(value, "y"):
        return {"x": value.x, "y": value.y}
    return value


def _event_dict(reconstruction):
    if reconstruction is None:
        return None
    event = getattr(reconstruction, "shot_event", getattr(reconstruction, "event", None))
    metadata = getattr(reconstruction, "metadata", {})
    return {"event": _plain(event), "metadata": _plain(metadata)}


def _analytics_dict(analytics_result):
    if analytics_result is None:
        return {"status": "unavailable", "reason": "analytics result unavailable"}
    return _plain(analytics_result)


def build_video_diagnostics(
    observations: VideoObservations,
    video_metadata: Any = None,
    *,
    artifact_references: Optional[Iterable[str]] = None,
    reconstruction=None,
    analytics_result=None,
    calibration=None,
    warnings: Optional[Iterable[str]] = None,
    model_provider_provenance: Optional[Dict[str, Any]] = None,
    status: str = "complete",
) -> VideoDiagnostics:
    """Build all contract fields without leaking source paths or model inputs."""
    if not isinstance(observations, VideoObservations):
        raise TypeError("observations must be validated VideoObservations")
    frame_observations = [_plain(item.to_dict()) for item in observations.items]
    contacts = [item for item in observations.items if item.contact is not None]
    landings = [item for item in observations.items if item.landing is not None]
    def mapped(point):
        if calibration is None or point is None:
            return None
        mapped_point = calibration.to_engine(point)
        return {"x": mapped_point.x, "y": mapped_point.y, "units": calibration.engine_units}
    contact = _plain({"frame_index": contacts[0].frame_index, "timestamp_seconds": contacts[0].timestamp_seconds, "point": contacts[0].contact, "mapped": mapped(contacts[0].contact)}) if contacts else {"status": "unavailable", "reason": "contact unavailable"}
    landing = _plain({"frame_index": landings[0].frame_index, "timestamp_seconds": landings[0].timestamp_seconds, "point": landings[0].landing, "mapped": mapped(landings[0].landing)}) if landings else {"status": "unavailable", "reason": "landing unavailable"}
    confidence = {}
    for item in observations.items:
        confidence[f"frame_{item.frame_index:06d}_golfer"] = item.golfer.confidence
    if contacts:
        confidence["contact"] = contacts[0].contact["confidence"]
    if landings:
        confidence["landing"] = landings[0].landing["confidence"]
    if confidence:
        confidence["overall"] = min(confidence.values())
    all_warnings = sorted(set(warnings or ()) | {warning for item in observations.items for warning in item.warnings})
    if not contacts:
        all_warnings.append("contact unavailable")
    if not landings:
        all_warnings.append("landing unavailable")
    all_warnings = sorted(set(all_warnings))
    provenance = model_provider_provenance or {"model": "none", "provider": "none", "mode": "deterministic-fixture"}
    return VideoDiagnostics(status=status, video_metadata=_plain(video_metadata) if video_metadata is not None else {},
        artifact_references=sorted(set(artifact_references or ())), frame_observations=frame_observations,
        contact=contact, landing=landing, normalized_shot=_event_dict(reconstruction),
        analytics_result=_analytics_dict(analytics_result), confidence_values=confidence,
        warnings=all_warnings, model_provider_provenance=_plain(provenance))


def serialize_video_diagnostics(diagnostics: VideoDiagnostics) -> str:
    if not isinstance(diagnostics, VideoDiagnostics):
        raise TypeError("diagnostics must be VideoDiagnostics")
    return diagnostics.to_json()

build_diagnostics = build_video_diagnostics
serialize_diagnostics = serialize_video_diagnostics
