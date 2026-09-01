"""Import submitted human annotations at the video-to-shot boundary."""

from __future__ import annotations

from typing import Any

from .errors import VideoContractError, VideoPathError
from .human_contracts import HumanAnnotationDocument
from .observations import VideoObservations
from .paths import ProjectBoundary
from .reconstruction import ReconstructionResult, ShotContext, reconstruct_shot


def _value(document: HumanAnnotationDocument, field: str, required: bool = True):
    annotation = document.payload[field]
    if annotation["source"] == "unavailable" or annotation["value"] is None:
        if required:
            raise VideoContractError(f"required evidence unavailable: {field}")
        return None
    return annotation["value"]


def _observations(document: HumanAnnotationDocument) -> VideoObservations:
    payload = document.payload
    video = payload["video"]
    anchor = _value(document, "golfer_anchor")
    contact = _value(document, "contact")
    landing = _value(document, "landing")
    club = _value(document, "club_selection")
    if not isinstance(club, str) or not club.strip():
        raise VideoContractError("required evidence unavailable: club_selection")
    context = _value(document, "context")
    if not isinstance(context, dict) or not isinstance(context.get("lie"), str) or not context["lie"].strip():
        raise VideoContractError("required evidence unavailable: context.lie")

    # Contact coordinates are optional in the contract only when unavailable;
    # when supplied, preserve the user-selected pixel exactly.
    contact_pixel = contact
    if not all(key in contact_pixel for key in ("x", "y")):
        raise VideoContractError("contact coordinate is unavailable")
    items = [
        {"frame_index": anchor["frame_index"], "timestamp_seconds": anchor["timestamp_seconds"],
         "golfer": {"bbox": {"x": anchor["x"], "y": anchor["y"], "width": 1.0, "height": 1.0},
                    "anchor": {"x": anchor["x"], "y": anchor["y"]}, "confidence": anchor["confidence"]},
         "club": {"name": club, "confidence": anchor["confidence"]}, "clubhead": None,
         "ball": None, "phase": "address", "contact": None, "intended_direction": None,
         "landing": None, "warnings": ["ball_missing"]},
        {"frame_index": contact["frame_index"], "timestamp_seconds": contact["timestamp_seconds"],
         "golfer": {"bbox": {"x": anchor["x"], "y": anchor["y"], "width": 1.0, "height": 1.0},
                    "anchor": {"x": anchor["x"], "y": anchor["y"]}, "confidence": anchor["confidence"]},
         "club": {"name": club, "confidence": anchor["confidence"]}, "clubhead": None,
         "ball": None, "phase": "contact", "contact": {"x": contact_pixel["x"], "y": contact_pixel["y"],
         "confidence": contact["confidence"], "method": "human_annotation"}, "landing": None,
         "intended_direction": None, "warnings": ["ball_missing"]},
        {"frame_index": landing["frame_index"], "timestamp_seconds": landing["timestamp_seconds"],
         "golfer": {"bbox": {"x": anchor["x"], "y": anchor["y"], "width": 1.0, "height": 1.0},
                    "anchor": {"x": anchor["x"], "y": anchor["y"]}, "confidence": anchor["confidence"]},
         "club": {"name": club, "confidence": anchor["confidence"]}, "clubhead": None,
         "ball": None, "phase": "landing", "contact": None, "landing": {"x": landing["x"],
         "y": landing["y"], "confidence": landing["confidence"], "method": "human_annotation"},
         "intended_direction": None, "warnings": ["ball_missing"]},
    ]
    try:
        return VideoObservations.from_dict({"schema_version": "video-observations.v1",
            "image": {"width": video["width"], "height": video["height"]}, "observations": items})
    except (ValueError, KeyError) as exc:
        raise VideoContractError(f"human evidence cannot form ordered observations: {exc}") from exc


def observations_from_human_annotations(document: HumanAnnotationDocument) -> VideoObservations:
    """Return validated pixel observations for deterministic artifact rendering."""
    if not isinstance(document, HumanAnnotationDocument):
        raise VideoContractError("human annotation document is required")
    return _observations(document)


def import_human_annotations(payload: Any, calibration, *, event_id: str, player_id: str,
                             tournament_id: str, hole_number: int, shot_number: int,
                             distance_to_pin: float, wind: dict, timestamp: str,
                             target_pixel: Any) -> ReconstructionResult:
    """Validate and reconstruct one submitted document through the existing seam."""
    document = payload if isinstance(payload, HumanAnnotationDocument) else HumanAnnotationDocument.from_dict(payload)
    if document.status != "submitted" or not document.payload["explicit_submit"]:
        raise VideoContractError("submitted status and explicit_submit=true are required")
    if document.payload["contact"]["source"] not in {"user_supplied", "user_confirmed", "observed"}:
        raise VideoContractError("contact evidence must be explicit and non-inferred")
    if document.payload["landing"]["source"] not in {"user_supplied", "user_confirmed", "observed"}:
        raise VideoContractError("landing evidence must be explicit and non-inferred")
    if calibration is None:
        raise VideoContractError("calibration is required")
    video = document.payload["video"]
    if (video["width"], video["height"]) != (calibration.width, calibration.height):
        raise VideoContractError("annotation video dimensions do not match calibration")
    if hasattr(calibration, "source_points") and hasattr(calibration, "engine_points"):
        annotation_sources = tuple((point["x"], point["y"]) for point in document.payload["calibration_points"])
        annotation_engine = tuple((point["x"], point["y"]) for point in document.payload["engine_points"])
        loaded_sources = tuple((point.x, point.y) for point in calibration.source_points)
        loaded_engine = tuple((point.x, point.y) for point in calibration.engine_points)
        if annotation_sources != loaded_sources or annotation_engine != loaded_engine:
            raise VideoContractError("annotation calibration points do not match calibration resource")
    observations = _observations(document)
    context_value = _value(document, "context")
    result = reconstruct_shot(observations, calibration, ShotContext(
        event_id=event_id, player_id=player_id, tournament_id=tournament_id,
        hole_number=hole_number, shot_number=shot_number, lie=context_value["lie"],
        club=_value(document, "club_selection"), distance_to_pin=distance_to_pin,
        wind=wind, timestamp=timestamp, target_pixel=target_pixel,
    ))
    metadata = dict(result.metadata)
    metadata.update({"source": "video-human-annotations.v1", "status": document.status,
        "explicit_submit": True, "video": {"width": video["width"], "height": video["height"],
        "frame_count": video["frame_count"], "duration_seconds": video["duration_seconds"]},
        "provenance": {field: {"source": document.payload[field]["source"],
                               "frame_index": document.payload[field]["value"].get("frame_index") if isinstance(document.payload[field]["value"], dict) else None,
                               "timestamp_seconds": document.payload[field]["value"].get("timestamp_seconds") if isinstance(document.payload[field]["value"], dict) else None,
                               "confidence": document.payload[field]["value"].get("confidence") if isinstance(document.payload[field]["value"], dict) else None}
                      for field in ("golfer_anchor", "contact", "landing")}})
    return ReconstructionResult(result.shot_event, metadata)


def load_human_annotations(resource, project_boundary: ProjectBoundary) -> HumanAnnotationDocument:
    if not isinstance(project_boundary, ProjectBoundary):
        raise VideoPathError("a ProjectBoundary is required for annotation resources")
    path = project_boundary.resolve_annotation(resource)
    try:
        return HumanAnnotationDocument.from_json(path.read_text())
    except (OSError, UnicodeError) as exc:
        raise VideoContractError(f"unable to load annotation JSON: {exc}") from exc


__all__ = ["import_human_annotations", "load_human_annotations", "observations_from_human_annotations"]
