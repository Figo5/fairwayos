"""Fixture-only reconstruction of one validated video shot into ``ShotEvent``.

This module is deliberately an adapter boundary: it does not alter the core
models or pipeline and keeps video provenance beside, rather than inside, the
existing event model.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..geometry import Point2D
from ..models import ShotEvent
from .errors import VideoReconstructionError, VideoReconstructionUnavailable
from .observations import VideoObservations


def _point(value: Any, name: str) -> Point2D:
    if isinstance(value, Point2D):
        x, y = value.x, value.y
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        x, y = value
    elif isinstance(value, dict) and {"x", "y"}.issubset(value):
        x, y = value["x"], value["y"]
    else:
        raise VideoReconstructionError(f"{name} must be a 2D point")
    if (isinstance(x, bool) or not isinstance(x, (int, float)) or
            isinstance(y, bool) or not isinstance(y, (int, float)) or
            not math.isfinite(x) or not math.isfinite(y)):
        raise VideoReconstructionError(f"{name} must contain finite coordinates")
    return Point2D(float(x), float(y))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VideoReconstructionError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class ShotContext:
    """Explicit non-video context required to create a ``ShotEvent``."""

    event_id: str
    player_id: str
    tournament_id: str
    hole_number: int
    shot_number: int
    lie: str
    club: Optional[str]
    distance_to_pin: float
    wind: Dict[str, float]
    timestamp: str
    target_pixel: Any
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        for name in ("event_id", "player_id", "tournament_id", "lie", "timestamp"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise VideoReconstructionError(f"{name} must be a non-empty string")
        for name in ("hole_number", "shot_number"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VideoReconstructionError(f"{name} must be a positive integer")
        if self.club is not None and (not isinstance(self.club, str) or not self.club.strip()):
            raise VideoReconstructionError("club must be a non-empty string when supplied")
        distance = _finite(self.distance_to_pin, "distance_to_pin")
        if distance <= 0:
            raise VideoReconstructionError("distance_to_pin must be positive")
        threshold = _finite(self.min_confidence, "min_confidence")
        if not 0 <= threshold <= 1:
            raise VideoReconstructionError("min_confidence must be between 0 and 1")
        if not isinstance(self.wind, dict):
            raise VideoReconstructionError("wind must be a dict")
        try:
            speed = _finite(self.wind["speed_mph"], "wind.speed_mph")
            _finite(self.wind["direction_deg"], "wind.direction_deg")
        except KeyError as exc:
            raise VideoReconstructionError(f"wind missing {exc.args[0]}") from exc
        if speed < 0:
            raise VideoReconstructionError("wind.speed_mph must be non-negative")
        object.__setattr__(self, "target_pixel", _point(self.target_pixel, "target_pixel"))


@dataclass(frozen=True)
class ReconstructionResult:
    """A core event plus sidecar video provenance (never fields on ``ShotEvent``)."""

    shot_event: ShotEvent
    metadata: Dict[str, Any]

    @property
    def event(self) -> ShotEvent:
        return self.shot_event


def _required(observations: VideoObservations, context: ShotContext):
    address = next((item for item in observations.items if item.phase == "address"), None)
    contact = next((item for item in observations.items if item.contact is not None), None)
    landing = next((item for item in observations.items if item.landing is not None), None)
    if address is None:
        raise VideoReconstructionUnavailable("address golfer anchor is unavailable")
    if contact is None:
        raise VideoReconstructionUnavailable("contact pixel/frame is unavailable")
    if landing is None:
        raise VideoReconstructionUnavailable("landing pixel/frame is unavailable")
    threshold = context.min_confidence
    if address.golfer.confidence < threshold:
        raise VideoReconstructionUnavailable("address anchor confidence is below threshold")
    if contact.contact["confidence"] < threshold:
        raise VideoReconstructionUnavailable("contact confidence is below threshold")
    if landing.landing["confidence"] < threshold:
        raise VideoReconstructionUnavailable("landing confidence is below threshold")
    if context.club is None:
        raise VideoReconstructionUnavailable("club context is unavailable")
    return address, contact, landing


def reconstruct_shot(observations: VideoObservations, calibration, context: ShotContext) -> ReconstructionResult:
    """Reconstruct exactly one ``ShotEvent`` from validated fixture observations."""
    if not isinstance(observations, VideoObservations):
        raise VideoReconstructionError("observations must be validated VideoObservations")
    if not isinstance(context, ShotContext):
        raise VideoReconstructionError("context must be ShotContext")
    if hasattr(calibration, "width") and hasattr(calibration, "height"):
        if observations.image_width != calibration.width or observations.image_height != calibration.height:
            raise VideoReconstructionError("observation image dimensions do not match calibration")
    if not (0 <= context.target_pixel.x <= observations.image_width and
            0 <= context.target_pixel.y <= observations.image_height):
        raise VideoReconstructionError("target_pixel is outside image bounds")
    address, contact, landing = _required(observations, context)

    # The only source->engine conversion calls. Contact is provenance only and
    # is intentionally not mapped into an event position.
    origin = calibration.to_engine(_point(address.golfer.anchor, "address anchor"))
    target = calibration.to_engine(context.target_pixel)
    actual = calibration.to_engine(_point(landing.landing, "landing"))
    if not all(math.isfinite(value) for point in (origin, target, actual) for value in (point.x, point.y)):
        raise VideoReconstructionError("calibration produced non-finite engine coordinates")
    event = ShotEvent(
        event_id=context.event_id, player_id=context.player_id, tournament_id=context.tournament_id,
        hole_number=context.hole_number, shot_number=context.shot_number,
        start_position=origin, target_position=target, actual_landing_position=actual,
        lie=context.lie, club=context.club, distance_to_pin=context.distance_to_pin,
        wind=dict(context.wind), timestamp=context.timestamp,
    )
    confidence = min(address.golfer.confidence, contact.contact["confidence"], landing.landing["confidence"])
    metadata = {
        "source": "video-fixture",
        "video_confidence": confidence,
        "frame_provenance": {
            "address_frame_index": address.frame_index,
            "contact_frame_index": contact.frame_index,
            "landing_frame_index": landing.frame_index,
        },
        "address_frame_index": address.frame_index,
        "contact_frame_index": contact.frame_index,
        "landing_frame_index": landing.frame_index,
        "timestamp": context.timestamp,
        "contact_timestamp_seconds": contact.timestamp_seconds,
        "landing_timestamp_seconds": landing.timestamp_seconds,
        "observations_schema_version": observations.schema_version,
    }
    return ReconstructionResult(event, metadata)


normalize_shot = reconstruct_shot
reconstruct_video_shot = reconstruct_shot
