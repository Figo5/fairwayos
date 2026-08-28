"""Synthetic TrackMan-shaped carry/side-offset shot adapter."""

import json
from pathlib import Path

from ..geometry import CoordinateMapper, Point2D
from ..models import ShotEvent
from .provider import (ProviderAdapterError, context_point, envelope_metadata,
                       ensure_context, map_point, number, positive, require_dict, require_str,
                       integer, provenance)

SCHEMA_VERSION = "trackman.v1"
_ALLOWED = {"provider", "schema_version", "source_record_id", "event_id", "player_id",
            "tournament_id", "hole_number", "shot_number", "timestamp", "lie", "club",
            "distance_to_pin", "wind", "metrics", "units"}
_REQUIRED = _ALLOWED - {"lie", "distance_to_pin", "wind"}


def adapt_trackman(raw: dict, coordinate_mapper: CoordinateMapper, course_context: dict,
                   strict: bool = False) -> ShotEvent:
    ensure_context(course_context, raw.get("player_id"), "TrackMan")
    diag = envelope_metadata(raw, "trackman", SCHEMA_VERSION, _REQUIRED, _ALLOWED, strict)
    if raw.get("units") != "yards":
        raise ProviderAdapterError("TrackMan units must be 'yards'")
    metrics = require_dict(raw.get("metrics"), "metrics")
    carry = positive(metrics.get("carry_yd"), "metrics.carry_yd")
    side = number(metrics.get("side_offset_yd"), "metrics.side_offset_yd")
    start = context_point(course_context, "start_position")
    aim = context_point(course_context, "aim_position")
    dx, dy = aim.x - start.x, aim.y - start.y
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0:
        raise ProviderAdapterError("course_context start_position and aim_position must differ")
    ux, uy = dx / length, dy / length
    # Signed side offset: positive is right of the aim line.
    landing = Point2D(start.x + ux * carry - uy * side,
                       start.y + uy * carry + ux * side)
    shot = ShotEvent(
        event_id=require_str(raw["event_id"], "event_id"),
        player_id=require_str(raw["player_id"], "player_id"),
        tournament_id=require_str(raw["tournament_id"], "tournament_id"),
        hole_number=integer(raw["hole_number"], "hole_number"),
        shot_number=integer(raw["shot_number"], "shot_number"),
        start_position=map_point(coordinate_mapper, start),
        target_position=map_point(coordinate_mapper, aim),
        actual_landing_position=map_point(coordinate_mapper, landing),
        lie=require_str(raw.get("lie", "fairway"), "lie"),
        club=require_str(raw["club"], "club"),
        distance_to_pin=positive(raw.get("distance_to_pin", carry), "distance_to_pin"),
        wind=require_dict(raw.get("wind", {"speed_mph": 0, "direction_deg": 0}), "wind"),
        timestamp=require_str(raw["timestamp"], "timestamp"),
    )
    shot.provenance = provenance(diag, "trackman-carry-side-v1")
    shot.provenance["units"] = "yards"
    shot.provenance["side_offset_convention"] = "+right_of_aim_line"
    shot.provenance["metrics"] = dict(metrics)
    return shot


class TrackManDataSource:
    def __init__(self, payload: dict, coordinate_mapper: CoordinateMapper,
                 course_context: dict, strict: bool = False):
        self.payload, self.mapper, self.course_context, self.strict = payload, coordinate_mapper, course_context, strict
        self.path = "provider:trackman:" + str(payload.get("source_record_id", "unknown"))

    def load_shot(self) -> ShotEvent:
        return adapt_trackman(self.payload, self.mapper, self.course_context, self.strict)


def load_trackman_json(path: Path, coordinate_mapper, course_context, strict=False):
    with open(path) as fh:
        return TrackManDataSource(json.load(fh), coordinate_mapper, course_context, strict).load_shot()
