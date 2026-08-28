"""Synthetic ShotLink-shaped GPS shot adapter."""

import math
from pathlib import Path
import json

from ..geometry import CoordinateMapper, Point2D
from ..models import ShotEvent
from .provider import (ProviderAdapterError, context_point, envelope_metadata,
                       ensure_context, map_point, number, positive, require_dict, require_str,
                       integer, provenance)

SCHEMA_VERSION = "shotlink.v1"
_ALLOWED = {"provider", "schema_version", "source_record_id", "event_id", "player_id",
            "tournament_id", "hole_number", "shot_number", "timestamp", "start_position",
            "target_position", "actual_landing_position", "lie", "club", "distance_to_pin",
            "wind", "geo_frame"}
_REQUIRED = _ALLOWED - {"geo_frame"}


def _gps(raw, name, frame, mapper):
    value = require_dict(raw.get(name), f"{name}")
    lat = number(value.get("latitude"), f"{name}.latitude")
    lon = number(value.get("longitude"), f"{name}.longitude")
    origin = require_dict(frame.get("origin"), "geo_frame.origin")
    lat0 = number(origin.get("latitude"), "geo_frame.origin.latitude")
    lon0 = number(origin.get("longitude"), "geo_frame.origin.longitude")
    # Explicit local frame: +x east, +y north; convert degrees to yards.
    scale = 111320.0 / 0.9144
    local = Point2D((lon - lon0) * math.cos(math.radians(lat0)) * scale,
                    (lat - lat0) * scale)
    return map_point(mapper, local)


def adapt_shotlink(raw: dict, coordinate_mapper: CoordinateMapper, course_context: dict,
                   strict: bool = False) -> ShotEvent:
    ensure_context(course_context, raw.get("player_id"), "ShotLink")
    diag = envelope_metadata(raw, "shotlink", SCHEMA_VERSION, _REQUIRED, _ALLOWED, strict)
    frame = require_dict(raw.get("geo_frame"), "geo_frame")
    if frame.get("units") != "degrees" or frame.get("axes") != "+x east, +y north":
        raise ProviderAdapterError("geo_frame must declare degrees and '+x east, +y north'")
    shot = ShotEvent(
        event_id=require_str(raw["event_id"], "event_id"),
        player_id=require_str(raw["player_id"], "player_id"),
        tournament_id=require_str(raw["tournament_id"], "tournament_id"),
        hole_number=integer(raw["hole_number"], "hole_number"),
        shot_number=integer(raw["shot_number"], "shot_number"),
        start_position=_gps(raw, "start_position", frame, coordinate_mapper),
        target_position=_gps(raw, "target_position", frame, coordinate_mapper),
        actual_landing_position=_gps(raw, "actual_landing_position", frame, coordinate_mapper),
        lie=require_str(raw["lie"], "lie"), club=require_str(raw["club"], "club"),
        distance_to_pin=positive(raw["distance_to_pin"], "distance_to_pin"),
        wind=require_dict(raw["wind"], "wind"), timestamp=require_str(raw["timestamp"], "timestamp"),
    )
    shot.provenance = provenance(diag, "shotlink-gps-v1")
    shot.provenance["geo_frame"] = dict(frame)
    return shot


class ShotLinkDataSource:
    def __init__(self, payload: dict, coordinate_mapper: CoordinateMapper,
                 course_context: dict, strict: bool = False):
        self.payload, self.mapper, self.course_context, self.strict = payload, coordinate_mapper, course_context, strict
        self.path = "provider:shotlink:" + str(payload.get("source_record_id", "unknown"))

    def load_shot(self) -> ShotEvent:
        return adapt_shotlink(self.payload, self.mapper, self.course_context, self.strict)


def load_shotlink_json(path: Path, coordinate_mapper, course_context, strict=False):
    with open(path) as fh:
        return ShotLinkDataSource(json.load(fh), coordinate_mapper, course_context, strict).load_shot()
