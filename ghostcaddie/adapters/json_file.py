"""Concrete JSON-file adapters for the three data-source Protocols.

These depend ONLY on models.py and geometry.py — the analytics core
(simulation/decision/hazards/dispersion/expected_strokes) is never imported
here, so the engine doesn't care where data came from.

The pure `_parse_*` helpers are the reusable JSON-to-domain boundary: they
validate dictionaries and required fields BEFORE float/int conversion, reject
booleans, non-finite numbers, non-integral ordinals, and malformed nested
polygons/wind, and accept an explicit CoordinateMapper boundary so session
parsing can map each inline position exactly once while the JSON adapters keep
their existing one-time mapping.
"""

import json
import math
from pathlib import Path
from typing import Dict, List

from ..geometry import CoordinateMapper, CoordinateSystem, Point2D
from ..models import (
    ClubProfile,
    CourseModel,
    LiePerformanceModifier,
    PlayerProfile,
    ShotEvent,
)


def _require_dict(raw, name: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a dict, got {type(raw).__name__}")


def _require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
    try:
        value = float(value)
    except OverflowError:
        # int too large for float conversion: reject under the documented
        # ValueError boundary rather than leaking OverflowError.
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_finite(value, name: str) -> float:
    v = _finite_number(value, name)
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v


def _positive_integral(value, name: str) -> int:
    v = _positive_finite(value, name)
    if v != int(v):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    return int(v)


def _non_negative_integral(value, name: str) -> int:
    v = _finite_number(value, name)
    if v < 0 or v != int(v):
        raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
    return int(v)


def _point_dict(raw, name: str) -> dict:
    """Validate a raw {"x": ..., "y": ...} dict and return it as plain floats."""
    _require_dict(raw, name)
    return {
        "x": _finite_number(raw.get("x"), f"{name}.x"),
        "y": _finite_number(raw.get("y"), f"{name}.y"),
    }


def _parse_point(raw, name: str = "point") -> Point2D:
    d = _point_dict(raw, name)
    return Point2D(d["x"], d["y"])


def _parse_point_list(raw_list, name: str) -> tuple:
    if not isinstance(raw_list, list):
        raise ValueError(f"{name} must be a list, got {type(raw_list).__name__}")
    return tuple(_parse_point(p, f"{name}[{i}]") for i, p in enumerate(raw_list))


def _parse_polygons(raw_list, name: str = "polygon") -> List[List[Point2D]]:
    if not isinstance(raw_list, list):
        raise ValueError(f"{name} must be a list, got {type(raw_list).__name__}")
    return [
        [_parse_point(p, f"{name}[{i}][{j}]") for j, p in enumerate(poly)]
        for i, poly in enumerate(raw_list)
    ]


def _parse_coordinate_system(raw) -> CoordinateSystem:
    _require_dict(raw, "coordinate_system")
    mode = raw.get("mode")
    if mode not in ("manual", "four_point"):
        raise ValueError(
            f"coordinate_system.mode must be 'manual' or 'four_point', got {mode!r}"
        )
    units = raw.get("units", "yards")
    if not isinstance(units, str) or not units:
        raise ValueError("coordinate_system.units must be a non-empty string")
    if mode == "four_point":
        return CoordinateSystem(
            mode=mode,
            origin=_parse_point(raw.get("origin", {"x": 0, "y": 0}), "coordinate_system.origin"),
            units=units,
            source_units=raw.get("source_units", "pixels"),
            source_points=_parse_point_list(
                raw.get("source_points"), "coordinate_system.source_points"
            ),
            engine_points=_parse_point_list(
                raw.get("engine_points"), "coordinate_system.engine_points"
            ),
        )
    return CoordinateSystem(
        mode=mode,
        origin=_parse_point(raw.get("origin"), "coordinate_system.origin"),
        units=units,
    )


def _parse_course(raw) -> CourseModel:
    _require_dict(raw, "course")
    return CourseModel(
        name=_require_str(raw.get("name"), "course.name"),
        par=_positive_integral(raw.get("par"), "course.par"),
        coordinate_system=_parse_coordinate_system(raw.get("coordinate_system")),
        fairway=_parse_polygons(raw.get("fairway"), "course.fairway"),
        green=_parse_polygons(raw.get("green"), "course.green"),
        bunkers=_parse_polygons(raw.get("bunkers", []), "course.bunkers"),
        water_hazards=_parse_polygons(raw.get("water_hazards", []), "course.water_hazards"),
        out_of_bounds=_parse_polygons(raw.get("out_of_bounds", []), "course.out_of_bounds"),
        pin_position=_parse_point(raw.get("pin_position"), "course.pin_position"),
        elevation=raw.get("elevation"),
    )


def _parse_wind(raw, name: str = "shot.wind") -> Dict[str, float]:
    _require_dict(raw, name)
    speed = _finite_number(raw.get("speed_mph"), f"{name}.speed_mph")
    direction = _finite_number(raw.get("direction_deg"), f"{name}.direction_deg")
    if speed < 0:
        raise ValueError(f"{name}.speed_mph must be non-negative")
    return {"speed_mph": speed, "direction_deg": direction}


def _parse_shot(raw, mapper: CoordinateMapper) -> ShotEvent:
    _require_dict(raw, "shot")
    return ShotEvent(
        event_id=_require_str(raw.get("event_id"), "shot.event_id"),
        player_id=_require_str(raw.get("player_id"), "shot.player_id"),
        tournament_id=_require_str(raw.get("tournament_id"), "shot.tournament_id"),
        hole_number=_positive_integral(raw.get("hole_number"), "shot.hole_number"),
        shot_number=_positive_integral(raw.get("shot_number"), "shot.shot_number"),
        start_position=mapper.to_engine(_point_dict(raw.get("start_position"), "shot.start_position")),
        target_position=mapper.to_engine(_point_dict(raw.get("target_position"), "shot.target_position")),
        actual_landing_position=mapper.to_engine(
            _point_dict(raw.get("actual_landing_position"), "shot.actual_landing_position")
        ),
        lie=_require_str(raw.get("lie"), "shot.lie"),
        club=_require_str(raw.get("club"), "shot.club"),
        distance_to_pin=_positive_finite(raw.get("distance_to_pin"), "shot.distance_to_pin"),
        wind=_parse_wind(raw.get("wind"), "shot.wind"),
        timestamp=_require_str(raw.get("timestamp"), "shot.timestamp"),
    )


def _parse_player(raw) -> PlayerProfile:
    _require_dict(raw, "player")
    player_id = _require_str(raw.get("player_id"), "player.player_id")
    clubs_raw = raw.get("clubs")
    if not isinstance(clubs_raw, dict) or not clubs_raw:
        raise ValueError("player.clubs must be a non-empty dict")
    clubs: Dict[str, ClubProfile] = {}
    for name, c in clubs_raw.items():
        _require_dict(c, f"player.clubs.{name}")
        clubs[name] = ClubProfile(
            club=name,
            carry_mean_yd=_finite_number(c.get("carry_mean_yd"), f"player.clubs.{name}.carry_mean_yd"),
            carry_stddev_yd=_finite_number(c.get("carry_stddev_yd"), f"player.clubs.{name}.carry_stddev_yd"),
            lateral_stddev_yd=_finite_number(c.get("lateral_stddev_yd"), f"player.clubs.{name}.lateral_stddev_yd"),
            miss_bias_yd=_finite_number(c.get("miss_bias_yd", 0.0), f"player.clubs.{name}.miss_bias_yd"),
            sample_size=_non_negative_integral(c.get("sample_size", 0), f"player.clubs.{name}.sample_size"),
        )
    lie_modifiers: Dict[str, LiePerformanceModifier] = {}
    lie_raw = raw.get("lie_modifiers", {})
    if not isinstance(lie_raw, dict):
        raise ValueError("player.lie_modifiers must be a dict")
    for lie, m in lie_raw.items():
        _require_dict(m, f"player.lie_modifiers.{lie}")
        lie_modifiers[lie] = LiePerformanceModifier(
            carry_multiplier=_finite_number(
                m.get("carry_multiplier", 1.0), f"player.lie_modifiers.{lie}.carry_multiplier"
            ),
            stddev_multiplier=_finite_number(
                m.get("stddev_multiplier", 1.0), f"player.lie_modifiers.{lie}.stddev_multiplier"
            ),
        )
    return PlayerProfile(player_id=player_id, clubs=clubs, lie_modifiers=lie_modifiers)


class JsonCourseDataSource:
    def __init__(self, path: Path):
        self.path = path

    def load_course(self) -> CourseModel:
        with open(self.path) as fh:
            raw = json.load(fh)
        return _parse_course(raw)


class JsonShotDataSource:
    def __init__(self, path: Path, coordinate_system: CoordinateSystem):
        self.path = path
        self.mapper = CoordinateMapper(coordinate_system)

    def load_shot(self) -> ShotEvent:
        with open(self.path) as fh:
            raw = json.load(fh)
        return _parse_shot(raw, self.mapper)


class JsonPlayerProfileSource:
    def __init__(self, path: Path):
        self.path = path

    def load_player(self) -> PlayerProfile:
        with open(self.path) as fh:
            raw = json.load(fh)
        return _parse_player(raw)
