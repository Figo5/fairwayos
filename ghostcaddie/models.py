"""Domain schema: profiles, shot events, courses, and the final recommendation."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .geometry import CoordinateSystem, Point2D


class RegionType(str, Enum):
    """Landing region. str-Enum so it serializes as a plain JSON string."""

    FAIRWAY = "fairway"
    ROUGH = "rough"
    GREEN = "green"
    BUNKER = "bunker"
    WATER = "water"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass(frozen=True)
class ClubProfile:
    club: str
    carry_mean_yd: float
    carry_stddev_yd: float
    lateral_stddev_yd: float
    miss_bias_yd: float = 0.0
    sample_size: int = 0


@dataclass(frozen=True)
class LiePerformanceModifier:
    carry_multiplier: float = 1.0
    stddev_multiplier: float = 1.0


@dataclass
class PlayerProfile:
    player_id: str
    clubs: Dict[str, ClubProfile]
    lie_modifiers: Dict[str, LiePerformanceModifier] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.clubs:
            raise ValueError("PlayerProfile requires at least one club")
        for club_name, club in self.clubs.items():
            if club.carry_stddev_yd <= 0 or club.lateral_stddev_yd <= 0:
                raise ValueError(
                    f"Club '{club_name}' has a non-positive stddev "
                    f"(carry={club.carry_stddev_yd}, lateral={club.lateral_stddev_yd}); "
                    "dispersion modeling requires positive spread."
                )


@dataclass
class ShotEvent:
    event_id: str
    player_id: str
    tournament_id: str
    hole_number: int
    shot_number: int
    start_position: Point2D
    target_position: Point2D
    actual_landing_position: Point2D
    lie: str
    club: str
    distance_to_pin: float
    wind: Dict[str, float]
    timestamp: str

    def __post_init__(self) -> None:
        if self.distance_to_pin <= 0:
            raise ValueError("distance_to_pin must be positive")
        if self.hole_number <= 0 or self.shot_number <= 0:
            raise ValueError("hole_number and shot_number must be positive")
        self._validate_wind()

    def _validate_wind(self) -> None:
        """Wind trust-boundary contract (see README "Wind-adjusted dispersion").

        `wind` must carry the two documented numeric keys:
        - `speed_mph`: non-negative mph of the wind vector.
        - `direction_deg`: the direction the wind vector TRAVELS TOWARD in the
          engine frame (top-down yards, 0 deg = +x tee-to-pin, 90 deg = +y,
          angles increasing counterclockwise). So for a straight +x shot, 0 is
          a tailwind, 180 a headwind, and 90/270 are crosswinds toward +/-y.
        Direction is intentionally left UNBOUNDED (e.g. -90 and 270 are both
        valid) so equivalent angles remain usable by sin/cos; speed is checked
        for finiteness and sign.
        """
        if not isinstance(self.wind, dict):
            raise ValueError("wind must be a dict with 'speed_mph' and 'direction_deg'")
        missing = [k for k in ("speed_mph", "direction_deg") if k not in self.wind]
        if missing:
            raise ValueError(f"wind missing required key(s): {', '.join(missing)}")
        speed, direction = self.wind["speed_mph"], self.wind["direction_deg"]
        if not isinstance(speed, (int, float)) or not isinstance(direction, (int, float)):
            raise ValueError("wind 'speed_mph' and 'direction_deg' must be numeric")
        if not math.isfinite(speed) or not math.isfinite(direction):
            raise ValueError("wind 'speed_mph' and 'direction_deg' must be finite")
        if speed < 0:
            raise ValueError("wind 'speed_mph' must be non-negative")


@dataclass
class CourseModel:
    name: str
    par: int
    coordinate_system: CoordinateSystem
    fairway: List[List[Point2D]]
    green: List[List[Point2D]]
    pin_position: Point2D
    bunkers: List[List[Point2D]] = field(default_factory=list)
    water_hazards: List[List[Point2D]] = field(default_factory=list)
    out_of_bounds: List[List[Point2D]] = field(default_factory=list)
    elevation: Optional[Dict[str, float]] = None  # captured when available; unused by milestone-1 math

    def __post_init__(self) -> None:
        all_polygons = (
            self.fairway
            + self.green
            + self.bunkers
            + self.water_hazards
            + self.out_of_bounds
        )
        for i, poly in enumerate(all_polygons):
            if len(poly) < 3:
                raise ValueError(f"Every course polygon needs >= 3 points (found {len(poly)})")


@dataclass
class Recommendation:
    recommended_club: str
    recommended_target: Point2D
    expected_strokes: float
    actual_expected_strokes: float
    decision_cost: float
    hazard_probabilities: Dict[str, float]  # RegionType.value keys for clean JSON
    confidence: str
    explanation: str
    provenance: Dict[str, object]

    def __post_init__(self) -> None:
        if self.confidence not in ("low", "medium", "high"):
            raise ValueError(
                f"confidence must be 'low', 'medium', or 'high', got {self.confidence!r}"
            )
