"""Dispersion modeling: probabilistic shot landing distributions.

Determinism is CRITICAL here. Every random draw goes through the passed-in
seeded random.Random instance — the global random module is never touched —
so a fixed seed reproduces an identical landing sequence every time.
"""

import math
import random
from typing import Dict, List, Optional, Protocol

from .config import SimulationConfig
from .geometry import Point2D, bearing_deg
from .models import ClubProfile, LiePerformanceModifier


class DispersionModel(Protocol):
    def sample_landing(
        self,
        start: Point2D,
        aim: Point2D,
        club: ClubProfile,
        lie_mod: Optional[LiePerformanceModifier],
        rng: random.Random,
        wind: Optional[Dict[str, float]] = None,
    ) -> Point2D: ...


class GaussianDispersionModel:
    """Normal carry/lateral dispersion, rotated onto the start->aim bearing.

    (along, lateral) are sampled in the club's "strike" frame — along the
    carry axis toward the aim, lateral perpendicular to it — then rotated by
    the bearing from start to aim and added to start to get an engine (x, y).

    Wind (see ShotEvent.wind contract) is projected onto the strike frame as a
    mean shift: the along Gaussian mean gains the wind's along component times
    the along-wind carry coefficient, and the lateral Gaussian mean gains the
    wind's left-lateral component times the crosswind drift coefficient. When
    `wind` is None or its speed is 0 the shifts are exactly zero and the
    landing sequence is bit-for-bit identical to the pre-wind implementation
    for the same seed (same two gauss draws, same rotation).
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

    def sample_landing(
        self,
        start: Point2D,
        aim: Point2D,
        club: ClubProfile,
        lie_mod: Optional[LiePerformanceModifier],
        rng: random.Random,
        wind: Optional[Dict[str, float]] = None,
    ) -> Point2D:
        carry_mult = lie_mod.carry_multiplier if lie_mod else 1.0
        stddev_mult = lie_mod.stddev_multiplier if lie_mod else 1.0
        effective_carry_mean = club.carry_mean_yd * carry_mult
        effective_carry_stddev = club.carry_stddev_yd * stddev_mult
        effective_lateral_stddev = club.lateral_stddev_yd * stddev_mult

        if wind is None or wind.get("speed_mph", 0) == 0:
            along_mean = effective_carry_mean
            lateral_mean = club.miss_bias_yd
        else:
            wind_radians = math.radians(wind["direction_deg"])
            wind_x = wind["speed_mph"] * math.cos(wind_radians)
            wind_y = wind["speed_mph"] * math.sin(wind_radians)
            shot_radians = math.radians(bearing_deg(start, aim))
            along_component_mph = (
                wind_x * math.cos(shot_radians) + wind_y * math.sin(shot_radians)
            )
            lateral_component_mph = (
                -wind_x * math.sin(shot_radians) + wind_y * math.cos(shot_radians)
            )
            along_mean = (
                effective_carry_mean
                + along_component_mph * self.config.along_wind_carry_yd_per_mph
            )
            lateral_mean = (
                club.miss_bias_yd
                + lateral_component_mph * self.config.crosswind_lateral_drift_yd_per_mph
            )

        along = rng.gauss(along_mean, effective_carry_stddev)
        lateral = rng.gauss(lateral_mean, effective_lateral_stddev)

        bearing = math.radians(bearing_deg(start, aim))
        dx = along * math.cos(bearing) - lateral * math.sin(bearing)
        dy = along * math.sin(bearing) + lateral * math.cos(bearing)
        return Point2D(start.x + dx, start.y + dy)

    def sample_many(
        self,
        start: Point2D,
        aim: Point2D,
        club: ClubProfile,
        lie_mod: Optional[LiePerformanceModifier],
        rng: random.Random,
        n: int,
        wind: Optional[Dict[str, float]] = None,
    ) -> List[Point2D]:
        return [
            self.sample_landing(start, aim, club, lie_mod, rng, wind)
            for _ in range(n)
        ]
