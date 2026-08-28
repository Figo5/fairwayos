"""Expected-strokes model: how many strokes does a landing from here cost?"""

from typing import Protocol

from .config import ExpectedStrokesConfig
from .models import RegionType


class ExpectedStrokesModel(Protocol):
    def expected_strokes_for_landing(
        self,
        region: RegionType,
        distance_to_pin_yd: float,
        original_distance_to_pin_yd: float,
    ) -> float: ...


class BaselineTourExpectedStrokesModel:
    """Flat-banded baseline derived from a hand-authored config table.

    Deliberately NOT interpolated: interpolating a hand-written 6-band table
    would imply a false precision the numbers don't have. The last band is a
    terminal value reused for any distance beyond it.
    """

    def __init__(self, config: ExpectedStrokesConfig):
        self.config = config

    def strokes_from_lie(self, region: RegionType, distance_yd: float) -> float:
        bands = self.config.baseline_table[region]
        value = bands[-1][1]
        for max_yards, strokes in bands:
            if distance_yd <= max_yards:
                value = strokes
                break
        return value

    def expected_strokes_for_landing(
        self,
        region: RegionType,
        distance_to_pin_yd: float,
        original_distance_to_pin_yd: float,
    ) -> float:
        if region == RegionType.WATER:
            # Drop at the hazard edge, treated as rough-equivalent from there.
            return self.config.water_penalty_strokes + self.strokes_from_lie(
                RegionType.ROUGH, distance_to_pin_yd
            )
        if region == RegionType.OUT_OF_BOUNDS:
            # Stroke-and-distance: replay from the ORIGINAL distance, not the
            # useless OB landing spot.
            return self.config.ob_penalty_strokes + self.strokes_from_lie(
                RegionType.FAIRWAY, original_distance_to_pin_yd
            )
        return self.strokes_from_lie(region, distance_to_pin_yd)
