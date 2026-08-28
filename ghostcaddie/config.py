"""All tunable knobs for the engine live here, nowhere else."""

"""All tunable knobs for the engine live here, nowhere else."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .models import RegionType


def _default_baseline_table() -> Dict[RegionType, List[Tuple[float, float]]]:
    return {
        RegionType.FAIRWAY: [(50, 2.5), (100, 2.7), (150, 2.9), (200, 3.1), (250, 3.3), (9999, 3.6)],
        RegionType.ROUGH: [(50, 2.7), (100, 2.9), (150, 3.1), (200, 3.4), (250, 3.7), (9999, 4.0)],
        RegionType.BUNKER: [(50, 2.8), (100, 3.0), (150, 3.3), (9999, 3.7)],
        RegionType.GREEN: [(10, 1.5), (20, 1.8), (40, 2.0), (9999, 2.2)],
    }


@dataclass(frozen=True)
class SimulationConfig:
    monte_carlo_samples: int = 300
    random_seed: int = 42
    candidate_aim_offsets_yd: Tuple[float, ...] = (-15.0, 0.0, 15.0)
    club_distance_tolerance_yd: float = 30.0
    # Wind linear-sensitivity coefficients, yards of landing shift per mph of
    # wind, applied to the wind vector projected onto the shot's strike frame.
    # Illustrative linear sensitivities, NOT physical flight constants (no
    # trajectory/spin/loft/launch modeling — see README limitations).
    along_wind_carry_yd_per_mph: float = 1.5
    crosswind_lateral_drift_yd_per_mph: float = 1.0


@dataclass(frozen=True)
class ExpectedStrokesConfig:
    water_penalty_strokes: float = 1.0
    ob_penalty_strokes: float = 2.0
    # Hand-authored illustrative flat-banded table. Bands are (max_yards, strokes),
    # scanned in order; the last band's terminal value is reused for any distance
    # beyond it. The GREEN terminal value is a coarse cap, explicitly NOT a
    # putting model.
    baseline_table: Dict[RegionType, List[Tuple[float, float]]] = field(
        default_factory=_default_baseline_table
    )


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig = SimulationConfig()
    expected_strokes: ExpectedStrokesConfig = ExpectedStrokesConfig()
    confidence_low_sample_threshold: int = 20
    confidence_medium_sample_threshold: int = 75

    @staticmethod
    def default() -> "Config":
        return Config()
