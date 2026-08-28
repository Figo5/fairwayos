"""Monte Carlo shot simulation over candidate aim points."""

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .dispersion import DispersionModel
from .expected_strokes import ExpectedStrokesModel
from .geometry import Point2D, bearing_deg, distance
from .hazards import classify_landing
from .models import CourseModel, PlayerProfile, RegionType, ShotEvent


@dataclass(frozen=True)
class Candidate:
    label: str
    club: str
    aim_point: Point2D


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    expected_strokes: float
    hazard_probabilities: Dict[RegionType, float]
    sample_count: int


class ShotSimulator:
    def __init__(
        self,
        dispersion: DispersionModel,
        strokes_model: ExpectedStrokesModel,
        course: CourseModel,
        config: SimulationConfig,
    ):
        self.dispersion = dispersion
        self.strokes_model = strokes_model
        self.course = course
        self.config = config

    def generate_candidates(self, shot: ShotEvent, player: PlayerProfile) -> List[Candidate]:
        """~4 clubs x 3 lateral aim offsets = ~12 candidates per shot.

        Aim depth: if the club's carry is within tolerance of the pin
        distance, aim at the pin depth; otherwise lay up at carry distance
        along the start->pin line (short or long of the pin).
        """
        bearing = bearing_deg(shot.start_position, shot.target_position)
        candidates: List[Candidate] = []
        for club_profile in player.clubs.values():
            if abs(club_profile.carry_mean_yd - shot.distance_to_pin) <= self.config.club_distance_tolerance_yd:
                depth = shot.distance_to_pin
                depth_label = "pin"
            else:
                depth = club_profile.carry_mean_yd
                depth_label = f"layup{int(depth)}"
            for offset in self.config.candidate_aim_offsets_yd:
                aim = self._point_on_line_plus_offset(
                    shot.start_position, bearing, depth, offset
                )
                label = f"{club_profile.club}_{depth_label}_{offset:+.0f}yd"
                candidates.append(Candidate(label=label, club=club_profile.club, aim_point=aim))
        return candidates

    @staticmethod
    def _point_on_line_plus_offset(
        start: Point2D, bearing_deg_: float, depth_yd: float, offset_yd: float
    ) -> Point2D:
        """Point `depth` yards along the bearing, then `offset` yards perpendicular."""
        b = math.radians(bearing_deg_)
        x = start.x + depth_yd * math.cos(b) - offset_yd * math.sin(b)
        y = start.y + depth_yd * math.sin(b) + offset_yd * math.cos(b)
        return Point2D(x, y)

    def evaluate_candidate(
        self,
        shot: ShotEvent,
        player: PlayerProfile,
        candidate: Candidate,
        rng: random.Random,
    ) -> CandidateResult:
        club = player.clubs[candidate.club]
        lie_mod = player.lie_modifiers.get(shot.lie)
        scores: List[float] = []
        region_counts: Dict[RegionType, int] = {}
        n = self.config.monte_carlo_samples
        for _ in range(n):
            landing = self.dispersion.sample_landing(
                shot.start_position, candidate.aim_point, club, lie_mod, rng,
                wind=shot.wind,
            )
            region = classify_landing(landing, self.course)
            region_counts[region] = region_counts.get(region, 0) + 1
            scores.append(
                self.strokes_model.expected_strokes_for_landing(
                    region,
                    distance(landing, self.course.pin_position),
                    shot.distance_to_pin,
                )
            )
        expected_strokes = statistics.mean(scores) + 1.0  # the shot itself
        hazard_probabilities = {
            region: round(count / n / 0.05) * 0.05
            for region, count in region_counts.items()
        }
        return CandidateResult(
            candidate=candidate,
            expected_strokes=expected_strokes,
            hazard_probabilities=hazard_probabilities,
            sample_count=n,
        )

    def run(self, shot: ShotEvent, player: PlayerProfile, rng: random.Random) -> List[CandidateResult]:
        """Evaluate every candidate in order, sharing ONE rng sequentially.

        Deterministic for a fixed seed: candidate order is fixed by
        generate_candidates, and each evaluation draws from the same
        sequential rng stream.
        """
        candidates = self.generate_candidates(shot, player)
        return [
            self.evaluate_candidate(shot, player, candidate, rng)
            for candidate in candidates
        ]
