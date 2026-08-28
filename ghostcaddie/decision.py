"""Decision layer: pick the best candidate and compare against the actual shot."""

import random

from .config import Config
from .explanation import explain
from .models import CourseModel, PlayerProfile, Recommendation, ShotEvent
from .simulation import Candidate, CandidateResult, ShotSimulator


def select_best(results: list) -> CandidateResult:
    """Min-by-expected-strokes. Python's min() is stable, so ties break by
    original list order deliberately — the engine never randomizes a tie."""
    return min(results, key=lambda r: r.expected_strokes)


def evaluate_actual_decision(
    shot: ShotEvent,
    player: PlayerProfile,
    simulator: ShotSimulator,
    rng: random.Random,
) -> CandidateResult:
    """Re-simulate the expectation of the ACTUAL (club, target) decision.

    This deliberately re-runs the same Monte Carlo path on a candidate built
    from what the player chose, rather than looking up the single historical
    actual_landing_position outcome. The actual landing field is kept for
    narrative/SVG display only and is never fed into the expected-strokes
    comparison, keeping the comparison apples-to-apples.
    """
    candidate = Candidate(
        label="actual",
        club=shot.club,
        aim_point=shot.target_position,
    )
    return simulator.evaluate_candidate(shot, player, candidate, rng)


def build_recommendation(
    shot: ShotEvent,
    player: PlayerProfile,
    course: CourseModel,
    best: CandidateResult,
    actual: CandidateResult,
    config: Config,
    provenance: dict,
) -> Recommendation:
    decision_cost = actual.expected_strokes - best.expected_strokes

    min_sample = min(
        player.clubs[best.candidate.club].sample_size,
        player.clubs[actual.candidate.club].sample_size,
    )
    if min_sample < config.confidence_low_sample_threshold:
        confidence = "low"
    elif min_sample < config.confidence_medium_sample_threshold:
        confidence = "medium"
    else:
        confidence = "high"

    hazard_probabilities = {
        region.value: prob for region, prob in best.hazard_probabilities.items()
    }

    recommendation = Recommendation(
        recommended_club=best.candidate.club,
        recommended_target=best.candidate.aim_point,
        expected_strokes=best.expected_strokes,
        actual_expected_strokes=actual.expected_strokes,
        decision_cost=decision_cost,
        hazard_probabilities=hazard_probabilities,
        confidence=confidence,
        explanation="",  # filled by explain() below
        provenance=provenance,
    )
    recommendation.explanation = explain(recommendation, shot, best)
    return recommendation
