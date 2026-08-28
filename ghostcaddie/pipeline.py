"""Pipeline: load -> simulate -> decide -> explain & render.

A pipeline-run artifact (not a domain schema), so PipelineResult lives here.
The literal code order below enforces the architecture rule: analytics
finishes before rendering starts (render_svg is called AFTER
build_recommendation).
"""

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from . import decision, overlay
from .config import Config
from .dispersion import GaussianDispersionModel
from .expected_strokes import BaselineTourExpectedStrokesModel
from .models import CourseModel, PlayerProfile, Recommendation, ShotEvent
from .simulation import CandidateResult, ShotSimulator


@dataclass
class PipelineResult:
    shot: ShotEvent
    course: CourseModel
    player: PlayerProfile
    candidate_results: List[CandidateResult]
    actual_result: CandidateResult
    recommendation: Recommendation
    svg: str


def run_pipeline(shot_source, course_source, player_source, config: Config) -> PipelineResult:
    course = course_source.load_course()
    shot = shot_source.load_shot()
    player = player_source.load_player()

    rng = random.Random(config.simulation.random_seed)
    simulator = ShotSimulator(
        GaussianDispersionModel(config.simulation),
        BaselineTourExpectedStrokesModel(config.expected_strokes),
        course,
        config.simulation,
    )
    candidate_results = simulator.run(shot, player, rng)
    actual_result = decision.evaluate_actual_decision(shot, player, simulator, rng)
    best = decision.select_best(candidate_results)

    # Adapter identifiers come from the composition root (the CLI) via a
    # lightweight attribute — the JSON adapters expose `.path`; a future
    # non-JSON adapter would expose some other identifier. The pipeline only
    # knows these sources structurally, per the adapter Protocol seam.
    provenance = {
        "shot_source": str(getattr(shot_source, "path", type(shot_source).__name__)),
        "course_source": str(getattr(course_source, "path", type(course_source).__name__)),
        "player_source": str(getattr(player_source, "path", type(player_source).__name__)),
        "random_seed": config.simulation.random_seed,
        "monte_carlo_samples": config.simulation.monte_carlo_samples,
        "engine_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_disclaimer": (
            "Synthetic/mock data only. Not sourced from ShotLink, TrackMan, "
            "TOURCAST, or any official PGA TOUR system. Not for competitive "
            "or broadcast use."
        ),
    }
    provider_provenance = getattr(shot, "provenance", None)
    if provider_provenance:
        provenance["provider"] = dict(provider_provenance)
    recommendation = decision.build_recommendation(
        shot, player, course, best, actual_result, config, provenance
    )
    svg = overlay.render_svg(course, shot, candidate_results, recommendation)
    return PipelineResult(
        shot=shot,
        course=course,
        player=player,
        candidate_results=candidate_results,
        actual_result=actual_result,
        recommendation=recommendation,
        svg=svg,
    )
