"""Fixture-only orchestration at the video-to-engine boundary.

The existing analytics pipeline remains the sole implementation of simulation,
decision, explanation, and rendering.  This module only reconstructs one event
and supplies it through protocol-compatible in-memory sources.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..config import Config
from ..pipeline import PipelineResult, run_pipeline
from ..session import InMemoryCourseSource, InMemoryPlayerSource, InMemoryShotSource
from .calibration import VideoCalibration
from .observations import VideoObservations
from .reconstruction import ReconstructionResult, ShotContext, reconstruct_shot
from .human_import import import_human_annotations


@dataclass(frozen=True)
class VideoPipelineResult:
    """Video sidecar metadata paired with the unchanged pipeline artifact."""

    reconstruction: ReconstructionResult
    pipeline_result: PipelineResult

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.reconstruction.metadata

    @property
    def sidecar_metadata(self) -> Dict[str, Any]:
        return self.reconstruction.metadata

    @property
    def pipeline(self) -> PipelineResult:
        return self.pipeline_result

    @property
    def shot_event(self):
        return self.reconstruction.shot_event

    @property
    def recommendation(self):
        return self.pipeline_result.recommendation

    @property
    def svg(self) -> str:
        return self.pipeline_result.svg

    @property
    def pipeline_status(self) -> str:
        return "complete"

    @property
    def analytics_status(self) -> str:
        return self.pipeline_status


def run_video_pipeline(
    observations: VideoObservations,
    calibration: VideoCalibration,
    context: ShotContext,
    course_source,
    player_source,
    config: Config,
) -> VideoPipelineResult:
    """Run fixture video evidence through exactly one unchanged pipeline call.

    ``course_source`` and ``player_source`` are existing protocol-compatible
    sources. Their values are loaded once and wrapped with descriptive inline
    identifiers so filesystem paths cannot enter recommendation provenance.
    """
    reconstruction = reconstruct_shot(observations, calibration, context)
    event = reconstruction.shot_event
    # ShotEvent intentionally has no video fields.  This dynamic, optional
    # provenance attribute is consumed by the existing pipeline only as a
    # provider provenance sidecar and never affects analytics.
    event.provenance = {
        "source": "video-fixture",
        "video": dict(reconstruction.metadata),
    }

    course = course_source.load_course()
    player = player_source.load_player()
    result = run_pipeline(
        InMemoryShotSource(event, "video:inline:shot"),
        InMemoryCourseSource(course, "video:inline:course"),
        InMemoryPlayerSource(player, "video:inline:player"),
        config,
    )
    return VideoPipelineResult(reconstruction=reconstruction, pipeline_result=result)


def run_human_video_pipeline(
    document,
    calibration: VideoCalibration,
    course_source,
    player_source,
    config: Config,
    *,
    event_id: str,
    tournament_id: str,
    hole_number: int,
    shot_number: int,
    distance_to_pin: float,
    wind: Dict[str, float],
    timestamp: str,
    target_pixel: Any,
) -> VideoPipelineResult:
    """Import submitted human evidence, then run the unchanged pipeline once."""
    player = player_source.load_player()
    reconstruction = import_human_annotations(
        document, calibration, event_id=event_id, player_id=player.player_id,
        tournament_id=tournament_id, hole_number=hole_number, shot_number=shot_number,
        distance_to_pin=distance_to_pin, wind=wind, timestamp=timestamp,
        target_pixel=target_pixel,
    )
    event = reconstruction.shot_event
    # Human labels may spell the same club differently than resource keys.
    aliases = {"7-iron": "7i", "8-iron": "8i", "9-iron": "9i", "pitching-wedge": "PW"}
    resource_club = aliases.get(event.club)
    if event.club not in player.clubs and resource_club in player.clubs:
        player.clubs[event.club] = player.clubs[resource_club]
    event.provenance = {"source": "video-human-annotations.v1", "video": dict(reconstruction.metadata)}
    course = course_source.load_course()
    result = run_pipeline(
        InMemoryShotSource(event, "video-human:inline:shot"),
        InMemoryCourseSource(course, "video-human:inline:course"),
        InMemoryPlayerSource(player, "video-human:inline:player"),
        config,
    )
    return VideoPipelineResult(reconstruction=reconstruction, pipeline_result=result)


# Explicit fixture spelling for callers while fixture mode is the only mode.
run_video_fixture_pipeline = run_video_pipeline
run_fixture_video_pipeline = run_video_pipeline

__all__ = [
    "VideoPipelineResult",
    "run_video_pipeline",
    "run_human_video_pipeline",
    "run_video_fixture_pipeline",
    "run_fixture_video_pipeline",
]
