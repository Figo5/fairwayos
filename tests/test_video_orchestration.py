import copy
import unittest
from pathlib import Path

from ghostcaddie.adapters.json_file import JsonCourseDataSource, JsonPlayerProfileSource
from ghostcaddie.config import Config
from ghostcaddie.pipeline import run_pipeline
from ghostcaddie.session import InMemoryCourseSource, InMemoryPlayerSource, InMemoryShotSource
from ghostcaddie.video.errors import VideoReconstructionUnavailable
from ghostcaddie.video.observations import VideoObservations
from ghostcaddie.video.orchestration import VideoPipelineResult, run_video_pipeline
from ghostcaddie.video.reconstruction import reconstruct_shot
from tests.test_video_reconstruction import PAYLOAD, calibration, context

DATA = Path(__file__).resolve().parent.parent / "data"


class TestVideoOrchestration(unittest.TestCase):
    def test_fixture_video_runs_unchanged_pipeline_and_returns_outputs(self):
        observations = VideoObservations.from_dict(PAYLOAD)
        result = run_video_pipeline(
            observations,
            calibration(),
            context(),
            JsonCourseDataSource(DATA / "sample_hole.json"),
            JsonPlayerProfileSource(DATA / "sample_player.json"),
            Config.default(),
        )
        self.assertIsInstance(result, VideoPipelineResult)
        self.assertTrue(result.pipeline_result.recommendation.recommended_club)
        self.assertIn("<svg", result.pipeline_result.svg)
        self.assertEqual(result.reconstruction.metadata["source"], "video-fixture")
        self.assertEqual(result.pipeline_result.shot.event_id, "E1")
        self.assertNotIn(str(DATA), repr(result.recommendation.provenance))

    def test_reconstructed_event_has_equivalent_analytics_to_direct_pipeline(self):
        observations = VideoObservations.from_dict(PAYLOAD)
        reconstruction = reconstruct_shot(observations, calibration(), context())
        course_source = JsonCourseDataSource(DATA / "sample_hole.json")
        player_source = JsonPlayerProfileSource(DATA / "sample_player.json")
        direct = run_pipeline(
            InMemoryShotSource(reconstruction.shot_event, "direct:inline:shot"),
            InMemoryCourseSource(course_source.load_course(), "direct:inline:course"),
            InMemoryPlayerSource(player_source.load_player(), "direct:inline:player"),
            Config.default(),
        )
        video = run_video_pipeline(observations, calibration(), context(), course_source, player_source, Config.default())
        self.assertEqual(video.recommendation.recommended_club, direct.recommendation.recommended_club)
        self.assertEqual(video.recommendation.recommended_target, direct.recommendation.recommended_target)
        self.assertEqual(video.recommendation.expected_strokes, direct.recommendation.expected_strokes)
        self.assertEqual(video.recommendation.actual_expected_strokes, direct.recommendation.actual_expected_strokes)
        self.assertEqual(video.recommendation.decision_cost, direct.recommendation.decision_cost)
        self.assertEqual(video.recommendation.hazard_probabilities, direct.recommendation.hazard_probabilities)

    def test_pipeline_is_invoked_once(self):
        import ghostcaddie.video.orchestration as orchestration
        calls = []
        original = orchestration.run_pipeline
        orchestration.run_pipeline = lambda *args: (calls.append(args) or original(*args))
        try:
            run_video_pipeline(VideoObservations.from_dict(PAYLOAD), calibration(), context(),
                               JsonCourseDataSource(DATA / "sample_hole.json"),
                               JsonPlayerProfileSource(DATA / "sample_player.json"), Config.default())
        finally:
            orchestration.run_pipeline = original
        self.assertEqual(len(calls), 1)

    def test_mapping_remains_exactly_once(self):
        class Spy:
            width, height = 1920, 1080
            def __init__(self): self.calls = []
            def to_engine(self, point):
                from ghostcaddie.geometry import Point2D
                self.calls.append(point)
                return Point2D(point.x / 10, point.y / 10)
        spy = Spy()
        run_video_pipeline(VideoObservations.from_dict(PAYLOAD), spy, context(),
                           JsonCourseDataSource(DATA / "sample_hole.json"),
                           JsonPlayerProfileSource(DATA / "sample_player.json"), Config.default())
        self.assertEqual(len(spy.calls), 3)

    def test_unavailable_video_evidence_is_gated_before_pipeline(self):
        payload = copy.deepcopy(PAYLOAD)
        payload["observations"][1]["landing"] = None
        observations = VideoObservations.from_dict(payload)
        with self.assertRaises(VideoReconstructionUnavailable):
            run_video_pipeline(observations, calibration(), context(),
                               JsonCourseDataSource(DATA / "sample_hole.json"),
                               JsonPlayerProfileSource(DATA / "sample_player.json"), Config.default())

    def test_fixed_seed_repeated_runs_have_identical_analytics(self):
        args = (VideoObservations.from_dict(PAYLOAD), calibration(), context(),
                JsonCourseDataSource(DATA / "sample_hole.json"),
                JsonPlayerProfileSource(DATA / "sample_player.json"), Config.default())
        first, second = run_video_pipeline(*args), run_video_pipeline(*args)
        self.assertEqual(first.recommendation.recommended_club, second.recommendation.recommended_club)
        self.assertEqual(first.recommendation.recommended_target, second.recommendation.recommended_target)
        self.assertEqual(first.recommendation.expected_strokes, second.recommendation.expected_strokes)
        self.assertEqual(first.recommendation.actual_expected_strokes, second.recommendation.actual_expected_strokes)
        self.assertEqual(first.recommendation.decision_cost, second.recommendation.decision_cost)
        self.assertEqual(first.recommendation.hazard_probabilities, second.recommendation.hazard_probabilities)
        self.assertEqual(first.svg, second.svg)


if __name__ == "__main__":
    unittest.main()
