import copy
import math
import unittest

from ghostcaddie.geometry import Point2D
from ghostcaddie.video.calibration import VideoCalibration
from ghostcaddie.video.errors import VideoReconstructionError, VideoReconstructionUnavailable
from ghostcaddie.video.observations import VideoObservations
from ghostcaddie.video.reconstruction import ShotContext, reconstruct_shot


PAYLOAD = {
    "schema_version": "video-observations.v1",
    "image": {"width": 1920, "height": 1080},
    "observations": [
        {"frame_index": 0, "timestamp_seconds": 0.0,
         "golfer": {"bbox": {"x": 100, "y": 100, "width": 200, "height": 400}, "anchor": {"x": 200, "y": 500}, "confidence": 0.97},
         "club": None, "clubhead": None, "ball": None, "phase": "address", "contact": None,
         "intended_direction": None, "landing": None, "warnings": ["ball_missing"]},
        {"frame_index": 1, "timestamp_seconds": 0.033,
         "golfer": {"bbox": {"x": 100, "y": 100, "width": 200, "height": 400}, "anchor": {"x": 200, "y": 500}, "confidence": 0.98},
         "club": {"name": "7i", "confidence": 0.9}, "clubhead": {"x": 230, "y": 480, "confidence": 0.9},
         "ball": {"x": 250, "y": 520, "confidence": 0.9}, "phase": "impact",
         "contact": {"x": 250, "y": 520, "confidence": 0.9, "method": "fixture"},
         "intended_direction": {"x": 1, "y": 0, "confidence": 0.9},
         "landing": {"x": 900, "y": 500, "confidence": 0.9, "method": "fixture"}, "warnings": []},
    ],
}


def calibration():
    return VideoCalibration(1920, 1080, "pixels", "yards",
        ((0, 0), (1920, 0), (1920, 1080), (0, 1080)),
        ((0, 0), (192, 0), (192, 108), (0, 108)))


def context(**overrides):
    values = dict(event_id="E1", player_id="P1", tournament_id="T1", hole_number=7,
                  shot_number=2, lie="fairway", club="7i", distance_to_pin=150.0,
                  wind={"speed_mph": 8.0, "direction_deg": 90.0}, timestamp="2026-08-27T12:00:00Z",
                  target_pixel=Point2D(700, 500))
    values.update(overrides)
    return ShotContext(**values)


class TestVideoReconstruction(unittest.TestCase):
    def setUp(self):
        self.observations = VideoObservations.from_dict(PAYLOAD)

    def test_normalizes_valid_observations_into_existing_shot_event(self):
        result = reconstruct_shot(self.observations, calibration(), context())
        event = result.shot_event
        self.assertEqual(event.event_id, "E1")
        self.assertEqual(event.start_position, Point2D(20, 50))
        self.assertEqual(event.target_position, Point2D(70, 50))
        self.assertEqual(event.actual_landing_position, Point2D(90, 50))
        self.assertEqual(event.club, "7i")
        self.assertEqual(result.metadata["address_frame_index"], 0)
        self.assertEqual(result.metadata["contact_frame_index"], 1)
        self.assertEqual(result.metadata["landing_frame_index"], 1)
        self.assertEqual(result.metadata["timestamp"], "2026-08-27T12:00:00Z")

    def test_maps_each_source_pixel_exactly_once(self):
        class Spy:
            def __init__(self): self.calls = []
            def to_engine(self, point):
                self.calls.append(point)
                return Point2D(point.x / 10, point.y / 10)
        spy = Spy()
        result = reconstruct_shot(self.observations, spy, context())
        self.assertIsNotNone(result.shot_event)
        self.assertEqual(spy.calls, [Point2D(200, 500), Point2D(700, 500), Point2D(900, 500)])

    def test_missing_landing_or_club_or_contact_is_structured_unavailable(self):
        for field, change in (("landing", lambda item: item.__setitem__("landing", None)),
                              ("contact", lambda item: item.__setitem__("contact", None))):
            payload = copy.deepcopy(PAYLOAD)
            change(payload["observations"][1])
            observations = VideoObservations.from_dict(payload)
            with self.subTest(field=field), self.assertRaises(VideoReconstructionUnavailable):
                reconstruct_shot(observations, calibration(), context())
        with self.assertRaises(VideoReconstructionUnavailable):
            reconstruct_shot(self.observations, calibration(), context(club=None))

    def test_low_confidence_is_gated(self):
        payload = copy.deepcopy(PAYLOAD)
        payload["observations"][1]["landing"]["confidence"] = 0.49
        payload["observations"][1]["warnings"] = ["low_confidence"]
        observations = VideoObservations.from_dict(payload)
        with self.assertRaises(VideoReconstructionUnavailable):
            reconstruct_shot(observations, calibration(), context())

    def test_rejects_non_finite_context_and_invalid_context(self):
        with self.assertRaises(VideoReconstructionError):
            reconstruct_shot(self.observations, calibration(), context(distance_to_pin=math.inf))
        with self.assertRaises(VideoReconstructionError):
            reconstruct_shot(self.observations, calibration(), context(target_pixel=(math.nan, 2)))
        with self.assertRaises(VideoReconstructionError):
            reconstruct_shot(self.observations, calibration(), context(wind={"speed_mph": -1, "direction_deg": 0}))

    def test_repeated_reconstruction_is_deterministic(self):
        first = reconstruct_shot(self.observations, calibration(), context())
        second = reconstruct_shot(self.observations, calibration(), context())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
