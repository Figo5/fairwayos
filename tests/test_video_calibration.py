import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.video.calibration import VideoCalibration, load_video_calibration
from ghostcaddie.video.errors import VideoCalibrationError, VideoPathError
from ghostcaddie.video.paths import ProjectBoundary
from ghostcaddie.video.extraction import _source_path


class TestVideoCalibration(unittest.TestCase):
    def _calibration(self):
        return VideoCalibration(
            image_width=1000, image_height=800, source_units="pixels", engine_units="yards",
            source_points=((100, 100), (900, 100), (900, 700), (100, 700)),
            engine_points=((0, 0), (300, 0), (300, 200), (0, 200)),
        )

    def test_valid_mapping_and_inverse_round_trip(self):
        calibration = self._calibration()
        mapped = calibration.to_engine({"x": 500, "y": 400})
        self.assertAlmostEqual(mapped.x, 150.0)
        self.assertAlmostEqual(mapped.y, 100.0)
        source = calibration.from_engine(mapped)
        self.assertAlmostEqual(source["x"], 500.0)
        self.assertAlmostEqual(source["y"], 400.0)

    def test_rejects_invalid_points_and_image_bounds(self):
        base = dict(image_width=1000, image_height=800, source_units="pixels", engine_units="yards",
                    source_points=((100, 100), (900, 100), (900, 700), (100, 700)),
                    engine_points=((0, 0), (300, 0), (300, 200), (0, 200)))
        cases = [
            dict(source_points=((100, 100), (100, 100), (900, 700), (100, 700))),
            dict(source_points=((100, 100), (500, 500), (900, 900), (100, 700))),
            dict(source_points=((100, 100), (900, 100), (900, 700), (math.nan, 700))),
            dict(source_points=((100, 100), (1001, 100), (900, 700), (100, 700))),
        ]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(VideoCalibrationError):
                VideoCalibration(**{**base, **override})

    def test_load_requires_project_bound_relative_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_file = root / "calibration.json"
            calibration_file.write_text(json.dumps({
                "image_width": 1000, "image_height": 800, "source_units": "pixels",
                "engine_units": "yards", "source_points": [{"x": x, "y": y} for x, y in ((100,100),(900,100),(900,700),(100,700))],
                "engine_points": [{"x": x, "y": y} for x, y in ((0,0),(300,0),(300,200),(0,200))],
            }))
            loaded = load_video_calibration("calibration.json", ProjectBoundary(root))
            self.assertEqual(loaded.image_width, 1000)
            with self.assertRaises(VideoPathError):
                load_video_calibration(str(calibration_file), ProjectBoundary(root))


class TestProjectBoundaryAndVideoSource(unittest.TestCase):
    def test_rejects_absolute_traversal_and_symlink_escape_for_project_resources(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root, external = Path(tmp), Path(outside)
            (root / "ok.json").write_text("{}")
            (external / "secret.json").write_text("{}")
            (root / "escape.json").symlink_to(external / "secret.json")
            boundary = ProjectBoundary(root)
            self.assertEqual(boundary.resolve_resource("ok.json"), (root / "ok.json").resolve())
            for value in (str(root / "ok.json"), "../ok.json", "escape.json"):
                with self.subTest(value=value), self.assertRaises(VideoPathError):
                    boundary.resolve_resource(value)

    def test_allows_absolute_regular_readable_video_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "round.mp4"
            source.write_bytes(b"video")
            self.assertEqual(_source_path(str(source)), source.resolve())
            with self.assertRaises(VideoPathError):
                _source_path(str(Path(tmp) / "missing.mp4"))


if __name__ == "__main__":
    unittest.main()
