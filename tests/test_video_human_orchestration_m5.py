import copy
import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ghostcaddie.adapters.json_file import JsonCourseDataSource, JsonPlayerProfileSource
from ghostcaddie.cli import main
from ghostcaddie.config import Config
from ghostcaddie.geometry import Point2D
from ghostcaddie.video.calibration import load_video_calibration
from ghostcaddie.video.human_import import load_human_annotations
from ghostcaddie.video.orchestration import run_human_video_pipeline
from ghostcaddie.video.paths import ProjectBoundary
from tests.test_video_human_contracts import TestHumanAnnotationContract

DATA = Path(__file__).resolve().parent.parent / "data"


class TestHumanVideoOrchestrationM5(unittest.TestCase):
    def setUp(self):
        source = TestHumanAnnotationContract()
        self.payload = source.valid_payload()
        self.payload.update(status="submitted", explicit_submit=True)
        self.payload["contact"] = {"value": {"x": 900.0, "y": 600.0, "frame_index": 20,
            "timestamp_seconds": 0.667, "confidence": 0.9, "phase": "contact"}, "source": "observed"}
        self.payload["landing"] = {"value": {"x": 1500.0, "y": 500.0, "frame_index": 40,
            "timestamp_seconds": 1.345, "confidence": 0.9}, "source": "observed"}
        self.payload["context"] = {"value": {"lie": "fairway"}, "source": "user_supplied"}

    def _resources(self, root):
        (root / "annotations.json").write_text(json.dumps(self.payload))
        (root / "calibration.json").write_text(json.dumps({
            "image_width": 1920, "image_height": 1080, "source_units": "pixels", "engine_units": "yards",
            "source_points": [{"x": 100, "y": 200}, {"x": 1700, "y": 200}, {"x": 1700, "y": 900}, {"x": 100, "y": 900}],
            "engine_points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]}))
        for name in ("sample_hole.json", "sample_player.json"):
            (root / name).write_text((DATA / name).read_text())

    def test_human_pipeline_runs_existing_pipeline_once_and_returns_analytics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self._resources(root); boundary = ProjectBoundary(root)
            import ghostcaddie.video.orchestration as orchestration
            calls = []
            original = orchestration.run_pipeline
            orchestration.run_pipeline = lambda *args: (calls.append(args) or original(*args))
            try:
                result = run_human_video_pipeline(load_human_annotations("annotations.json", boundary),
                    load_video_calibration("calibration.json", boundary),
                    JsonCourseDataSource(root / "sample_hole.json"), JsonPlayerProfileSource(root / "sample_player.json"),
                    Config.default(), event_id="E1", tournament_id="T1", hole_number=7, shot_number=2,
                    distance_to_pin=150, wind={"speed_mph": 8, "direction_deg": 90}, timestamp="t",
                    target_pixel=Point2D(700, 500))
            finally:
                orchestration.run_pipeline = original
            self.assertEqual(len(calls), 1)
            import ghostcaddie.pipeline as protected_pipeline
            self.assertNotIn("video-human", inspect.getsource(protected_pipeline))
            self.assertTrue(result.recommendation.recommended_club)
            self.assertIn("<svg", result.svg)
            self.assertEqual(result.metadata["source"], "video-human-annotations.v1")
            self.assertEqual(result.pipeline_status, "complete")

    def test_video_human_analyze_writes_three_artifacts_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self._resources(root); out = root / "out"
            video = root / "sample.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=10", "-t", "0.5", "-pix_fmt", "yuv420p", str(video)], check=True)
            main(["video-human-analyze", "--annotations", "annotations.json", "--calibration", "calibration.json",
                "--course", "sample_hole.json", "--player", "sample_player.json", "--project-root", str(root),
                "--video", str(video), "--out", str(out), "--event-id", "E1", "--target-x", "700", "--target-y", "500"])
            for name in ("recommendation.json", "overlay.svg", "normalized_shot.json"):
                self.assertTrue((out / name).is_file(), name)
            text = (out / "recommendation.json").read_text()
            self.assertNotIn(str(root), text)
            payload = json.loads(text)
            self.assertEqual(payload["analytics_status"], "complete")
            self.assertEqual(payload["provenance"]["provider"]["source"], "video-human-annotations.v1")

    def test_video_human_analyze_generates_offline_video_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self._resources(root); out = root / "out"
            video = root / "sample.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=1920x1080:r=10", "-t", "0.5", "-pix_fmt", "yuv420p",
                str(video)], check=True)
            main(["video-human-analyze", "--annotations", "annotations.json", "--calibration", "calibration.json",
                "--course", "sample_hole.json", "--player", "sample_player.json", "--video", str(video),
                "--project-root", str(root), "--out", str(out), "--event-id", "E1", "--target-x", "700", "--target-y", "500", "--render-video"])
            for name in ("contact_sheet.jpg", "annotation_workspace.html", "diagnostics.json", "normalized_shot.json", "recommendation.json", "overlay.svg", "annotated_video.mp4"):
                self.assertTrue((out / name).is_file(), name)
            self.assertTrue(list((out / "frames").glob("frame_*.jpg")))
            self.assertTrue(list((out / "annotated_frames").glob("frame_*.jpg")))
            workspace_html = (out / "annotation_workspace.html").read_text()
            self.assertIn('href",f.filename', workspace_html)
            self.assertIn("frames/frame_000001.jpg", workspace_html)
            diagnostics = json.loads((out / "diagnostics.json").read_text())
            self.assertEqual(diagnostics["status"], "complete")
            self.assertEqual(diagnostics["model_provider_provenance"]["mode"], "human-submitted")
            self.assertTrue(all(not Path(ref).is_absolute() for ref in diagnostics["artifact_references"]))
            for path in out.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".html", ".svg"}:
                    self.assertNotIn(str(root), path.read_text())

    def test_video_human_analyze_requires_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self._resources(root)
            with self.assertRaises(SystemExit):
                main(["video-human-analyze", "--annotations", "annotations.json", "--calibration", "calibration.json",
                    "--course", "sample_hole.json", "--player", "sample_player.json", "--project-root", str(root),
                    "--out", str(root / "out"), "--event-id", "E1", "--target-x", "700", "--target-y", "500"])


if __name__ == "__main__":
    unittest.main()
