import json
import math
import unittest

from ghostcaddie.video.contracts import VideoDiagnostics, VideoMetadata
from ghostcaddie.video.errors import VideoContractError


class TestVideoMetadata(unittest.TestCase):
    def test_valid_metadata_serializes_without_source_identifier(self):
        metadata = VideoMetadata(
            container_format="mov,mp4,m4a,3gp,3g2,mj2",
            codec="h264",
            width=1920,
            height=1080,
            frame_rate=59.94,
            frame_count=1200,
            duration_seconds=20.02,
            source_identifier="/private/secret/round.mp4",
        )
        payload = metadata.to_dict()
        self.assertEqual(payload["width"], 1920)
        self.assertNotIn("source_identifier", payload)
        self.assertNotIn("round.mp4", json.dumps(payload))

    def test_metadata_rejects_invalid_and_non_finite_values(self):
        for field, value in (("width", 0), ("height", -1), ("frame_rate", 0), ("duration_seconds", math.nan)):
            with self.subTest(field=field):
                kwargs = dict(container_format="mp4", codec="h264", width=1920, height=1080,
                              frame_rate=30.0, frame_count=None, duration_seconds=2.0)
                kwargs[field] = value
                with self.assertRaises(VideoContractError):
                    VideoMetadata(**kwargs)


class TestVideoDiagnostics(unittest.TestCase):
    def test_diagnostics_rejects_malformed_top_level_container_shapes(self):
        cases = (
            {"status": []},
            {"status": {}},
            {"status": None},
            {"artifact_references": None},
            {"artifact_references": "artifact.jpg"},
            {"frame_observations": None},
            {"frame_observations": "frames"},
            {"warnings": "warning"},
            {"confidence_values": None},
            {"confidence_values": []},
            {"model_provider_provenance": None},
            {"model_provider_provenance": []},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(VideoContractError):
                    VideoDiagnostics(**overrides)

    def test_versioned_diagnostics_contract_has_all_milestone_fields(self):
        diagnostics = VideoDiagnostics(
            status="complete",
            video_metadata={"container_format": "mp4", "codec": "h264", "width": 1,
                            "height": 1, "frame_rate": 30.0, "duration_seconds": 1.0},
            artifact_references=["diagnostics.json", "frames/contact.jpg"],
            frame_observations=[{"frame_index": 12, "timestamp_seconds": 0.4}],
            contact={"frame_index": 12},
            landing={"frame_index": 40},
            normalized_shot={"event_id": "shot-1"},
            analytics_result={"status": "pending"},
            confidence_values={"overall": 0.8},
            warnings=["optional field unavailable"],
            model_provider_provenance={"model": "none", "provider": "none"},
        )
        payload = diagnostics.to_dict()
        self.assertEqual(payload["schema_version"], "video-diagnostics.v1")
        for key in ("status", "video_metadata", "artifact_references", "frame_observations",
                    "contact", "landing", "normalized_shot", "analytics_result",
                    "confidence_values", "warnings", "model_provider_provenance"):
            self.assertIn(key, payload)
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_metadata_mapping_also_excludes_source_identifier(self):
        diagnostics = VideoDiagnostics(video_metadata={"codec": "h264", "source_identifier": "/secret/video.mp4"})
        serialized = diagnostics.to_dict()
        self.assertNotIn("source_identifier", serialized["video_metadata"])
        self.assertNotIn("video.mp4", json.dumps(serialized))

    def test_diagnostics_rejects_absolute_artifacts_and_sensitive_values(self):
        with self.assertRaises(VideoContractError):
            VideoDiagnostics(artifact_references=["/tmp/contact.jpg"])
        with self.assertRaises(VideoContractError):
            VideoDiagnostics(artifact_references=["frames/../secret.jpg"])
        with self.assertRaises(VideoContractError):
            VideoDiagnostics(warnings=["prompt: use SECRET_API_KEY=abc"])

    def test_diagnostics_rejects_absolute_source_paths_in_nested_payloads(self):
        with self.assertRaises(VideoContractError):
            VideoDiagnostics(frame_observations=[{"source_path": "/private/video.mp4"}])
        with self.assertRaises(VideoContractError):
            VideoDiagnostics(normalized_shot={"source_file": "~/Videos/round.mov"})


if __name__ == "__main__":
    unittest.main()
