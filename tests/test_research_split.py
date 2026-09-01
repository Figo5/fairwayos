import unittest

from ghostcaddie.video.errors import VideoContractError
from ghostcaddie.video.research_split import (
    SCHEMA_VERSION,
    serialize_split_manifest,
    validate_split_manifest,
)


class TestResearchSplitManifest(unittest.TestCase):
    def valid_payload(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "owned-research-v1",
            "status": "frozen",
            "clips": [
                {"clip_id": "clip-a", "source_id": "source-a", "subject_id": "golfer-a", "sequence_id": "seq-a", "sha256": "a" * 64, "split": "train"},
                {"clip_id": "clip-b", "source_id": "source-b", "subject_id": "golfer-b", "sequence_id": "seq-b", "sha256": "b" * 64, "split": "validation"},
                {"clip_id": "clip-c", "source_id": "source-c", "subject_id": "golfer-c", "sequence_id": "seq-c", "sha256": "c" * 64, "split": "held_out"},
            ],
            "warnings": [],
        }

    def test_validates_frozen_golfer_disjoint_partitions(self):
        result = validate_split_manifest(self.valid_payload())
        self.assertEqual(result["status"], "frozen")
        self.assertEqual(serialize_split_manifest(result), serialize_split_manifest(self.valid_payload()))

    def test_rejects_missing_required_partition(self):
        payload = self.valid_payload()
        payload["clips"] = payload["clips"][:2]
        with self.assertRaisesRegex(VideoContractError, "held_out"):
            validate_split_manifest(payload)

    def test_rejects_subject_source_or_sequence_leakage(self):
        for field in ("subject_id", "source_id", "sequence_id"):
            payload = self.valid_payload()
            payload["clips"][2][field] = payload["clips"][0][field]
            with self.subTest(field=field), self.assertRaises(VideoContractError):
                validate_split_manifest(payload)

    def test_rejects_duplicate_clip_hash_and_unsafe_paths(self):
        payload = self.valid_payload()
        payload["clips"][1]["clip_id"] = payload["clips"][0]["clip_id"]
        with self.assertRaises(VideoContractError):
            validate_split_manifest(payload)
        payload = self.valid_payload()
        payload["clips"][0]["source_id"] = "/private/video.mp4"
        with self.assertRaises(VideoContractError):
            validate_split_manifest(payload)

    def test_requires_frozen_status_and_valid_hash(self):
        payload = self.valid_payload()
        payload["status"] = "draft"
        with self.assertRaises(VideoContractError):
            validate_split_manifest(payload)
        payload = self.valid_payload()
        payload["clips"][0]["sha256"] = "not-a-hash"
        with self.assertRaises(VideoContractError):
            validate_split_manifest(payload)


if __name__ == "__main__":
    unittest.main()
