import unittest

from ghostcaddie.video.clubhead_annotation_dataset import (
    SCHEMA_VERSION,
    ClubheadAnnotationDataset,
    serialize_dataset,
    validate_dataset,
)
from ghostcaddie.video.errors import VideoContractError


class TestClubheadAnnotationDataset(unittest.TestCase):
    def valid_payload(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "draft",
            "explicit_submit": False,
            "video": {
                "clip_id": "mmu_seed",
                "width": 600,
                "height": 480,
                "frame_count": 2,
                "frame_rate": 25.0,
            },
            "split": "train",
            "rights": "research_only_local",
            "provenance": {
                "label_type": "human",
                "pseudo_label": False,
                "ground_truth": True,
                "research_only": True,
                "production_eligible": False,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "source_frame_index": 40,
                    "timestamp_seconds": 1.6,
                    "clubhead": {
                        "value": {"x": 120.0, "y": 300.0},
                        "visibility": "visible",
                        "source": "human_ground_truth",
                    },
                    "shaft": {
                        "value": {
                            "grip": {"x": 300.0, "y": 100.0},
                            "neck": {"x": 130.0, "y": 290.0},
                        },
                        "visibility": "visible",
                        "source": "human_ground_truth",
                    },
                    "notes": [],
                },
                {
                    "frame_index": 1,
                    "source_frame_index": 41,
                    "timestamp_seconds": 1.64,
                    "clubhead": {"value": None, "visibility": "occluded", "source": "unavailable"},
                    "shaft": {"value": None, "visibility": "occluded", "source": "unavailable"},
                    "notes": ["clubhead occluded by ball/scene"],
                },
            ],
            "warnings": [],
        }

    def test_validates_frame_level_clubhead_and_shaft_labels(self):
        dataset = ClubheadAnnotationDataset(self.valid_payload())
        self.assertEqual(len(dataset.frames), 2)
        self.assertEqual(dataset.frames[0]["source_frame_index"], 40)
        self.assertFalse(dataset.to_dict()["provenance"]["production_eligible"])

    def test_requires_explicit_pseudo_label_flags(self):
        payload = self.valid_payload()
        payload["provenance"] = {
            "label_type": "pseudo",
            "pseudo_label": True,
            "ground_truth": False,
            "research_only": True,
            "production_eligible": False,
        }
        validate_dataset(payload)
        payload["provenance"]["ground_truth"] = True
        with self.assertRaises(VideoContractError):
            validate_dataset(payload)

    def test_rejects_invalid_frame_mapping_or_visible_missing_point(self):
        payload = self.valid_payload()
        payload["frames"][1]["source_frame_index"] = 40
        with self.assertRaises(VideoContractError):
            validate_dataset(payload)
        payload = self.valid_payload()
        payload["frames"][1]["clubhead"] = {
            "value": None,
            "visibility": "visible",
            "source": "unavailable",
        }
        with self.assertRaises(VideoContractError):
            validate_dataset(payload)

    def test_rejects_frame_count_that_does_not_match_annotation_frames(self):
        payload = self.valid_payload()
        payload["video"]["frame_count"] = 99
        with self.assertRaises(VideoContractError):
            validate_dataset(payload)

    def test_serialization_is_deterministic_and_rejects_nonfinite_values(self):
        payload = self.valid_payload()
        first = serialize_dataset(payload)
        second = serialize_dataset(ClubheadAnnotationDataset.from_json(first))
        self.assertEqual(first, second)
        payload["frames"][0]["clubhead"]["value"]["x"] = float("nan")
        with self.assertRaises(VideoContractError):
            validate_dataset(payload)


if __name__ == "__main__":
    unittest.main()
