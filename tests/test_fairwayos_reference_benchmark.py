import json
import unittest
from pathlib import Path

from ghostcaddie.video.clubhead_annotation_dataset import (
    SCHEMA_VERSION,
    validate_dataset,
)


FIXTURE = Path(__file__).parent / "fixtures" / "fairwayos_reference_benchmark.json"


class FairwayOSReferenceBenchmarkTests(unittest.TestCase):
    def test_fixture_is_a_minimal_non_ground_truth_reference(self):
        payload = json.loads(FIXTURE.read_text())
        validated = validate_dataset(payload)
        provenance = validated["provenance"]
        self.assertEqual(validated["schema_version"], SCHEMA_VERSION)
        self.assertEqual(provenance["label_type"], "model")
        self.assertTrue(provenance["pseudo_label"])
        self.assertFalse(provenance["ground_truth"])
        self.assertTrue(provenance["research_only"])
        self.assertFalse(provenance["production_eligible"])
        self.assertEqual(validated["frames"][0]["clubhead"]["source"], "unavailable")
        self.assertEqual(validated["frames"][0]["shaft"]["source"], "unavailable")

    def test_fixture_keeps_ai_reference_separate_from_ground_truth(self):
        payload = json.loads(FIXTURE.read_text())
        self.assertNotIn("ground_truth", payload["frames"][0])
        self.assertEqual(payload["provenance"]["ground_truth"], False)
        self.assertTrue(payload["warnings"][0].startswith("AI-assisted reference only"))


if __name__ == "__main__":
    unittest.main()
