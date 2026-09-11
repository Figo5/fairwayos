"""TDD: source-specific assistance must be explicit and hash-bound."""
import unittest
from ghostcaddie.upload.seeds import Seed, SeedBundle, SeedRejected


class SeedTests(unittest.TestCase):
    def test_seed_requires_the_source_hash(self):
        with self.assertRaises(SeedRejected):
            SeedBundle.from_dict({"seeds": [{"target": "clubhead", "frame": 310,
                                             "box_xyxy": [1, 2, 3, 4]}]},
                                 source_sha256="a" * 64)

    def test_seed_bound_to_another_source_is_refused(self):
        with self.assertRaises(SeedRejected) as cm:
            SeedBundle.from_dict({"source_sha256": "b" * 64, "seeds": []},
                                 source_sha256="a" * 64)
        self.assertIn("different source", str(cm.exception).lower())

    def test_matching_seed_is_accepted_and_marked_assisted(self):
        b = SeedBundle.from_dict(
            {"source_sha256": "a" * 64,
             "seeds": [{"target": "clubhead", "frame": 310, "box_xyxy": [1, 2, 3, 4]}]},
            source_sha256="a" * 64)
        s = b.for_target("clubhead")
        self.assertIsInstance(s, Seed)
        self.assertEqual(s.initialization, "assisted")
        self.assertTrue(s.disclosure)

    def test_absent_target_seed_returns_none_not_a_guess(self):
        b = SeedBundle.from_dict({"source_sha256": "a" * 64, "seeds": []},
                                 source_sha256="a" * 64)
        self.assertIsNone(b.for_target("ball"))

    def test_seed_geometry_is_validated(self):
        with self.assertRaises(SeedRejected):
            SeedBundle.from_dict(
                {"source_sha256": "a" * 64,
                 "seeds": [{"target": "clubhead", "frame": 1, "box_xyxy": [5, 5, 1, 1]}]},
                source_sha256="a" * 64)

    def test_seed_is_never_ground_truth(self):
        b = SeedBundle.from_dict(
            {"source_sha256": "a" * 64,
             "seeds": [{"target": "clubhead", "frame": 3, "box_xyxy": [0, 0, 4, 4]}]},
            source_sha256="a" * 64)
        d = b.for_target("clubhead").to_dict()
        self.assertFalse(d["ground_truth"])
        self.assertTrue(d["pseudo_label"])


if __name__ == "__main__":
    unittest.main()
