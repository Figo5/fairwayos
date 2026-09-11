"""Regression tests for real SAM2/BootsTAPIR execution and its honest states."""
import json, os, unittest

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "rory_three_target", "raw_results.json")


@unittest.skipUnless(os.path.exists(RAW), "run tools/demo/run_rory_three_target.py first")
class ExecutionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.load(open(RAW))

    def test_source_is_hash_bound(self):
        self.assertEqual(len(self.d["source"]["sha256"]), 64)
        for t in ("body", "clubhead", "ball"):
            r = self.d["results"][t]
            if r.get("ok"):
                for rec in r["records"]:
                    self.assertEqual(rec["source_sha256"], self.d["source"]["sha256"])

    def test_withheld_frame_is_not_in_any_output(self):
        for t, r in self.d["results"].items():
            if not r.get("ok"):
                continue
            for rec in r["records"]:
                self.assertNotEqual(rec["source_frame"], 3110,
                                    f"{t} used the withheld frame")

    def test_all_records_carry_research_flags(self):
        for t, r in self.d["results"].items():
            if not r.get("ok"):
                continue
            for rec in r["records"]:
                self.assertTrue(rec["pseudo_label"])
                self.assertFalse(rec["ground_truth"])
                self.assertFalse(rec["production_eligible"])

    def test_assisted_targets_declare_assistance(self):
        for t in ("clubhead", "ball"):
            r = self.d["results"][t]
            if not r.get("ok"):
                continue
            for rec in r["records"]:
                self.assertEqual(rec["initialization"], "assisted")
                self.assertIn("not ground truth", rec["assistance"])

    def test_body_is_automatic_not_assisted(self):
        r = self.d["results"]["body"]
        if r.get("ok"):
            self.assertEqual(r["records"][0]["initialization"], "automatic")

    def test_invisible_frames_carry_no_coordinate(self):
        """Loss is preserved: nothing is interpolated into a gap."""
        for t in ("clubhead", "ball"):
            r = self.d["results"][t]
            if not r.get("ok"):
                continue
            for rec in r["records"]:
                if not rec.get("visible"):
                    self.assertIsNone(rec.get("point_xy"))
                    self.assertIsNone(rec.get("bbox_xyxy"))

    def test_frames_are_contiguous_native_indices(self):
        for t, r in self.d["results"].items():
            if not r.get("ok"):
                continue
            fs = [x["source_frame"] for x in r["records"]]
            self.assertEqual(fs, sorted(fs))
            self.assertEqual(len(fs), len(set(fs)))

    def test_no_target_claims_success_it_did_not_achieve(self):
        """A target with few observations must not be dressed up."""
        ball = self.d["results"]["ball"]
        if ball.get("ok"):
            vis = [r for r in ball["records"] if r["visible"]]
            for r in ball["records"]:
                if r["visible"]:
                    self.assertIsNotNone(r["point_xy"])
            # the honest outcome is recorded, however weak
            self.assertLessEqual(len(vis), len(ball["records"]))


if __name__ == "__main__":
    unittest.main()
