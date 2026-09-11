"""RED-first: a missing clubhead seed must not destroy ball and pose output.

Reset brief (2026-09-11): "Do not wait for perfect clubhead/impact to deliver
useful golf analysis."

Measured on real PGA footage 2026-09-11: both the evaluation interval (Will
Zalatoris) and the held-out interval (Rory McIlroy) aborted the WHOLE run with
reason "no reliable club seed", discarding pose and ball as well. Clubhead is
locally falsified at this resolution, so gating every other layer behind it
means PGA footage can never produce anything automatically.

Correct behaviour: layers degrade INDEPENDENTLY. A layer that cannot initialise
reports unavailable with its reason; layers that can still run, still run.
"""
import unittest

from ghostcaddie.video.pga_research_analyzer import (
    automatic_tee_seed,
    layer_initialisation,
)


class LayerDegradationTests(unittest.TestCase):
    def test_missing_club_seed_does_not_block_ball(self):
        st = layer_initialisation(ball_tee=(556.0, 623.0), club_seed=None)
        self.assertTrue(st["ball"]["can_run"])
        self.assertFalse(st["clubhead"]["can_run"])
        self.assertEqual(st["clubhead"]["state"], "unavailable")
        self.assertIn("seed", st["clubhead"]["reason"])

    def test_missing_ball_tee_does_not_block_clubhead(self):
        st = layer_initialisation(ball_tee=None, club_seed={"x": 1, "y": 2,
                                                            "source_frame": 0})
        self.assertFalse(st["ball"]["can_run"])
        self.assertTrue(st["clubhead"]["can_run"])

    def test_both_missing_blocks_both_but_still_reports(self):
        st = layer_initialisation(ball_tee=None, club_seed=None)
        self.assertFalse(st["ball"]["can_run"])
        self.assertFalse(st["clubhead"]["can_run"])
        for layer in ("ball", "clubhead"):
            self.assertEqual(st[layer]["state"], "unavailable")
            self.assertTrue(st[layer]["reason"])

    def test_any_layer_runnable_means_run_proceeds(self):
        st = layer_initialisation(ball_tee=(1.0, 2.0), club_seed=None)
        self.assertTrue(any(v["can_run"] for v in st.values()))

    def test_seed_sources_are_labelled_never_called_automatic_detection(self):
        st = layer_initialisation(ball_tee=(1.0, 2.0), club_seed=None,
                                  ball_tee_source="automatic_unconfirmed")
        self.assertEqual(st["ball"]["seed_source"], "automatic_unconfirmed")
        self.assertNotEqual(st["ball"]["seed_source"], "detected")

    def test_assisted_seed_is_labelled_assisted(self):
        st = layer_initialisation(ball_tee=(1.0, 2.0), club_seed=None,
                                  ball_tee_source="assisted")
        self.assertEqual(st["ball"]["seed_source"], "assisted")


class AutomaticTeeSeedTests(unittest.TestCase):
    """The seed locator must stay bounded and must never invent a seed."""

    def test_no_frames_yields_no_seed(self):
        self.assertIsNone(automatic_tee_seed([], roi_fraction=0.5))

    def test_roi_fraction_is_validated(self):
        with self.assertRaises(ValueError):
            automatic_tee_seed([], roi_fraction=0.0)
        with self.assertRaises(ValueError):
            automatic_tee_seed([], roi_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
