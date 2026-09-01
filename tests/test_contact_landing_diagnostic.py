import unittest

from ghostcaddie.video.contact_landing_diagnostic import diagnose_contact_landing


class ContactLandingDiagnosticTests(unittest.TestCase):
    def test_contact_candidate_keeps_swingnet_event_separate_from_validation(self):
        result = diagnose_contact_landing(
            ball_track=[
                {"frame_index": 9, "timestamp_seconds": 0.90, "x": 10, "y": 20, "confidence": 0.9, "provenance": "observed"},
                {"frame_index": 10, "timestamp_seconds": 1.00, "x": 12, "y": 19, "confidence": 0.85, "provenance": "observed"},
            ],
            event_timestamps=[{"event": "Impact", "timestamp_seconds": 1.01, "confidence": 0.92, "source": "SwingNet"}],
            pose_cues=[{"timestamp_seconds": 1.0, "cues": {"contact": 0.8}}],
        )
        self.assertTrue(result.available)
        self.assertEqual([c.kind for c in result.candidates], ["contact"])
        self.assertEqual(result.candidates[0].event_source, "swingnet")
        self.assertEqual(result.validated_events, ())
        self.assertFalse(result.candidates[0].validated)
        self.assertIn("explicit_ball_candidate", result.candidates[0].evidence)

    def test_research_candidate_provenance_cannot_supply_contact_evidence(self):
        result = diagnose_contact_landing(
            ball_track=[
                {"frame_index": 10, "timestamp_seconds": 1.0, "x": 1, "y": 2,
                 "confidence": 0.99, "provenance": "research_candidate"},
            ],
            event_timestamps=[{"event": "Impact", "timestamp_seconds": 1.0,
                               "confidence": 1.0, "source": "native"}],
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.candidates)
        self.assertEqual(result.reason, "no_explicit_ball_candidate_near_event")

    def test_no_explicit_ball_point_returns_unavailable_even_with_event_and_pose(self):
        result = diagnose_contact_landing(
            ball_track=[{"frame_index": 10, "timestamp_seconds": 1.0, "x": 1, "y": 2, "confidence": 0.99, "provenance": "predicted"}],
            event_timestamps=[{"event": "Impact", "timestamp_seconds": 1.0, "confidence": 1.0, "source": "native"}],
            pose_cues=[{"timestamp_seconds": 1.0, "cues": {"contact": 1.0}}],
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.candidates)
        self.assertEqual(result.reason, "no_explicit_ball_candidate_near_event")

    def test_missing_or_untrusted_ball_provenance_cannot_supply_contact_evidence(self):
        for provenance in (None, "inferred", "made_up", "generic_model_prediction"):
            item = {"frame_index": 10, "timestamp_seconds": 1.0, "x": 1, "y": 2,
                    "confidence": 0.99}
            if provenance is not None:
                item["provenance"] = provenance
            result = diagnose_contact_landing(
                [item],
                [{"event": "Impact", "timestamp_seconds": 1.0,
                  "confidence": 1.0, "source": "native"}],
            )
            with self.subTest(provenance=provenance):
                self.assertFalse(result.available)
                self.assertEqual(result.reason, "no_explicit_ball_candidate_near_event")

    def test_landing_requires_native_landing_timestamp_or_strong_generic_pose_and_two_points(self):
        track = [
            {"frame_index": 20, "timestamp_seconds": 2.0, "x": 20, "y": 10, "confidence": 0.9, "provenance": "observed"},
            {"frame_index": 21, "timestamp_seconds": 2.1, "x": 21, "y": 10, "confidence": 0.9, "provenance": "observed"},
        ]
        weak = diagnose_contact_landing(track, [{"event": "Impact", "timestamp_seconds": 1.0, "confidence": 0.9, "source": "SwingNet"}], [{"timestamp_seconds": 2.0, "cues": {"landing": 0.7}}])
        self.assertFalse(weak.available)
        strong = diagnose_contact_landing(track, [{"event": "Impact", "timestamp_seconds": 1.0, "confidence": 0.9, "source": "SwingNet"}], [{"timestamp_seconds": 2.0, "cues": {"landing": 0.9}}])
        self.assertTrue(strong.available)
        self.assertEqual([c.kind for c in strong.candidates], ["landing"])
        native = diagnose_contact_landing(track, [{"event": "Landing", "timestamp_seconds": 2.02, "confidence": 0.8, "source": "native"}], [])
        self.assertTrue(native.available)
        self.assertEqual(native.candidates[0].kind, "landing")

    def test_insufficient_event_confidence_is_unavailable_and_order_is_deterministic(self):
        result = diagnose_contact_landing(
            [{"frame_index": 1, "timestamp_seconds": 0.1, "x": 2, "y": 3, "confidence": 1.0, "provenance": "observed"}],
            [{"event": "Impact", "timestamp_seconds": 0.1, "confidence": 0.2, "source": "native"}],
            [],
        )
        self.assertFalse(result.available)
        self.assertIsNone(result.candidates)
        self.assertEqual(result.reason, "event_confidence_below_threshold")


if __name__ == "__main__":
    unittest.main()
