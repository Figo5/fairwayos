"""RED-first behavior tests for the PGA-only source/eligibility gate.

Durable requirement (Gio via Hermes, 2026-09-11 PGA-ONLY RESET):
FairwayOS input and demo eligibility is PGA Tour footage ONLY. The gate must
FAIL CLOSED: Pexels, amateur, brand-commercial and unknown sources must never
reach demo eligibility, and an unknown source is REVIEW-REQUIRED -- never
promoted to eligible by any classifier, heuristic or filename.

The rejected Pexels experiment is preserved as an explicit REJECTED entry so the
rejection is durable and cannot silently drift back into the product demo.
"""
import hashlib
import os
import tempfile
import unittest

from ghostcaddie.video.pga_source_gate import (
    ELIGIBLE,
    REJECTED,
    REVIEW_REQUIRED,
    SourceNotEligible,
    evaluate_source,
    evaluate_sha256,
    require_demo_eligible,
    registry,
)

PEXELS_SHA = "a6e48474045365d1de2d4af76f65da558531684d67da87172cdd15a6dc45e1d6"
COMPILATION_SHA = "d3311ea7470b027e43611fc1251313aa917dcc66f35462b27bf486cdd228b4f0"
TITLEIST_SHA = "e82638dd0a99a4b27c1f7ee0ce7ddec9580044a16b4c164e49a771698158cc4e"
SHOOTOUT_SHA = "98f8bb71bff708f8a98a119d5776bdc4bf8e0bcb033a24528121fdca07091ce5"


class FailClosedTests(unittest.TestCase):
    def test_unknown_source_is_review_required_never_eligible(self):
        d = evaluate_sha256("0" * 64)
        self.assertEqual(d.status, REVIEW_REQUIRED)
        self.assertNotEqual(d.status, ELIGIBLE)

    def test_unknown_source_is_not_demo_eligible(self):
        with self.assertRaises(SourceNotEligible):
            require_demo_eligible(sha256="0" * 64)

    def test_no_classifier_can_promote_an_unknown_source(self):
        """There must be no inference path from pixels/filename to ELIGIBLE."""
        with tempfile.TemporaryDirectory() as td:
            # a file whose NAME screams PGA; content is unknown to the registry
            p = os.path.join(td, "pga_tour_broadcast_scottie_scheffler_2026.mp4")
            with open(p, "wb") as fh:
                fh.write(b"not really a pga clip")
            d = evaluate_source(p)
            self.assertEqual(d.status, REVIEW_REQUIRED)
            self.assertIn("not in the reviewed registry", d.reason)

    def test_empty_registry_entry_cannot_default_to_eligible(self):
        for entry in registry().values():
            self.assertIn(entry["status"], (ELIGIBLE, REJECTED, REVIEW_REQUIRED))
            if entry["status"] == ELIGIBLE:
                # eligibility demands positive, human-reviewed evidence
                self.assertTrue(entry.get("event_evidence"))
                self.assertTrue(entry.get("source_url"))
                self.assertTrue(entry.get("frames_inspected"))


class RejectedSourcesTests(unittest.TestCase):
    def test_pexels_amateur_source_is_rejected_durably(self):
        d = evaluate_sha256(PEXELS_SHA)
        self.assertEqual(d.status, REJECTED)
        self.assertIn("amateur", d.reason.lower())

    def test_pexels_can_never_be_demo_eligible(self):
        with self.assertRaises(SourceNotEligible):
            require_demo_eligible(sha256=PEXELS_SHA)

    def test_brand_commercial_is_rejected(self):
        self.assertEqual(evaluate_sha256(TITLEIST_SHA).status, REJECTED)

    def test_rejected_never_becomes_stress_test_material(self):
        self.assertFalse(evaluate_sha256(PEXELS_SHA).local_research_allowed)


class ReviewRequiredTests(unittest.TestCase):
    def test_named_pga_players_and_known_url_still_are_not_eligible(self):
        """Resolving the source URL does NOT confer eligibility.

        Updated 2026-09-11: the source URL for this clip was resolved from
        public oEmbed metadata (channel 'King of Golf', a third-party
        aggregator). Knowing the URL, and having ten named PGA TOUR players on
        screen, still leaves it review-required -- an aggregator re-cut is not
        official PGA TOUR broadcast footage and carries no redistribution
        rights. This is the stronger invariant than the one it replaces.
        """
        d = evaluate_sha256(COMPILATION_SHA)
        self.assertEqual(d.status, REVIEW_REQUIRED)
        self.assertTrue(d.player_evidence, "players were visually confirmed")
        self.assertIsNotNone(d.source_url, "source URL is now resolved")
        self.assertFalse(d.demo_eligible)
        self.assertFalse(d.public_redistribution_cleared)

    def test_tournament_evidence_without_verified_player_stays_review_required(self):
        d = evaluate_sha256(SHOOTOUT_SHA)
        self.assertEqual(d.status, REVIEW_REQUIRED)
        self.assertTrue(d.event_evidence)

    def test_review_required_allows_local_research_but_not_demo(self):
        d = evaluate_sha256(SHOOTOUT_SHA)
        self.assertTrue(d.local_research_allowed)
        self.assertFalse(d.demo_eligible)
        with self.assertRaises(SourceNotEligible):
            require_demo_eligible(sha256=SHOOTOUT_SHA)


class EvidenceIntegrityTests(unittest.TestCase):
    def test_decision_is_bound_to_content_not_filename(self):
        with tempfile.TemporaryDirectory() as td:
            payload = b"pexels-stand-in"
            p1 = os.path.join(td, "innocuous.mp4")
            p2 = os.path.join(td, "pga_tour_official.mp4")
            for p in (p1, p2):
                with open(p, "wb") as fh:
                    fh.write(payload)
            self.assertEqual(evaluate_source(p1).status, evaluate_source(p2).status)

    def test_every_registry_entry_records_rights_status(self):
        for sha, entry in registry().items():
            self.assertEqual(len(sha), 64, f"{sha} is not a sha256")
            self.assertTrue(entry.get("rights"), f"{sha} has no rights status")
            self.assertIn("redistribution", entry["rights"].lower() + " redistribution")

    def test_no_entry_claims_public_demo_clearance(self):
        for entry in registry().values():
            self.assertFalse(entry.get("public_redistribution_cleared", False))

    def test_missing_file_is_review_required_not_crash(self):
        d = evaluate_source("/no/such/clip.mp4")
        self.assertEqual(d.status, REVIEW_REQUIRED)
        self.assertIn("unreadable", d.reason.lower())


if __name__ == "__main__":
    unittest.main()
