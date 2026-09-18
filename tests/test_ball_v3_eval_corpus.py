"""The v3 evaluation corpus must stay uncontaminated and frozen.

These are the guards that make a v3 final-test number mean anything. They are
deliberately blunt: if the corpus ever overlaps the data the models were built
from, the number is not a generalisation result and should fail loudly here
rather than be discovered afterwards.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ghostcaddie.video import ball_dataset as bds

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "ball_dataset_v3_eval"
V2_SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"
LEGACY_SPLIT = ROOT / "data" / "ball_split.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _family(clip_id: str) -> str:
    return "pexels_" + str(clip_id).removeprefix("pexels_")[:4]


class V3EvalCorpus(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = _load(V3 / "sources.json")
        self.split = _load(V3 / "split.json")

    def test_manifests_satisfy_the_repository_contract(self):
        bds.validate_sources(self.sources)
        bds.validate_split(self.split, sources=self.sources)

    def test_split_is_frozen(self):
        self.assertEqual(self.split["status"], "frozen")

    def test_evaluation_only_no_train_split(self):
        splits = {clip["split"] for clip in self.split["clips"]}
        self.assertNotIn("train", splits,
                         "the v3 corpus is evaluation-only; a train split here "
                         "would let a model see its own test set")
        self.assertEqual(splits, {"validation", "final_test"})

    def test_no_clip_or_family_comes_from_the_v2_or_legacy_corpus(self):
        forbidden_clips: set[str] = set()
        forbidden_families: set[str] = set()
        for path in (V2_SPLIT, LEGACY_SPLIT):
            for clip in _load(path)["clips"]:
                forbidden_clips.add(clip["clip_id"])
                forbidden_families.add(_family(clip["clip_id"]))
                group = clip.get("source_group") or clip.get("source_id")
                if group:
                    forbidden_families.add(group)
        for clip in self.split["clips"]:
            self.assertNotIn(clip["clip_id"], forbidden_clips,
                             f"{clip['clip_id']} is already a v2/legacy clip")
            self.assertNotIn(clip["source_group"], forbidden_families,
                             f"{clip['clip_id']} shares a shoot family with the "
                             "corpus the models were built from")
            self.assertNotIn(_family(clip["clip_id"]), forbidden_families)

    def test_no_shoot_family_straddles_validation_and_final_test(self):
        placement: dict[str, str] = {}
        for clip in self.split["clips"]:
            group = clip["source_group"]
            if group in placement:
                self.assertEqual(placement[group], clip["split"],
                                 f"family {group} appears in two splits")
            placement[group] = clip["split"]

    def test_each_split_can_measure_both_tracking_and_fabrication(self):
        for name in ("validation", "final_test"):
            clips = [c for c in self.split["clips"] if c["split"] == name]
            self.assertGreaterEqual(len(clips), 5, name)
            self.assertTrue(
                any(c["content_class"] in ("swing", "putting") for c in clips),
                f"{name} has no footage of a ball being played")
            self.assertTrue(
                any(not c["holds_playable_ball"] for c in clips),
                f"{name} has no ball-free sequence, so it cannot measure whether "
                "the tracker fabricates a ball")

    def test_every_split_clip_is_in_sources_with_a_matching_hash(self):
        by_id = {clip["clip_id"]: clip for clip in self.sources["clips"]}
        for clip in self.split["clips"]:
            self.assertIn(clip["clip_id"], by_id)
            self.assertEqual(by_id[clip["clip_id"]]["sha256"], clip["sha256"])

    def test_media_is_not_committed(self):
        for clip in self.sources["clips"]:
            self.assertFalse(clip["media_path"].startswith("data/"),
                             "media must live under out/ and stay gitignored")
        self.assertFalse(self.sources["license_summary"]["media_committed"])

    def test_every_clip_is_rights_cleared_stock(self):
        for clip in self.sources["clips"]:
            self.assertEqual(clip["rights_class"], "stock_license_reusable")
            self.assertEqual(clip["provider"], "pexels")
            self.assertTrue(clip["license_url"])


if __name__ == "__main__":
    unittest.main()
