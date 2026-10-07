"""Isolation guards for the v3 research pass.

Two things have to stay true for a v3 number to mean anything, and both are
enforced here rather than remembered:

* the v3 final-test split stays sealed until it is deliberately consumed, and
  nothing that trains can reach the v3 evaluation corpus at all;
* the canonical pipeline does not import any of this. The ball layer is
  ``UNAVAILABLE`` on the canonical path, and it stays that way unless a pass is
  classified PROMOTED -- which this one may not be.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V3_MODULES = ("ball_v3_blobs", "ball_v3_body", "ball_v3_context",
              "ball_v3_dataset", "ball_v3_proposals", "ball_v3_ranker",
              "ball_v3_tracker", "ball_v3_tracklets")

CANONICAL_MODULES = ("canonical", "canonical_worker")


def load_tool(name: str):
    """Import a research CLI as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location(
        f"v3_tool_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FinalTestStaysSealed(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = load_tool("ball_v3_eval")

    def test_final_test_is_refused_without_the_explicit_flag(self):
        with self.assertRaises(SystemExit) as caught:
            self.tool.load_split("final_test", False)
        self.assertIn("sealed", str(caught.exception))

    def test_final_test_opens_only_with_the_flag_and_matches_the_frozen_split(self):
        frozen = {c["clip_id"] for c in json.loads(
            (ROOT / "data/ball_dataset_v3_eval/split.json").read_text("utf-8"))["clips"]
            if c["split"] == "final_test"}
        records = json.loads((self.tool.V3 / "sources.json").read_text("utf-8"))["clips"]
        if not any((ROOT / r["media_path"]).is_file() for r in records
                   if r["clip_id"] in frozen):
            self.skipTest("Optional private final-test media is not distributed")
        clips = self.tool.load_split("final_test", True)
        # only clips whose media is on disk can be returned, so this is a subset
        self.assertTrue({c["clip_id"] for c in clips} <= frozen)
        self.assertTrue(clips, "no final-test media on disk")

    def test_validation_never_returns_a_final_test_clip(self):
        split = json.loads(
            (ROOT / "data/ball_dataset_v3_eval/split.json").read_text("utf-8"))
        sealed = {c["clip_id"] for c in split["clips"] if c["split"] == "final_test"}
        for clip in self.tool.load_split("validation", False):
            self.assertNotIn(clip["clip_id"], sealed)

    def test_the_configuration_hash_covers_what_must_be_frozen(self):
        """A final-test number is only reproducible if the configuration that
        produced it is recorded, so the digest must move when any of it moves."""
        from ghostcaddie.video import ball_v3_proposals as bp
        from ghostcaddie.video import ball_v3_tracker as btrack
        missing = Path("/nonexistent/weights.pt")
        base = self.tool.configuration_hash(
            btrack.TrackerParams(), bp.ProposalParams(), missing, None)
        changed = self.tool.configuration_hash(
            btrack.TrackerParams(min_margin=0.99), bp.ProposalParams(), missing, None)
        self.assertNotEqual(base["configuration_sha256"],
                            changed["configuration_sha256"])
        self.assertIn("tracker_params", base["configuration"])
        self.assertIn("proposal_params", base["configuration"])
        self.assertIn("generators", base["configuration"])


class ValidatorEntryPointStaysSealed(unittest.TestCase):
    """The validator's *entry point* must refuse a sealed clip before any read.

    The existing isolation test calls ``assert_not_final_test`` directly with
    recording stand-ins installed, which proves the guard itself touches nothing.
    It does not prove that the real ``main`` calls the guard *first*: a guard that
    runs after the decode is reached would pass that test and still decode a
    sealed clip. This one drives ``main`` with a mocked run list whose only entry
    is a sealed clip **id and metadata only** -- no media is read to build it --
    while the decoder, the body cache and the model loader are recording
    stand-ins.
    """

    def setUp(self) -> None:
        self.tool = load_tool("ball_v3_validate")

    def _install(self, touched: list):
        import cv2
        from ghostcaddie.video import ball_v3_body as bbody
        from ghostcaddie.video import ball_v3_ranker as brank
        real = (cv2.VideoCapture, bbody.BodyCache.pose, brank.load_ranker)

        def capture(*args, **kwargs):
            touched.append(f"decode:{args[:1]}")
            raise AssertionError("decode reached for a sealed clip")

        def pose(self, *args, **kwargs):
            touched.append(f"body_cache:{args[:1]}")
            raise AssertionError("body cache reached for a sealed clip")

        def loader(*args, **kwargs):
            touched.append("model_loader")
            raise AssertionError("model loader reached for a sealed clip")

        cv2.VideoCapture = capture
        bbody.BodyCache.pose = pose
        brank.load_ranker = loader
        return real

    def _restore(self, real) -> None:
        import cv2
        from ghostcaddie.video import ball_v3_body as bbody
        from ghostcaddie.video import ball_v3_ranker as brank
        cv2.VideoCapture, bbody.BodyCache.pose, brank.load_ranker = real

    def mocked_run(self, clip_id: str) -> dict:
        return {"clip_id": clip_id, "media": Path("/nonexistent/sealed_clip.mp4"),
                "indices": list(range(15)), "labels": {}, "labelled": True,
                "family": clip_id}

    def test_main_refuses_a_sealed_clip_before_any_media_or_cache_read(self):
        sealed = sorted(self.tool.sealed_clip_ids())
        self.assertTrue(sealed, "no sealed clip ids to test against")
        touched: list[str] = []
        real_runs = self.tool.legacy_runs
        real = self._install(touched)
        self.tool.legacy_runs = lambda limit: [self.mocked_run(sealed[0])]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out.json"
            try:
                with self.assertRaises(SystemExit) as caught:
                    self.tool.main(["legacy", "--out", str(out),
                                    "--frames-per-run", "15"])
            finally:
                self.tool.legacy_runs = real_runs
                self._restore(real)
            self.assertIn("sealed", str(caught.exception))
            self.assertEqual(touched, [],
                             f"the guard fired too late: {touched}")
            self.assertFalse(out.exists(), "main wrote an artifact anyway")

    def test_the_same_mock_without_a_sealed_id_does_reach_the_decoder(self):
        """The control: the stand-ins are wired, so the test above can fail."""
        touched: list[str] = []
        real_runs = self.tool.legacy_runs
        real = self._install(touched)
        self.tool.legacy_runs = lambda limit: [
            self.mocked_run("synthetic_clip_that_is_not_in_any_split")]
        try:
            with self.assertRaises(AssertionError):
                self.tool.main(["legacy", "--out",
                                str(Path(tempfile.gettempdir()) / "unused.json")])
        finally:
            self.tool.legacy_runs = real_runs
            self._restore(real)
        self.assertTrue(touched, "the decoder stand-in was never reached")


class TrainingCannotSeeTheEvaluationCorpus(unittest.TestCase):
    def test_the_trainer_only_reads_v2_train_clips(self):
        tool = load_tool("ball_v3_train")
        split = json.loads(
            (ROOT / "data/ball_dataset_v2/split.json").read_text("utf-8"))
        allowed = {c["clip_id"] for c in split["clips"] if c["split"] == "train"}
        for clip in tool.train_clips():
            self.assertIn(clip["clip_id"], allowed)

    def test_a_sealed_clip_is_refused_before_any_media_or_cache_access(self):
        """Behavioural, not textual.

        The check this replaces asserted that a substring appeared somewhere in a
        source file, and only looked at the first occurrence. That passes as soon
        as the right words are anywhere in the file, which is not the property
        that matters. This one installs recording stand-ins for the three things
        that could actually consume the corpus -- video decode, the body cache and
        the stored feature lists -- and asserts the guard fires before any of them
        is reached.
        """
        import cv2
        from ghostcaddie.video import ball_v3_body as bbody
        from ghostcaddie.video import ball_v3_dataset as bdata
        tool = load_tool("ball_v3_train")
        sealed = sorted(tool.sealed_clip_ids())
        self.assertTrue(sealed, "no sealed clip ids to test against")
        touched: list[str] = []
        real_capture = cv2.VideoCapture
        real_pose = bbody.BodyCache.pose
        real_reader = bdata.ExampleReader

        def capture(*args, **kwargs):
            touched.append(f"decode:{args[:1]}")
            raise AssertionError("decode reached for a sealed clip")

        def pose(self, *args, **kwargs):
            touched.append(f"body_cache:{args[:1]}")
            raise AssertionError("body cache reached for a sealed clip")

        def reader(*args, **kwargs):
            touched.append("feature_store")
            raise AssertionError("feature store reached for a sealed clip")

        cv2.VideoCapture = capture
        bbody.BodyCache.pose = pose
        bdata.ExampleReader = reader
        try:
            for clip_id in sealed:
                with self.assertRaises(SystemExit):
                    tool.assert_not_evaluation([clip_id])
        finally:
            cv2.VideoCapture = real_capture
            bbody.BodyCache.pose = real_pose
            bdata.ExampleReader = real_reader
        self.assertEqual(touched, [])

    def test_the_guard_is_not_a_blanket_refusal(self):
        """A made-up id passes, so the test above is about the sealed clips and
        not about a guard that refuses everything."""
        tool = load_tool("ball_v3_train")
        tool.assert_not_evaluation(["synthetic_clip_that_is_not_in_any_split"])
        with self.assertRaises(SystemExit):
            tool.assert_not_evaluation(
                ["synthetic_clip_that_is_not_in_any_split"] +
                [sorted(tool.sealed_clip_ids())[0]])

    def test_training_never_resolves_media_for_a_sealed_clip(self):
        tool = load_tool("ball_v3_train")
        sealed = tool.sealed_clip_ids()
        for clip in tool.train_clips():
            self.assertNotIn(clip["clip_id"], sealed)
            self.assertNotIn("ball_dataset_v3_eval", str(clip.get("media", "")))
        source = (ROOT / "tools" / "ball_v3_train.py").read_text("utf-8")
        for line in source.splitlines():
            if "ball_dataset_v3_eval" in line:
                self.assertIn("split", line.lower(),
                              "only the frozen split may be named in the "
                              f"trainer: {line.strip()}")


class CanonicalStaysIsolated(unittest.TestCase):
    def test_canonical_modules_do_not_import_v3_research_code(self):
        for name in CANONICAL_MODULES:
            source = (ROOT / "ghostcaddie" / "video" / f"{name}.py").read_text("utf-8")
            for module in V3_MODULES:
                self.assertNotIn(module, source,
                                 f"{name}.py imports {module}: the v3 pass is "
                                 "research-only unless it is PROMOTED")

    def test_the_canonical_ball_layer_is_still_unavailable(self):
        from ghostcaddie.video import canonical
        layer = canonical.ball_layer()
        self.assertEqual(layer.state, "unavailable")
        self.assertEqual(layer.method, "none")
        self.assertEqual(layer.coverage["frames_with_observation"], 0)
        self.assertEqual(layer.provenance["routes_admitted"], [])

    def test_v3_results_are_never_marked_production_eligible(self):
        from ghostcaddie.video import ball_v3_tracker as btrack
        provenance = btrack._provenance(None, {}, None)
        self.assertTrue(provenance["research_only"])
        self.assertFalse(provenance["ground_truth"])
        self.assertTrue(provenance["no_network"])


if __name__ == "__main__":
    unittest.main()
