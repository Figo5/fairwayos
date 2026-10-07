"""Worker-pass-1 regressions: provenance, grouping, features, splits, abstention.

Each of these pins a property that a measurement in
``docs/ball-tracker-v3-results.md`` depends on. If one of them breaks, the
number it supports stops meaning what the document says it means:

* a shard is one synthetic sequence, and an interrupted shard is not read;
* two sequences whose windows touch are never linked into one track;
* a feature vector is never zero by absence;
* the temporal split is by shoot family and is aligned with the ranker holdout;
* the tracker emits no coordinate on a frame its accepted track does not hold.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_v3_dataset as bdata      # noqa: E402
from ghostcaddie.video import ball_v3_proposals as bp       # noqa: E402
from ghostcaddie.video import ball_v3_ranker as brank       # noqa: E402
from ghostcaddie.video import ball_v3_tracklets as btrk     # noqa: E402


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(
        f"worker_tool_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def turf(width: int = 240, height: int = 180, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = rng.integers(20, 55, (height, width))
    frame[..., 1] = rng.integers(90, 150, (height, width))
    frame[..., 2] = rng.integers(30, 70, (height, width))
    return frame


def node(index: int, x: float, y: float, *, radius: float = 5.0,
         probability: float = 0.9, torso=None) -> btrk.Node:
    return btrk.Node(frame_index=index, x=x, y=y, radius=radius,
                     probability=probability, margin=0.4, is_top1=True,
                     torso_centre=torso)


def complete(one: btrk.Node) -> btrk.Node:
    """Fill the caller-supplied fields so the node is not a gap."""
    one.chroma_score = 0.5
    one.brightness_score = 0.2
    one.radius_over_short_side = 0.004
    one.appearance = (120.0, 18.0, 25.0)
    one.body_available = True
    return one


class FrameViews(unittest.TestCase):
    """The shared-pyramid crop must describe the same place as the direct one."""

    def _textured(self):
        import cv2
        rng = np.random.default_rng(17)
        frame = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
        cv2.circle(frame, (400, 300), 40, (250, 250, 250), -1)
        cv2.circle(frame, (200, 200), 6, (240, 240, 240), -1)
        return frame

    def test_the_crop_matches_the_direct_path_within_a_bounded_delta(self):
        from ghostcaddie.video import ball_v3_context as bctx
        frame = self._textured()
        views = bctx.FrameViews(frame)
        for x, y, radius in ((400.0, 300.0, 40.0), (200.0, 200.0, 6.0),
                             (400.0, 300.0, 12.0), (390.0, 290.0, 2.0)):
            with self.subTest(radius=radius):
                direct = bctx.candidate_stack(frame, None, x, y, radius)
                shared = views.stack(x, y, radius)
                self.assertEqual(direct.shape, shared.shape)
                delta = np.abs(direct.astype(np.int16)
                               - shared.astype(np.int16))
                # a slightly different resampling order, not a different place:
                # the mean per-channel difference stays in the low single digits
                # out of 255, and the tight crop -- the one that describes the
                # object -- is unchanged for radii at or below the level target
                self.assertLess(float(delta.mean()), 4.0,
                                f"mean channel delta {delta.mean()}")

    def test_small_candidates_are_untouched(self):
        """At or below the level target both crops are taken at full resolution,
        so the two paths are bit-identical. Radius 4 is the boundary: its context
        window is exactly ``CONTEXT_LEVEL_PX``."""
        from ghostcaddie.video import ball_v3_context as bctx
        frame = self._textured()
        views = bctx.FrameViews(frame)
        self.assertEqual(bctx.CONTEXT_LEVEL_PX, 128)
        for radius in (2.0, 3.0, 4.0):
            direct = bctx.candidate_stack(frame, None, 200.0, 200.0, radius)
            shared = views.stack(200.0, 200.0, radius)
            np.testing.assert_array_equal(direct, shared)
        # above it the context crop is resampled from a level: bounded, not equal
        direct = bctx.candidate_stack(frame, None, 200.0, 200.0, 8.0)
        shared = views.stack(200.0, 200.0, 8.0)
        self.assertLess(float(np.abs(direct.astype(np.int16)
                                     - shared.astype(np.int16)).mean()), 2.0)
        # the tight crop is what describes the object, and it is unchanged
        np.testing.assert_array_equal(direct[..., :3], shared[..., :3])


class Shards(unittest.TestCase):
    def example(self, index: int, *, sequence_id: str, clip_id: str = "clip_a"):
        stacks = np.full((3, bdata.bctx.GRID, bdata.bctx.GRID, bdata.bctx.CHANNELS),
                         7, np.uint8)
        numeric = np.zeros((3, brank.NUMERIC_FEATURES), np.float32)
        return bdata.FrameExample(
            clip_id=clip_id, frame_index=index, stacks=stacks, numeric=numeric,
            target=1, truth=(10.0, 10.0, 5.0), proposed=True, width=1920,
            height=1080, sequence_id=sequence_id, round_index=3,
            source_sha256="a" * 64, split="train", source_group="pexels_1234",
            background_ball_annotation=False, pool_size=40)

    def test_an_interrupted_shard_is_skipped_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_shard = root / "good"
            writer = bdata.ExampleWriter(complete_shard)
            writer.add(self.example(10, sequence_id="s1"))
            writer.close()
            broken = root / "bad"
            broken.mkdir()
            (broken / "index.jsonl").write_text("{}\n", encoding="utf-8")
            (broken / "stacks.u8").write_bytes(b"")

            reader = bdata.MultiReader.from_root(root)
            self.assertEqual(len(reader.records), 1)
            self.assertEqual([Path(p).name for p in reader.skipped], ["bad"])

    def test_a_complete_shard_round_trips_its_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard"
            writer = bdata.ExampleWriter(shard, meta={"synthetic_kind": "fast"})
            writer.add(self.example(10, sequence_id="clip_a#r03"))
            summary = writer.close()
            self.assertEqual(summary["lists"], 1)
            self.assertEqual(len(summary["stacks_sha256"]), 64)

            reader = bdata.ExampleReader(shard)
            record = reader.records[0]
            self.assertEqual(record["sequence_id"], "clip_a#r03")
            self.assertEqual(record["round_index"], 3)
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["source_group"], "pexels_1234")
            self.assertEqual(record["pool_size"], 40)
            self.assertEqual(record["source_sha256"], "a" * 64)
            self.assertEqual(reader.stacks_for(record).shape,
                             (3, bdata.bctx.GRID, bdata.bctx.GRID, bdata.bctx.CHANNELS))

    def test_a_shard_with_no_lists_is_readable(self):
        """A sequence whose every frame was a proposal miss still generated.

        Its stacks file is empty, which cannot be memory-mapped; refusing to read
        it would lose the record that the round ran and produced nothing.
        """
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "empty"
            writer = bdata.ExampleWriter(shard, meta={"synthetic_kind": "stationary"})
            summary = writer.close()
            self.assertEqual(summary["lists"], 0)
            reader = bdata.ExampleReader(shard)
            self.assertEqual(len(reader.records), 0)
            combined = bdata.MultiReader.from_root(Path(directory))
            self.assertEqual(len(combined.records), 0)
            self.assertEqual(combined.skipped, [])

    def test_a_shard_cannot_be_closed_twice(self):
        """Double-closing would write a second manifest over a live one."""
        with tempfile.TemporaryDirectory() as directory:
            writer = bdata.ExampleWriter(Path(directory) / "shard")
            writer.add(self.example(10, sequence_id="s1"))
            writer.close()
            with self.assertRaises(RuntimeError):
                writer.close()


class Grouping(unittest.TestCase):
    def test_a_frame_gap_breaks_a_track(self):
        """An interrupted shard must not be linked across its own hole."""
        per_frame = {i: [node(i, 10.0 + 6 * i, 10.0)] for i in (0, 1, 3, 4)}
        tracks = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in per_frame})
        self.assertEqual(tracks, [], "a gap must not be bridged")
    def test_two_sequences_whose_windows_touch_are_not_merged(self):
        tool = load_tool("ball_v3_train")
        records = []
        for index in range(6):
            records.append({"clip_id": "c", "frame_index": index,
                            "sequence_id": "c#r00"})
        for index in range(6, 12):
            records.append({"clip_id": "c", "frame_index": index,
                            "sequence_id": "c#r01"})
        grouped, broken = tool.sequences_from_records(records)
        self.assertEqual(sorted(grouped), ["c#r00", "c#r01"])
        self.assertEqual(broken, [])
        self.assertEqual(len(grouped["c#r00"]), 6)

    def test_an_interrupted_sequence_is_reported(self):
        tool = load_tool("ball_v3_train")
        records = [{"clip_id": "c", "frame_index": i, "sequence_id": "c#r00"}
                   for i in (5, 6, 8, 9)]
        grouped, broken = tool.sequences_from_records(records)
        self.assertEqual(broken, ["c#r00"])
        self.assertEqual(len(grouped["c#r00"]), 4)


class FeaturesAreNeverZeroByAbsence(unittest.TestCase):
    def test_the_vector_matches_the_declared_names(self):
        per_frame = {i: [complete(node(i, 10.0 + 6 * i, 10.0))] for i in range(5)}
        track = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in range(5)})[0]
        features = btrk.tracklet_features(track, window=5, camera_shift={})
        self.assertEqual(len(features), len(btrk.FEATURE_NAMES))

    def test_unset_caller_fields_are_detected(self):
        per_frame = {i: [node(i, 10.0 + 6 * i, 10.0)] for i in range(5)}
        track = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in range(5)})[0]
        # the caller-field check the training builder drops on
        self.assertEqual(btrk.node_gaps(track, include_body=False),
                         ["appearance", "brightness_score", "chroma_score",
                          "radius_over_short_side"])
        # and separately: nobody recorded pose availability, which is a gap too
        self.assertEqual(btrk.node_gaps(track),
                         ["appearance", "body_pose", "brightness_score",
                          "chroma_score", "radius_over_short_side"])
        self.assertIsNone(btrk.body_coverage(track))
        for one in track.nodes:
            complete(one)
        self.assertEqual(btrk.node_gaps(track), [])
        self.assertEqual(btrk.body_coverage(track), 1.0)

    def test_build_tracklets_fills_rank_and_competitor(self):
        per_frame = {0: [node(0, 10.0, 10.0, probability=0.7),
                         node(0, 90.0, 90.0, probability=0.3),
                         node(0, 150.0, 20.0, probability=0.1)],
                     1: [node(1, 16.0, 10.0, probability=0.8)],
                     2: [node(2, 22.0, 10.0, probability=0.8)]}
        track = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in per_frame},
                                     pool_sizes={0: 240})[0]
        first = track.nodes[0]
        self.assertEqual(first.rank, 0)
        self.assertEqual(first.frame_nodes, 3)
        self.assertEqual(first.pool_size, 240)
        self.assertAlmostEqual(first.competitor_probability, 0.3, places=6)

    def test_cycle_agreement_is_measured_not_assumed(self):
        """A straight track round-trips; an inconsistent one does not.

        ``build_tracklets`` is not the way to test this: with a link tolerance of
        eight radii, almost any track that survives linking is smooth enough to
        round-trip, which is exactly the weakness the module documents. The
        measurement is tested directly on hand-built geometry instead.
        """
        straight = btrk.Tracklet(nodes=[node(i, 10.0 + 6.0 * i, 10.0)
                                        for i in range(5)])
        self.assertEqual(btrk._cycle_agreement(straight, {}, {}, btrk.LINK_RADII),
                         1.0)

        # the third node is 50 px off the line the next step implies; the link
        # radius at radius 5 is 40 px, so this fails the reverse check
        bent_nodes = [node(i, 10.0 + 6.0 * i, 10.0) for i in range(5)]
        bent_nodes[1].y = 90.0
        bent = btrk.Tracklet(nodes=bent_nodes)
        self.assertLess(btrk._cycle_agreement(bent, {}, {}, btrk.LINK_RADII), 1.0)

    def test_a_link_never_crosses_a_frame_gap(self):
        """Bridging a hole in the input would manufacture continuity."""
        per_frame = {0: [node(0, 10.0, 10.0)], 1: [node(1, 16.0, 10.0)],
                     3: [node(3, 28.0, 10.0)], 4: [node(4, 34.0, 10.0)]}
        tracks = btrk.build_tracklets(per_frame, {i: (0.0, 0.0) for i in per_frame},
                                      min_length=2)
        for track in tracks:
            frames = [n.frame_index for n in track.nodes]
            self.assertEqual(frames, list(range(frames[0], frames[0] + len(frames))),
                             f"track spans a gap: {frames}")

    def test_club_features_are_declared_unavailable_not_zero(self):
        self.assertIn("club_lock", btrk.UNAVAILABLE_FEATURES)
        self.assertNotIn("club_lock", btrk.FEATURE_NAMES)
        self.assertTrue(btrk.UNAVAILABLE_FEATURES["club_lock"])

    def test_every_ablation_group_names_real_features(self):
        for group, names in btrk.MOTION_GROUPS.items():
            for name in names:
                self.assertIn(name, btrk.FEATURE_NAMES,
                              f"{group} names {name}, which is not a feature")

    def test_dropping_a_group_zeroes_exactly_that_group(self):
        mask = btrk.mask_for_ablation(["body"])
        for position, name in enumerate(btrk.FEATURE_NAMES):
            expected = 0.0 if name in btrk.MOTION_GROUPS["body"] else 1.0
            self.assertEqual(mask[position], expected, name)
        # support_fraction is in no group: it is a count, not a behaviour
        self.assertEqual(mask[btrk.FEATURE_NAMES.index("support_fraction")], 1.0)


class Splits(unittest.TestCase):
    def test_a_family_holding_a_ranker_holdout_clip_is_held_out(self):
        tool = load_tool("ball_v3_train")
        records = tool.source_records()
        train_ids = {r["clip_id"] for r in json.loads(
            tool.V2_SPLIT.read_text("utf-8"))["clips"] if r["split"] == "train"}
        # Family isolation depends on committed metadata, not private video
        # availability. Exercise the real roster even in a fresh clone.
        clips = [r for ident, r in records.items() if ident in train_ids]
        self.assertTrue(clips)
        by_family: dict[str, list[str]] = {}
        for clip in clips:
            by_family.setdefault(clip["source_group"], []).append(clip["clip_id"])
        family, members = max(by_family.items(), key=lambda item: len(item[1]))
        assignment = tool.family_split([c["clip_id"] for c in clips], [members[0]])
        for member in members:
            self.assertEqual(assignment[member], "validation",
                             f"{member} shares a shoot family with {members[0]}")
        for clip_id, label in assignment.items():
            if label == "train":
                self.assertNotEqual(
                    tool.source_records()[clip_id]["source_group"], family)

    def test_the_trainer_refuses_an_evaluation_clip(self):
        tool = load_tool("ball_v3_train")
        sealed = sorted(tool.eval_clip_ids())
        self.assertTrue(sealed)
        with self.assertRaises(SystemExit):
            tool.assert_not_evaluation(sealed)
        # and it is happy with the clips it is actually allowed to use
        tool.assert_not_evaluation([c["clip_id"] for c in tool.train_clips()])

    def test_the_sealed_list_is_exactly_the_frozen_final_test(self):
        tool = load_tool("ball_v3_train")
        frozen = {c["clip_id"] for c in json.loads(
            (ROOT / "data/ball_dataset_v3_eval/split.json").read_text("utf-8"))["clips"]
            if c["split"] == "final_test"}
        self.assertEqual(tool.sealed_clip_ids(), frozen)

    def test_the_trainer_reads_only_split_metadata_from_the_eval_corpus(self):
        """It must be able to name the sealed clips and nothing more.

        The exclusion list has to come from the frozen split, so the path is
        legitimate; what must stay impossible is opening that corpus's *media* or
        sources from training code.
        """
        source = (ROOT / "tools" / "ball_v3_train.py").read_text("utf-8")
        self.assertIn("ball_dataset_v3_eval", source)
        self.assertNotIn("sources.json", source.split("V3_EVAL_SPLIT")[1][:200],
                         "only split.json may be read from the eval corpus")
        self.assertNotIn("video_assets", source)


class TrackerEmitsNoStaleCoordinate(unittest.TestCase):
    """Invariants that hold for any ranker output, so they can be pinned here."""

    def _result(self, frames, indices):
        from ghostcaddie.video import ball_v3_tracker as btrack
        device = brank.device()
        model = brank.CandidateRanker().to(device)
        return btrack.track_ball_v3(
            frames, indices, ranker=model,
            proposal_params=bp.ProposalParams(work_max_side=480, max_per_frame=20,
                                              per_method_limit=16),
            params=btrack.TrackerParams(top_k=4, min_support=3))

    def test_no_coordinate_outside_an_observed_frame(self):
        import cv2
        frames = []
        for i in range(6):
            frame = turf(seed=21)
            cv2.circle(frame, (60 + 14 * i, 90), 6, (250, 250, 250), -1,
                       lineType=cv2.LINE_AA)
            frames.append(frame)
        result = self._result(frames, list(range(6)))
        for state in result.states:
            if state.state == "OBSERVED":
                self.assertIsNotNone(state.x)
                self.assertIsNotNone(state.y)
                self.assertIsNotNone(state.radius)
                self.assertFalse(state.interpolated,
                                 "interpolation must never produce OBSERVED")
            else:
                self.assertIsNone(state.x, state.state)
                self.assertIsNone(state.y, state.state)
                self.assertIsNone(state.radius, state.state)

    def test_observed_always_clears_the_margin(self):
        """A near-tie is not an observation, whatever the model's scale is."""
        from ghostcaddie.video import ball_v3_tracker as btrack
        import cv2
        frames = []
        for i in range(6):
            frame = turf(seed=31)
            cv2.circle(frame, (50, 90), 6, (250, 250, 250), -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(frame, (150, 90), 6, (252, 250, 250), -1,
                       lineType=cv2.LINE_AA)
            frames.append(frame)
        result = self._result(frames, list(range(6)))
        for state in result.states:
            if state.state == "OBSERVED":
                self.assertIsNotNone(state.margin)
                self.assertIsNotNone(state.probability)
                self.assertGreaterEqual(float(state.margin),
                                        btrack.TrackerParams().min_margin)
                self.assertGreaterEqual(float(state.probability),
                                        btrack.TrackerParams().min_probability)

    def test_a_perfectly_static_scene_is_rejected_as_scenery(self):
        """Identical frames: every candidate is motionless after compensation,
        which is the definition of scenery rather than a played ball."""
        frames = [turf(seed=11) for _ in range(5)]
        result = self._result(frames, list(range(5)))
        self.assertFalse(result.accepted)
        self.assertTrue(any("static" in reason
                            for reason in result.rejection_reasons),
                        result.rejection_reasons)
        for state in result.states:
            self.assertNotEqual(state.state, "OBSERVED")
            self.assertIsNone(state.x)

    def test_a_cpu_discriminator_works_on_any_tracker_device(self):
        """``load_discriminator`` returns a CPU model; the tracker runs on MPS.

        Sending an MPS tensor into a CPU convolution raises, and this pass hit
        exactly that: every discriminator validation run failed. The feature
        tensor is now placed on the discriminator's own device.
        """
        from ghostcaddie.video import ball_v3_tracker as btrack
        model = btrk.TrackletDiscriminator()
        features = [0.5] * len(btrk.FEATURE_NAMES)
        probability = btrack._track_probability(model, features,
                                                brank.device())
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_the_stated_heuristic_is_used_when_there_is_no_model(self):
        """An ablation with the discriminator removed must not silently return 1."""
        from ghostcaddie.video import ball_v3_tracker as btrack
        features = [0.0] * len(btrk.FEATURE_NAMES)
        named = dict(zip(btrk.FEATURE_NAMES, features))
        named["mean_rank_probability"] = 0.9
        named["top1_fraction"] = 1.0
        named["static_fraction"] = 0.0
        values = [named[name] for name in btrk.FEATURE_NAMES]
        self.assertAlmostEqual(
            btrack._track_probability(None, values, None), 0.95, places=6)

    def test_a_rejected_track_never_emits_a_coordinate(self):
        frames = [turf(seed=11) for _ in range(4)]
        result = self._result(frames, list(range(4)))
        self.assertFalse(result.accepted)
        self.assertTrue(result.rejection_reasons)
        for state in result.states:
            self.assertNotEqual(state.state, "OBSERVED")
            self.assertIsNone(state.x)


class AblationMaskIsPartOfTheCheckpoint(unittest.TestCase):
    """A dropped column has random first-layer weights.

    Training with a mask and then deploying without one multiplies those random
    weights by live feature values. For the raw 0-255 intensity group that was
    measured to collapse validation AUC from 0.9713 to 0.5612. These tests exist
    so a checkpoint cannot be run in a configuration it was not trained in.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_saved_ablation_checkpoint_carries_its_mask(self):
        import torch  # noqa: F401  (import guard: torch is a hard requirement here)
        mask = btrk.mask_for_ablation(["body"])
        model = btrk.TrackletDiscriminator(mask=mask)
        path = self.tmp / "ablation.pt"
        btrk.save_discriminator(model, path)
        loaded = btrk.load_discriminator(path)
        self.assertTrue(np.array_equal(btrk.discriminator_mask(loaded), mask))
        self.assertEqual(sorted(btrk.masked_feature_names(loaded)),
                         sorted(btrk.dropped_feature_names(["body"])))

    def test_the_forward_pass_applies_the_mask(self):
        """``model(x) == model(x * mask)`` for a loaded ablation."""
        import torch
        mask = btrk.mask_for_ablation(["motion"])
        model = btrk.TrackletDiscriminator(mask=mask).eval()
        path = self.tmp / "motion.pt"
        btrk.save_discriminator(model, path)
        loaded = btrk.load_discriminator(path)
        torch.manual_seed(0)
        x = torch.rand(6, len(btrk.FEATURE_NAMES))
        tensor_mask = torch.from_numpy(mask)
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), loaded(x * tensor_mask))

    def test_a_dropped_column_cannot_change_the_answer(self):
        """Arbitrary values in dropped columns must be inert, not merely small."""
        import torch
        mask = btrk.mask_for_ablation(["appearance"])
        model = btrk.TrackletDiscriminator(mask=mask).eval()
        torch.manual_seed(1)
        x = torch.rand(5, len(btrk.FEATURE_NAMES))
        altered = x.clone()
        for name in btrk.dropped_feature_names(["appearance"]):
            altered[:, btrk.FEATURE_NAMES.index(name)] = 255.0
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(altered))

    def test_a_kept_column_still_changes_the_answer(self):
        """The mask must not be a blanket constant: a live column still matters."""
        import torch
        model = btrk.TrackletDiscriminator(mask=btrk.mask_for_ablation(["body"]))
        model.eval()
        torch.manual_seed(2)
        x = torch.rand(3, len(btrk.FEATURE_NAMES))
        altered = x.clone()
        altered[:, btrk.FEATURE_NAMES.index("mean_rank_probability")] = 9.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(x), model(altered)))

    def test_the_mask_moves_with_the_model_device(self):
        import torch
        mask = btrk.mask_for_ablation(["chroma"])
        model = btrk.load_discriminator(self._saved(mask))
        moved = model.to("cpu")
        self.assertEqual(moved.feature_mask.device.type, "cpu")
        x = torch.rand(2, len(btrk.FEATURE_NAMES))
        with torch.no_grad():
            moved(x)
        self.assertTrue(np.array_equal(btrk.discriminator_mask(moved), mask))

    def test_a_legacy_mask_less_checkpoint_loads_as_a_full_model(self):
        """Backwards compatibility, stated rather than assumed.

        A bare ``state_dict`` records no mask, so it can only mean "full model".
        An ablation saved that way must be migrated, not loaded and hoped about.
        """
        import torch
        model = btrk.TrackletDiscriminator()
        path = self.tmp / "legacy.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                    if k != "feature_mask"}, path)
        loaded = btrk.load_discriminator(path)
        self.assertEqual(btrk.masked_feature_names(loaded), [])
        self.assertTrue(np.allclose(btrk.discriminator_mask(loaded), 1.0))

    def test_migrating_an_ablation_checkpoint_records_its_mask(self):
        import torch
        model = btrk.TrackletDiscriminator()
        source = self.tmp / "legacy_nobody.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                    if k != "feature_mask"}, source)
        target = self.tmp / "nobody.pt"
        record = btrk.migrate_ablation_checkpoint(source, target, ["body"])
        self.assertTrue(record["migrated"])
        loaded = btrk.load_discriminator(target)
        self.assertEqual(sorted(btrk.masked_feature_names(loaded)),
                         sorted(btrk.dropped_feature_names(["body"])))
        # the weights are untouched: migration records the mask, it does not retrain
        self.assertTrue(np.array_equal(
            loaded.net[0].weight.detach().cpu().numpy(),
            model.net[0].weight.detach().cpu().numpy()))

    def _saved(self, mask) -> Path:
        path = self.tmp / "saved.pt"
        btrk.save_discriminator(btrk.TrackletDiscriminator(mask=mask), path)
        return path


class TieCorrectAuc(unittest.TestCase):
    """The estimator that produced the first report's AUC table rewarded collapse.

    It ranked with ``argsort`` over positives-then-negatives and assigned distinct
    ranks, so every tie was broken in the positive class's favour. A model whose
    output had collapsed to a constant scored 0.674 where the correct answer is
    0.5, and the no-motion ablation read 0.9820 against a tie-correct 0.9636.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool("ball_v3_train")

    def test_a_constant_score_is_chance(self):
        scores = np.full(20, 0.5)
        labels = np.r_[np.ones(7), np.zeros(13)]
        self.assertAlmostEqual(self.tool._auc(scores, labels), 0.5, places=9)

    def test_perfect_separation_is_one(self):
        scores = np.r_[np.linspace(0.6, 1.0, 5), np.linspace(0.0, 0.5, 5)]
        labels = np.r_[np.ones(5), np.zeros(5)]
        self.assertAlmostEqual(self.tool._auc(scores, labels), 1.0, places=9)

    def test_a_reversed_ranking_is_zero(self):
        scores = np.r_[np.linspace(0.0, 0.4, 5), np.linspace(0.5, 1.0, 5)]
        labels = np.r_[np.ones(5), np.zeros(5)]
        self.assertAlmostEqual(self.tool._auc(scores, labels), 0.0, places=9)

    def test_mixed_ties_score_between(self):
        """A tie across the class boundary is worth exactly half.

        Positives are at scores 1.0, 0.5, 0.0 and negatives likewise. Of the nine
        pairs: three wins, three losses and three ties -- so 0.5, which is the
        number a model with no resolution should get.
        """
        scores = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.tool._auc(scores, labels), 0.5, places=9)
        self.assertEqual(self.tool._auc_ties(scores), 6)

    def test_permutation_invariance(self):
        rng = np.random.default_rng(3)
        scores = np.r_[rng.normal(0.7, 0.1, 12), rng.normal(0.4, 0.2, 18)]
        labels = np.r_[np.ones(12), np.zeros(18)]
        reference = self.tool._auc(scores, labels)
        for seed in (1, 2, 3, 4):
            order = np.random.default_rng(seed).permutation(len(scores))
            self.assertAlmostEqual(
                self.tool._auc(scores[order], labels[order]), reference, places=12)

    def test_ties_are_counted_for_the_report(self):
        self.assertEqual(self.tool._auc_ties(np.array([1.0, 1.0, 2.0, 3.0])), 2)
        self.assertEqual(self.tool._auc_ties(np.array([1.0, 2.0, 3.0])), 0)


class BodyMissingnessIsRecordedNotInferred(unittest.TestCase):
    """A zero body vector is not evidence that the pose was missing.

    A candidate genuinely far from every landmark produces zeros from real
    geometry. Only an explicit availability flag separates "not near the body"
    from "nobody looked", and 241 of 776 tracklets turned out to have no pose
    anywhere while reporting no node gaps at all.
    """

    def _track(self, count: int, available) -> btrk.Tracklet:
        """A real track, so rank/frame_nodes are filled the way the builder fills
        them, with pose availability set explicitly afterwards."""
        per_frame = {i: [complete(node(i, 10.0 + i, 20.0))] for i in range(count)}
        track = btrk.build_tracklets(
            per_frame, {i: (0.0, 0.0) for i in per_frame})[0]
        for one in track.nodes:
            one.body_available = available
        return track

    def test_a_zero_body_vector_is_not_a_gap_when_pose_was_available(self):
        track = self._track(4, True)
        for one in track.nodes:
            one.inside_torso = 0.0
            one.distance_to_hands = 0.0
        self.assertEqual(btrk.node_gaps(track), [])
        self.assertEqual(btrk.body_coverage(track), 1.0)

    def test_a_missing_pose_is_a_gap_that_the_builder_does_not_drop_on(self):
        track = self._track(4, False)
        # the caller fields are all present, so the training builder keeps it...
        self.assertEqual(btrk.node_gaps(track, include_body=False), [])
        # ...but the audit can see the pose was missing
        self.assertEqual(btrk.node_gaps(track), ["body_pose"])
        self.assertEqual(btrk.body_coverage(track), 0.0)

    def test_unrecorded_availability_is_unknown_not_zero(self):
        track = self._track(3, None)
        self.assertIsNone(btrk.body_coverage(track))
        self.assertEqual(btrk.node_gaps(track), ["body_pose"])


class LazyMotionIsUnavailableNotZero(unittest.TestCase):
    """A statistic nobody computed must not read as a measured zero."""

    def test_lazy_contexts_report_no_motion_statistic(self):
        from ghostcaddie.video import ball_candidates as bc
        frames = [turf(seed=seed) for seed in range(4)]
        _contexts, info = bc.build_contexts(frames, list(range(4)), motion="lazy")
        self.assertEqual(info["motion_mode"], "lazy")
        self.assertIsNone(info["median_camera_motion_fraction"],
                          "lazy motion must report unavailable, not 0.0")

    def test_eager_contexts_still_report_the_statistic(self):
        from ghostcaddie.video import ball_candidates as bc
        frames = [turf(seed=seed) for seed in range(4)]
        _contexts, info = bc.build_contexts(frames, list(range(4)), motion="eager")
        self.assertIsNotNone(info["median_camera_motion_fraction"])


class LegacyGeneratorDefaultsAreUnchanged(unittest.TestCase):
    """The shared generator module is used by v1/v2 as well as v3.

    Its default must reproduce the pre-optimisation behaviour; v3 opts into the
    new path explicitly. Before this, the default silently changed
    ``radial_symmetry`` enough that 0 of 16 candidate positions were shared with
    the original on one real frame.
    """

    def setUp(self) -> None:
        from ghostcaddie.video import ball_candidates as bc
        self.bc = bc
        self.context = bc.build_contexts([turf(seed=7)], [0])[0][0]

    def _positions(self, candidates):
        return sorted((round(c.x, 4), round(c.y, 4)) for c in candidates)

    def test_every_shared_generator_defaults_to_the_original_path(self):
        from ghostcaddie.video import ball_v3_proposals as bp
        import inspect
        for name in bp.PYRAMID_METHODS:
            function = bp.GENERATORS[name]
            default = inspect.signature(function).parameters["pyramid"].default
            self.assertFalse(default, f"{name} must default to pyramid=False")

    def test_the_default_matches_explicit_false(self):
        for name in ("radial_symmetry", "local_contrast", "brightness_cue"):
            function = self.bc.GENERATORS[name]
            self.assertEqual(
                self._positions(function(self.context, limit=16)),
                self._positions(function(self.context, limit=16, pyramid=False)),
                f"{name} default is not the original path")

    def test_v3_still_opts_into_the_shared_pyramid(self):
        """The one-line default change must not have disabled the optimisation."""
        from ghostcaddie.video import ball_v3_proposals as bp
        self.assertTrue(bp.ProposalParams().pyramid)
        for name in bp.PYRAMID_METHODS:
            self.assertIn(name, bp.GENERATORS,
                          f"{name} has no generator to opt in with")


class FrameViewsOnANonBlankFixture(unittest.TestCase):
    """The crop comparison has to run on an image with structure in it.

    The fixture (``tests/fixtures/frameviews_turf.png``, 320x240, ~90 KB,
    synthetic, regenerable with ``make_frameviews_fixture.py``) is committed so
    this regression does not depend on real media being present, and it is
    deliberately non-blank: on a blank frame every crop is the same crop.

    **No equivalence is claimed.** Review 1 measured the two paths on real crops
    at mean |delta pixel| 0.905/255 and 1.758/255, with the ranker's top-1
    flipping on 3 of 15 frames on one clip and 0 of 15 on the other: an
    approximation and a preprocess shift. What is asserted here is the structure
    that measurement left intact -- small crops identical, large crops bounded --
    not the real-image decision rate.
    """

    def frame(self):
        import cv2
        path = ROOT / "tests" / "fixtures" / "frameviews_turf.png"
        image = cv2.imread(str(path))
        self.assertIsNotNone(image, f"missing fixture {path}")
        return np.asarray(image, dtype=np.uint8)

    def test_the_fixture_is_not_blank(self):
        frame = self.frame()
        self.assertGreater(float(frame.std()), 20.0)
        self.assertGreater(len(np.unique(frame[..., 1])), 40)

    def test_small_crops_are_identical_and_large_crops_bounded(self):
        import cv2
        from ghostcaddie.video import ball_v3_context as bctx
        frame = self.frame()
        views = bctx.FrameViews(frame)
        for radius in (2.0, 3.0, 4.0):
            with self.subTest(radius=radius):
                direct = bctx.candidate_stack(frame, None, 150.0, 120.0, radius)
                shared = views.stack(150.0, 120.0, radius)
                np.testing.assert_array_equal(direct, shared)
        for x, y, radius in ((150.0, 120.0, 40.0), (52.0, 190.0, 6.0),
                             (268.0, 62.0, 9.0), (60.0, 200.0, 2.0)):
            with self.subTest(radius=radius):
                direct = bctx.candidate_stack(frame, None, x, y, radius)
                shared = views.stack(x, y, radius)
                self.assertEqual(direct.shape, shared.shape)
                delta = np.abs(direct.astype(np.int16) - shared.astype(np.int16))
                self.assertLess(float(delta.mean()), 4.0,
                                f"mean channel delta {delta.mean()} at {radius}")
        # a residual, where the fixture has one, must be carried into both paths
        views_with_motion = bctx.FrameViews(frame, np.full(frame.shape[:2], 12.0,
                                                          np.float32))
        with_motion = views_with_motion.stack(150.0, 120.0, 40.0)
        without = views.stack(150.0, 120.0, 40.0)
        self.assertGreater(float(np.abs(with_motion[..., 6].astype(np.int16)
                                       - without[..., 6].astype(np.int16)).max()),
                           0.0, "the motion channel must not be ignored")


class ExtractionCacheReuse(unittest.TestCase):
    """A reused extraction must decide exactly like a freshly built one.

    The whole eight-variant plan rests on ``--extraction-cache``: if a cache hit
    changed a single state, every variant after the first would be measuring
    something else. This is the test that was missing when the cache key bound
    only the clip id, the first index and a count -- a different stride with the
    same endpoints, or a different ``top_k``, reused the same file.
    """

    def setUp(self) -> None:
        import torch
        from ghostcaddie.video import ball_v3_tracker as btrack
        self.btrack = btrack
        self.tool = load_tool("ball_v3_validate")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.media = self.root / "synthetic_source.bin"
        self.media.write_bytes(b"not a real video: the fingerprint binds its bytes")
        self.indices = list(range(6))
        self.frames = []
        for i in range(6):
            import cv2
            frame = turf(seed=41)
            cv2.circle(frame, (60 + 12 * i, 90), 5, (250, 250, 250), -1,
                       lineType=cv2.LINE_AA)
            self.frames.append(frame)
        torch.manual_seed(0)
        self.ranker = brank.CandidateRanker().to(brank.device())
        self.proposal_params = bp.ProposalParams(work_max_side=480,
                                                 max_per_frame=20,
                                                 per_method_limit=16)
        self.tracker_params = btrack.TrackerParams(top_k=4, min_support=3)
        self.ranker_sha = "e" * 64

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self) -> dict:
        return {"clip_id": "synthetic_cache_clip", "media": self.media,
                "indices": self.indices, "labels": {}, "labelled": False,
                "family": "synthetic"}

    def _extraction(self, cache_root):
        return self.tool._load_or_build_extraction(
            cache_root, self._run(), self.indices, self.frames, self.ranker, {},
            self.proposal_params, self.tracker_params, self.ranker_sha)

    def _fingerprint(self, **change) -> str:
        """The fingerprint the validator computes for this call, optionally altered."""
        arguments = dict(
            source_sha256=self.btrack.file_sha256(self.media),
            indices=self.indices, ranker_sha256=self.ranker_sha,
            proposal_params=self.proposal_params,
            tracker_params=self.tracker_params, body_mode="nobody",
            poses_digest=self.btrack.pose_digest({}))
        arguments.update(change)
        return self.btrack.extraction_fingerprint(**arguments)["fingerprint"]

    def _states(self, extraction) -> list:
        result = self.btrack.track_ball_v3(
            self.frames, self.indices, ranker=self.ranker, poses={},
            params=self.tracker_params, proposal_params=self.proposal_params,
            extraction=extraction)
        return [(s.frame_index, s.state, s.x, s.y, s.radius)
                for s in result.states]

    def test_a_reused_extraction_decides_identically(self):
        direct, reused_direct = self._extraction(None)
        built, reused_built = self._extraction(self.root)
        again, reused_again = self._extraction(self.root)
        self.assertFalse(reused_direct)
        self.assertFalse(reused_built)
        self.assertTrue(reused_again, "the second call must come off disk")
        self.assertEqual(direct["frames"], built["frames"])
        # the tracker's decision is the thing that matters, not just the fields
        self.assertEqual(self._states(direct), self._states(built))
        self.assertEqual(self._states(direct), self._states(again))
        self.assertTrue(self._states(direct), "no states at all: vacuous test")

    def test_the_fingerprint_moves_when_anything_relevant_moves(self):
        base = self._fingerprint()
        variants = {
            "a different stride": dict(indices=[1, 2, 3, 4, 5, 7]),
            "a different top_k": dict(
                tracker_params=self.btrack.TrackerParams(top_k=8)),
            "a different proposal param": dict(
                proposal_params=bp.ProposalParams(pyramid=False)),
            "a different ranker": dict(ranker_sha256="f" * 64),
            "a different source": dict(source_sha256="a" * 64),
            "body inputs": dict(body_mode="body"),
            "a different pose set": dict(poses_digest="1" * 64),
        }
        for name, change in variants.items():
            with self.subTest(change=name):
                self.assertNotEqual(self._fingerprint(**change), base, name)

    def test_a_mismatched_cache_entry_is_refused_not_reused(self):
        """The file's recorded fingerprint is checked, not just its name."""
        fingerprint = self._fingerprint()
        path = self.root / f"synthetic_cache_clip__{fingerprint[:32]}.json"
        path.write_text(json.dumps({"version": self.btrack.EXTRACTION_VERSION,
                                    "body_inputs": False,
                                    "fingerprint": "deadbeef",
                                    "frames": {}, "shifts": {}, "info": {},
                                    "timings": {}}), encoding="utf-8")
        self.assertTrue(path.exists())
        with self.assertRaises(SystemExit) as caught:
            self._extraction(self.root)
        self.assertIn("refusing extraction cache", str(caught.exception))

    def test_an_unrelated_cache_entry_is_not_consulted(self):
        """A file written under a different key must not be a hit at all."""
        (self.root / "synthetic_cache_clip__deadbeefdeadbeef.json").write_text(
            json.dumps({"version": self.btrack.EXTRACTION_VERSION,
                        "fingerprint": "deadbeef" * 8}), encoding="utf-8")
        extraction, reused = self._extraction(self.root)
        self.assertFalse(reused)
        self.assertIn("frames", extraction)


class CacheHitTimingIsInherited(unittest.TestCase):
    """A cache hit's extraction timings are the build run's, not this run's."""

    def test_the_tracker_marks_them_and_the_validator_separates_them(self):
        from ghostcaddie.video import ball_v3_tracker as btrack
        self.assertIn("motion_s", btrack.TIMING_KEYS_FROM_EXTRACTION)
        self.assertNotIn("association_s", btrack.TIMING_KEYS_FROM_EXTRACTION)
        source = (ROOT / "tools" / "ball_v3_validate.py").read_text("utf-8")
        self.assertIn("timings_inherited_extraction_build", source)
        self.assertIn("inherited_extraction_build_seconds", source)


if __name__ == "__main__":
    unittest.main()
