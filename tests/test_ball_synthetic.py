"""Synthetic ground-truth generator tests.

The generator is the foundation of this research pass, so its claims are tested
rather than assumed: the label must sit where the frame actually changed, motion
blur must follow speed, occlusion must change the state, an off-screen ball must
leave the frame untouched, an invisible insertion must be flagged instead of
quietly becoming a positive example, and nothing may be synthesized from a
validation or final-test background.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ghostcaddie.video import ball_synthetic as bs  # noqa: E402

SOURCES = ROOT / "data" / "ball_dataset_v2" / "sources.json"
SPLIT = ROOT / "data" / "ball_dataset_v2" / "split.json"


def grass_frame(width: int = 320, height: int = 240, seed: int = 3) -> np.ndarray:
    """A low-texture green background: measurable, and not a real clip."""
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 1] = 110 + rng.normal(0, 4, (height, width))
    base[..., 0] = 70 + rng.normal(0, 3, (height, width))
    base[..., 2] = 55 + rng.normal(0, 3, (height, width))
    return np.clip(base, 0, 255).astype(np.uint8)


def flat_white_frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 250, dtype=np.uint8)


class InsertionGeometryTests(unittest.TestCase):
    def test_the_label_sits_where_the_frame_actually_changed(self):
        frames = [grass_frame() for _ in range(6)]
        modified, samples = bs.synthesize_sequence(frames, seed=5,
                                                   forced_kind="linear")
        checked = 0
        for before, after, sample in zip(frames, modified, samples):
            if sample.visibility != "VISIBLE" or not sample.verified:
                continue
            measured = bs.measure_insertion(before, after, sample.x, sample.y,
                                            sample.radius_px)
            self.assertIsNotNone(measured["error_px"])
            self.assertLessEqual(
                measured["error_px"], max(2.5, sample.radius_px),
                "the changed pixels must be where the label says the ball is")
            checked += 1
        self.assertGreaterEqual(checked, 3, "expected some verified insertions")

    def test_labels_keep_sub_pixel_precision(self):
        frames = [grass_frame() for _ in range(6)]
        _, samples = bs.synthesize_sequence(frames, seed=9, forced_kind="linear")
        fractional = [s for s in samples if abs(s.x % 1) > 1e-6 or abs(s.y % 1) > 1e-6]
        self.assertTrue(fractional,
                        "a moving ball must produce sub-pixel positions, otherwise "
                        "the generator is rounding the answer it is supposed to test")

    def test_a_stationary_ball_stays_stationary(self):
        frames = [grass_frame() for _ in range(6)]
        _, samples = bs.synthesize_sequence(frames, seed=13, forced_kind="stationary")
        spread = max(abs(s.x - samples[0].x) for s in samples)
        self.assertLess(spread, 1.0)

    def test_an_offscreen_ball_leaves_the_frame_untouched_and_is_not_visible(self):
        frames = [grass_frame() for _ in range(6)]
        modified, samples = bs.synthesize_sequence(frames, seed=21,
                                                   forced_kind="entrance")
        offscreen = [i for i, s in enumerate(samples) if s.visibility == "OCCLUDED"]
        for index in offscreen:
            self.assertTrue(np.array_equal(frames[index], modified[index]),
                            "a frame whose label says OCCLUDED must not contain a "
                            "painted ball")

    def test_crop_labels_are_inside_the_crop(self):
        frame = grass_frame(200, 160)
        crop = bs.extract_crop(frame, 100.0, 80.0, 64)
        self.assertIsNotNone(crop)
        lx, ly = bs.crop_label(100.0, 80.0, 64)
        self.assertTrue(0 <= lx < 64 and 0 <= ly < 64)

    def test_a_crop_that_would_leave_the_frame_is_refused(self):
        frame = grass_frame(200, 160)
        self.assertIsNone(bs.extract_crop(frame, 2.0, 2.0, 64))


class AppearanceTests(unittest.TestCase):
    def test_motion_blur_grows_with_speed(self):
        slow, fast = [], []
        for seed in range(24):
            frames = [grass_frame() for _ in range(8)]
            modified, samples = bs.synthesize_sequence(frames, seed=seed)
            speed = bs.TrajectorySpec()
            blur = max(s.blur_px for s in samples)
            (fast if blur > 2.0 else slow).append(blur)
        self.assertTrue(slow and fast,
                        "the generator must produce both near-static and blurred "
                        f"balls; got slow={len(slow)} fast={len(fast)}")

    def test_blur_is_zero_for_a_stationary_ball(self):
        frames = [grass_frame() for _ in range(6)]
        _, samples = bs.synthesize_sequence(frames, seed=4, forced_kind="stationary")
        self.assertEqual(max(s.blur_px for s in samples), 0.0)

    def test_a_moving_ball_gets_blur_proportional_to_speed(self):
        frames = [grass_frame() for _ in range(8)]
        _, slow = bs.synthesize_sequence(frames, seed=4, forced_kind="slow")
        _, fast = bs.synthesize_sequence(frames, seed=4, forced_kind="fast")
        self.assertGreater(max(s.blur_px for s in slow), 0.0)
        self.assertGreater(max(s.blur_px for s in fast), max(s.blur_px for s in slow))

    def test_occlusion_changes_the_reported_state(self):
        frames = [grass_frame() for _ in range(14)]
        seen = set()
        for seed in range(30):
            _, samples = bs.synthesize_sequence(frames, seed=seed)
            seen.update(s.visibility for s in samples)
        self.assertIn("PARTIAL", seen)
        self.assertTrue(seen & {"PARTIAL", "OCCLUDED"})

    def test_ball_radius_is_in_the_tiny_object_range(self):
        rng = __import__("random").Random(7)
        for _ in range(40):
            appearance = bs.appearance_for(1080, rng)
            if 1080 * 0.004 > 12.0:
                continue
            self.assertGreaterEqual(appearance.radius_px, 2.0)
            self.assertLessEqual(appearance.radius_px, 1080 * 0.08 + 1e-6)

    def test_rendered_patch_is_not_a_hard_white_disc(self):
        appearance = bs.BallAppearance(radius_px=6.0, colour="white", shading=0.45,
                                       specular=0.6, shadow=0.0, noise=0.0,
                                       dimple=0.0)
        patch = bs.render_ball_patch(appearance)
        corner = patch[2, 2]
        self.assertEqual(corner[3], 0, "the patch must be transparent outside the ball")
        height, width = patch.shape[:2]
        interior = patch[height // 3:2 * height // 3, width // 3:2 * width // 3, :3]
        # spatial variation, not channel variation: a white ball has equal channels
        self.assertGreater(float(interior[:, :, 0].std()), 1.0,
                           "shading must vary across the ball, otherwise it is a "
                           "flat disc and the dataset is trivially separable")
        lit = patch[int(height * 0.36), int(width * 0.36), :3].mean()
        dark = patch[int(height * 0.64), int(width * 0.64), :3].mean()
        self.assertGreater(float(lit), float(dark))


class HonestyTests(unittest.TestCase):
    def test_every_label_declares_synthetic_ground_truth_origin(self):
        frames = [grass_frame() for _ in range(6)]
        _, samples = bs.synthesize_sequence(frames, seed=2)
        for sample in samples:
            label = sample.as_label()
            self.assertEqual(label["origin"], bs.ORIGIN_SYNTHETIC)
            self.assertIn(label["visibility"], bs.VISIBILITY_STATES)
            self.assertEqual(label["origin"], "SYNTHETIC_GROUND_TRUTH")

    def test_an_unmeasurable_insertion_is_flagged_not_kept_as_a_positive(self):
        """The rule, tested directly rather than through placement luck."""
        verified, reason = bs.classify_insertion(margin_px=3.0, error_px=0.5,
                                                 radius_px=4.0, changed_pixels=50)
        self.assertFalse(verified)
        self.assertIn("frame edge", reason)

        verified, reason = bs.classify_insertion(margin_px=200.0, error_px=41.0,
                                                 radius_px=6.0, changed_pixels=12)
        self.assertFalse(verified)
        self.assertIn("not measurable", reason)

        verified, reason = bs.classify_insertion(margin_px=200.0, error_px=None,
                                                 radius_px=6.0, changed_pixels=0)
        self.assertFalse(verified)

        verified, reason = bs.classify_insertion(margin_px=200.0, error_px=1.4,
                                                 radius_px=6.0, changed_pixels=90)
        self.assertTrue(verified)
        self.assertEqual(reason, "")

        # a macro ball is allowed a proportionally larger centroid error
        verified, _ = bs.classify_insertion(margin_px=400.0, error_px=20.0,
                                            radius_px=25.0, changed_pixels=2000)
        self.assertTrue(verified)

    def test_insertions_verified_in_a_normal_frame_carry_their_error(self):
        frames = [grass_frame() for _ in range(10)]
        _, samples = bs.synthesize_sequence(frames, seed=12, forced_kind="slow")
        verified = [s for s in samples if s.visibility == "VISIBLE" and s.verified]
        self.assertTrue(verified)
        for sample in verified:
            self.assertIsNotNone(sample.centroid_error_px)
            self.assertLessEqual(sample.centroid_error_px, max(2.0, sample.radius_px))

    def test_the_force_kind_argument_is_respected(self):
        frames = [grass_frame() for _ in range(6)]
        _, samples = bs.synthesize_sequence(frames, seed=3, forced_kind="stationary")
        self.assertLess(max(abs(s.x - samples[0].x) for s in samples), 1.0)
        _, moving = bs.synthesize_sequence(frames, seed=3, forced_kind="linear")
        self.assertGreater(max(abs(s.x - moving[0].x) for s in moving), 1.0)


class SplitIsolationTests(unittest.TestCase):
    def setUp(self):
        if not SPLIT.exists():
            self.skipTest("dataset split not present")
        self.split = json.loads(SPLIT.read_text())

    def test_synthesis_refuses_validation_and_final_test_clips(self):
        for split_name in ("validation", "final_test"):
            ids = [c["clip_id"] for c in self.split["clips"]
                   if c["split"] == split_name][:1]
            with self.assertRaises(ValueError) as ctx:
                bs.assert_training_clips(self.split, ids)
            self.assertIn("non-train clips", str(ctx.exception))

    def test_synthesis_accepts_train_clips(self):
        ids = [c["clip_id"] for c in self.split["clips"]
               if c["split"] == "train"][:3]
        bs.assert_training_clips(self.split, ids)

    def test_the_train_set_is_what_the_manifest_says(self):
        train = {c["clip_id"] for c in self.split["clips"] if c["split"] == "train"}
        self.assertEqual(len(train), 34)


class RealClipGenerationTests(unittest.TestCase):
    """End-to-end on a real train clip, if the (uncommitted) media is present."""

    def setUp(self):
        if not (SOURCES.exists() and SPLIT.exists()):
            self.skipTest("dataset not built")
        self.sources = json.loads(SOURCES.read_text())
        self.split = json.loads(SPLIT.read_text())
        self.train = [c for c in self.sources["clips"]
                      if c["clip_id"] in {x["clip_id"] for x in self.split["clips"]
                                          if x["split"] == "train"}]
        self.media_root = ROOT / "out" / "ball_dataset_v2" / "media"
        if not self.train or not (self.media_root / Path(
                self.train[0]["media_path"]).name).exists():
            self.skipTest("dataset media not present locally")

    def test_generates_labelled_crops_and_refuses_non_train_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "synthetic"
            manifest = bs.generate_synthetic_crops(
                train_clips=self.train[:2], media_root=self.media_root, out_dir=out,
                crop_size=48, sequences=2, sequence_length=6, seed=99)
            self.assertGreater(manifest["positives"], 0)
            self.assertGreater(manifest["negatives"], 0)
            self.assertEqual(manifest["origin"], bs.ORIGIN_SYNTHETIC)
            records = bs.load_crop_records(out / "crops.jsonl")
            self.assertEqual(len(records),
                             manifest["positives"] + manifest["negatives"])
            for record in records:
                if record["positive"]:
                    self.assertLessEqual(record["radius"], 64)
                    self.assertTrue(0 <= record["label_x"] <= 48)
                    self.assertTrue(0 <= record["label_y"] <= 48)
                    self.assertEqual(record["origin"], bs.ORIGIN_SYNTHETIC)
            images = list((out / "crops").glob("*.jpg"))
            self.assertEqual(len(images), len(records))

    def test_generation_never_touches_a_non_train_clip(self):
        other = [c for c in self.sources["clips"]
                 if c["clip_id"] not in {x["clip_id"] for x in self.split["clips"]
                                         if x["split"] == "train"}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bs.assert_training_clips(self.split, [other[0]["clip_id"]])


if __name__ == "__main__":
    unittest.main()
