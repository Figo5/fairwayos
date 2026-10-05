"""Bounded synthetic tests for the untrained structural putter-head detector.

These fixtures prove the GATES and the POLICY of ``ghostcaddie.demo.putter_detector``
on tiny drawn frames -- a compact head, a thin shaft, an oversized leg, a round
ball-like blob, a shaft with no head, two identical heads, a rotated head -- and prove
the runner's routing table. They prove nothing about accuracy on real clips, which this
unit tests do not measure.

Run::

    <venv with cv2>/bin/python -m unittest tests.test_putter_detector -v
"""
from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostcaddie.demo import putter_detector as pd  # noqa: E402
from ghostcaddie.demo import target_benchmark as bench  # noqa: E402
from ghostcaddie.demo.foreground_tracker import (  # noqa: E402
    METHOD_FOREGROUND, METHOD_REACQUIRED as FG_METHOD_REACQUIRED, ORIGIN_OPERATOR,
    REACQ_WINDOW_FRAMES as FG_REACQ_WINDOW_FRAMES, STATUS_LOST, STATUS_MISSING,
    STATUS_OBSERVED)

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV = True
except ImportError:                                            # pragma: no cover
    HAVE_CV = False

BACKGROUND = 40
HEAD_VALUE = 235
HEAD_W, HEAD_H = 11, 6
SHAFT_W, SHAFT_LEN = 2, 14
FRAME_H, FRAME_W = 90, 140


def empty_frame(level=BACKGROUND):
    rng = np.random.default_rng(11)
    frame = np.full((FRAME_H, FRAME_W, 3), level, dtype=np.int16)
    frame += rng.integers(-2, 3, size=frame.shape, dtype=np.int16)
    return np.clip(frame, 0, 255).astype(np.uint8)


def bright_background_frame():
    """A bright background: a DARK head has to be found on it, and a bright blob on it is
    not automatically a head."""
    return empty_frame(level=210)


def draw_head(frame, cx, cy, value=HEAD_VALUE):
    x0, y0 = cx - HEAD_W // 2, cy - HEAD_H // 2
    frame[y0:y0 + HEAD_H, x0:x0 + HEAD_W] = value


def draw_head_rotated(frame, cx, cy, angle, value=HEAD_VALUE):
    box = cv2.boxPoints(((float(cx), float(cy)), (float(HEAD_W), float(HEAD_H)), angle))
    cv2.fillPoly(frame, [np.round(box).astype(np.int32)], (value, value, value))


def draw_shaft_up(frame, cx, cy, gap=0, length=SHAFT_LEN, width=SHAFT_W, value=HEAD_VALUE):
    """A thin vertical line rising from the head's top edge (touching it by default)."""
    x0 = cx - width // 2
    top = cy - HEAD_H // 2
    frame[top - length - gap:top - gap, x0:x0 + width] = value


def draw_leg(frame, cx, cy, width=14, height=16, value=HEAD_VALUE):
    x0, y0 = cx - width // 2, cy - height // 2
    frame[y0:y0 + height, x0:x0 + width] = value


def draw_ball(frame, cx, cy, radius=5, value=HEAD_VALUE):
    cv2.circle(frame, (cx, cy), radius, (value, value, value), -1)


def head_and_shaft(cx=70, cy=40):
    frame = empty_frame()
    draw_head(frame, cx, cy)
    draw_shaft_up(frame, cx, cy)
    return frame


@unittest.skipUnless(HAVE_CV, "the structural detector needs numpy and cv2")
class StructuralDetectorTests(unittest.TestCase):
    def seed_of(self, frame, x, y):
        descriptor, reason, evidence = pd.describe_seed(frame, x, y)
        self.assertIsNone(reason, evidence)
        self.assertIsNotNone(descriptor, evidence)
        return descriptor

    def request(self, frame, anchor, radius=None):
        return pd.SearchRequest(frame, float(anchor[0]), float(anchor[1]),
                                pd.SEARCH_RADIUS_PX if radius is None else radius,
                                False, "test request")

    def decision(self, run, frame):
        for item in run.decisions:
            if item.source_frame == frame:
                return item
        self.fail(f"no decision for frame {frame}")

    # -- structural gates ------------------------------------------------

    def test_a_compact_head_with_an_attached_thin_shaft_is_detected(self):
        frame = head_and_shaft()
        seed = self.seed_of(frame, 70, 40)
        match, evidence = pd.detect(frame, self.request(11, (70, 40)), seed)
        self.assertTrue(match.found, evidence)
        self.assertEqual(match.transform, "head+shaft")
        self.assertGreaterEqual(match.score, pd.MIN_SCORE)
        self.assertLessEqual(abs(match.x - 70), 1.5, evidence["chosen"])
        self.assertLessEqual(abs(match.y - 40), 1.5, evidence["chosen"])
        chosen = evidence["chosen"]
        self.assertTrue(all(chosen["gates"].values()), chosen["gates"])
        self.assertTrue(chosen["shaft"]["supported"], chosen["shaft"])
        self.assertEqual(chosen["shaft"]["reason"],
                         "thin_structure_endpoint_attaches_to_head")
        self.assertGreater(chosen["shaft"]["extend_distance_px"],
                           chosen["shaft"]["attach_distance_px"])

    def test_a_rotated_head_is_detected_without_assumed_orientation(self):
        frame = empty_frame()
        draw_head_rotated(frame, 70, 40, 45.0)
        seed = self.seed_of(frame, 70, 40)
        match, evidence = pd.detect(frame, self.request(11, (70, 40)), seed)
        self.assertTrue(match.found, evidence)
        self.assertEqual(match.transform, "head-only")
        self.assertLessEqual(abs(match.x - 70), 2.0, evidence["chosen"])
        self.assertLessEqual(abs(match.y - 40), 2.0, evidence["chosen"])

    def test_a_larger_static_leg_is_refused_on_size_not_on_clothing_assumptions(self):
        seed = self.seed_of(head_and_shaft(), 70, 40)
        leg_frame = empty_frame()
        draw_leg(leg_frame, 70, 40)
        match, evidence = pd.detect(leg_frame, self.request(11, (70, 40)), seed)
        self.assertFalse(match.found)
        self.assertEqual(evidence["reason"], "no_admissible_component")
        self.assertTrue(evidence["components"])
        self.assertFalse(any(record["gates"]["size"] for record in evidence["components"]))
        self.assertIn("size", evidence["failed_gates"])

    def test_a_round_ball_like_distractor_is_refused(self):
        seed = self.seed_of(head_and_shaft(), 70, 40)
        ball_frame = empty_frame()
        draw_ball(ball_frame, 70, 40)
        match, evidence = pd.detect(ball_frame, self.request(11, (70, 40)), seed)
        self.assertFalse(match.found)
        record = evidence["components"][0]
        self.assertGreaterEqual(record["roundness"], pd.ROUNDNESS_MAX)
        self.assertFalse(record["gates"]["not_ball_like"], record)
        self.assertEqual(list(evidence["failed_gates"]), ["not_ball_like"])

    def test_a_thin_shaft_with_no_head_is_refused(self):
        seed = self.seed_of(head_and_shaft(), 70, 40)
        shaft_only = empty_frame()
        draw_shaft_up(shaft_only, 70, 60)
        match, evidence = pd.detect(shaft_only, self.request(11, (70, 50)), seed)
        self.assertFalse(match.found)
        self.assertEqual(evidence["components"], [])
        self.assertEqual(evidence["reason"], "no_admissible_component")

    def test_a_shaft_longer_than_the_seed_roi_still_reports_the_head_core(self):
        cx, cy = 70, 70
        frame = empty_frame()
        draw_head(frame, cx, cy)
        draw_shaft_up(frame, cx, cy, length=60)      # extends past the seed and search context
        seed = self.seed_of(frame, cx, cy)
        match, evidence = pd.detect(frame, self.request(11, (cx, cy)), seed)
        self.assertTrue(match.found, evidence)
        chosen = evidence["chosen"]
        origin = evidence["crop_origin_native"]
        blob_y = origin[1] + chosen["centroid_crop"][1]
        # Reconstructed head descriptors and the reported point both exclude the shaft.
        self.assertLessEqual(abs(blob_y - cy), 1.5, chosen)
        self.assertLessEqual(abs(match.y - cy), 1.5, chosen)
        self.assertLessEqual(abs(match.x - cx), 1.5, chosen)
        self.assertTrue(chosen["shaft"]["supported"], chosen["shaft"])
        run = pd.track_putter_sequence([(10, frame), (11, frame)], seed_frame=10,
                                       seed_x=cx, seed_y=cy, interval_frames=[10, 11])
        self.assertTrue(run.available, run.seed_evidence)
        self.assertEqual(run.decisions[-1].status, STATUS_OBSERVED)
        self.assertLessEqual(abs(run.decisions[-1].y - cy), 1.5)

    def test_connected_head_and_ball_cores_are_not_averaged_into_one_point(self):
        frame = empty_frame()
        draw_head(frame, 70, 40)
        draw_ball(frame, 88, 40)
        frame[39:41, 75:85] = HEAD_VALUE  # thin contact/shaft connects the regions
        analysis = pd._analyse(frame, pd.POLARITY_BRIGHTER)
        centers = [c['core_centroid_crop'] for c in analysis['components']]
        self.assertTrue(any(abs(x-70)<2 and abs(y-40)<2 for x,y in centers))
        self.assertTrue(any(abs(x-88)<2 and abs(y-40)<2 for x,y in centers))
        self.assertFalse(any(76<x<82 for x,y in centers), centers)

    def test_seed_context_dimensions_do_not_depend_on_coordinate_parity(self):
        frame = empty_frame()
        for x in (70, 71):
            crop, _, _ = pd._crop(frame, x, 40, 26.5)
            self.assertEqual(crop.shape[:2], (53, 53))

    def test_a_dark_head_is_detected_from_the_seed_polarity_not_from_brightness(self):
        frame = bright_background_frame()
        draw_head(frame, 70, 40, value=60)
        draw_shaft_up(frame, 70, 40, value=60)
        seed = self.seed_of(frame, 70, 40)
        self.assertEqual(seed["polarity"], pd.POLARITY_DARKER)
        self.assertLess(seed["contrast"], 0)
        match, evidence = pd.detect(frame, self.request(11, (70, 40)), seed)
        self.assertTrue(match.found, evidence)
        self.assertEqual(evidence["polarity"], pd.POLARITY_DARKER)
        self.assertLessEqual(abs(match.x - 70), 1.5, evidence["chosen"])
        self.assertTrue(evidence["chosen"]["shaft"]["supported"], evidence["chosen"])

    def test_a_bright_object_is_not_read_as_a_head_under_a_darker_seed(self):
        dark = bright_background_frame()
        draw_head(dark, 70, 40, value=60)
        seed = self.seed_of(dark, 70, 40)
        brighter = bright_background_frame()
        draw_head(brighter, 70, 40, value=250)
        match, evidence = pd.detect(brighter, self.request(11, (70, 40)), seed)
        self.assertFalse(match.found, evidence)
        self.assertEqual(evidence["reason"], "no_admissible_component")

    def test_two_identical_heads_abstain(self):
        frame = empty_frame()
        draw_head(frame, 61, 40)
        draw_head(frame, 79, 40)
        seed = self.seed_of(frame, 61, 40)
        match, evidence = pd.detect(frame, self.request(11, (61, 40)), seed)
        self.assertFalse(match.found)
        self.assertEqual(evidence["reason"], "ambiguous_two_identical_heads")
        self.assertEqual(len(evidence["candidates_top2"]), 2)

    def test_a_thin_shaft_on_only_one_of_two_identical_heads_selects_that_head(self):
        frame = empty_frame()
        draw_head(frame, 61, 40)
        draw_head(frame, 79, 40)
        draw_shaft_up(frame, 79, 40, gap=1)
        seed = self.seed_of(frame, 61, 40)
        match, evidence = pd.detect(frame, self.request(11, (61, 40)), seed)
        self.assertTrue(match.found, evidence)
        self.assertEqual(match.transform, "head+shaft")
        self.assertLessEqual(abs(match.x - 79), 1.5, evidence["chosen"])
        self.assertTrue(evidence["chosen"]["shaft"]["supported"])
        # The evidence term already separates the two scores here, so no ambiguity arose:
        # shaft support is worth more than the ambiguity margin (the next test checks that).
        self.assertEqual(match.reason, "")

    def test_shaft_support_separates_a_near_identical_pair(self):
        frame = empty_frame()
        draw_head(frame, 61, 40)
        draw_head(frame, 79, 40, value=228)     # near-identical, but no shaft of its own
        draw_shaft_up(frame, 79, 40, gap=1)
        seed = self.seed_of(frame, 61, 40)
        match, evidence = pd.detect(frame, self.request(11, (61, 40)), seed)
        self.assertTrue(match.found, evidence)
        self.assertEqual(match.transform, "head+shaft")
        self.assertLessEqual(abs(match.x - 79), 1.5, evidence["chosen"])
        # Shaft support is worth more than the ambiguity margin, so the pair is separated
        # by the score itself: there is no within-margin pair left to abstain on.
        self.assertEqual(match.reason, "")
        self.assertNotIn("candidates_top2", evidence)

    # -- seed ------------------------------------------------------------

    def test_a_seed_without_compact_head_evidence_is_refused_and_fails_closed(self):
        descriptor, reason, evidence = pd.describe_seed(empty_frame(), 70, 40)
        self.assertIsNone(descriptor)
        self.assertEqual(reason, "seed_has_no_compact_head_evidence")
        self.assertEqual(evidence["reason"], reason)
        frames = [(number, empty_frame()) for number in range(10, 16)]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 16)))
        self.assertFalse(run.available)
        self.assertEqual(run.reason, reason)
        self.assertEqual(run.summary()["automatic_observed_frames"], 0)
        seed_decision = run.decisions[0]
        self.assertEqual(seed_decision.origin, ORIGIN_OPERATOR)
        self.assertEqual(seed_decision.method, pd.METHOD_SEED)
        for decision in run.decisions[1:]:
            self.assertIn(decision.status, (STATUS_MISSING, STATUS_LOST))
            self.assertIsNone(decision.x)

    def test_the_seed_roi_is_read_only_and_measures_native_head_size(self):
        descriptor = self.seed_of(head_and_shaft(), 70, 40)
        self.assertEqual(descriptor["roi_px"], pd.SEED_ROI_PX)
        self.assertGreaterEqual(pd.SEED_ROI_PX, pd.SEED_ROI_MIN_PX)
        self.assertEqual(pd.SEED_ROI_PX, 53)  # fixed wider contrast context, no extra seed
        self.assertGreaterEqual(descriptor["equivalent_diam_px"], pd.HEAD_MIN_DIAM_PX)
        self.assertLessEqual(descriptor["equivalent_diam_px"], pd.HEAD_MAX_DIAM_PX)
        self.assertTrue(all(descriptor["gates"].values()), descriptor["gates"])

    # -- policy ----------------------------------------------------------

    def test_a_stationary_head_at_address_is_measured_frame_after_frame(self):
        frames = [(number, head_and_shaft()) for number in range(10, 27)]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 27)))
        self.assertTrue(run.available, run.seed_evidence)
        automatic = [d for d in run.decisions if d.origin == "automatic"]
        self.assertEqual(len(automatic), 16)
        self.assertTrue(all(d.method == pd.METHOD_DETECTOR for d in automatic))
        self.assertTrue(all(abs(d.x - 70) <= 1.5 and abs(d.y - 40) <= 1.5
                            for d in automatic), [d.to_point() for d in automatic])
        self.assertFalse(any(d.reacquired for d in automatic))
        # The inherited policy numbers the first segment 0 when the run is continuous from
        # the seed frame; it is a re-acquisition that opens the NEXT segment.
        self.assertTrue(all(d.segment == 0 for d in automatic))
        self.assertEqual(run.policy_summary["segments"], 1)

    def test_reappearance_needs_a_second_current_frame_and_never_backfills(self):
        frames = [(number, head_and_shaft()) for number in range(10, 31)
                  if number < 13 or number > 17]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 31)))
        self.assertEqual(self.decision(run, 13).status, STATUS_MISSING)
        self.assertEqual(self.decision(run, 14).status, STATUS_MISSING)
        self.assertEqual(self.decision(run, 15).status, STATUS_LOST)
        self.assertIsNone(self.decision(run, 15).x)
        self.assertEqual(self.decision(run, 18).status, STATUS_LOST)
        self.assertIsNone(self.decision(run, 18).x)
        self.assertIn("NOT be backfilled", self.decision(run, 18).note)
        confirmed = self.decision(run, 19)
        self.assertEqual(confirmed.status, STATUS_OBSERVED)
        self.assertTrue(confirmed.reacquired)
        self.assertEqual(confirmed.method, pd.METHOD_REACQUIRED)
        self.assertEqual(confirmed.segment, self.decision(run, 12).segment + 1)
        self.assertEqual(run.policy_summary["segments"], 2)
        self.assertGreater(confirmed.x, 0)
        after = self.decision(run, 20)
        self.assertEqual(after.status, STATUS_OBSERVED)
        self.assertFalse(after.reacquired)
        self.assertEqual(after.method, pd.METHOD_DETECTOR)

    def test_the_recovery_window_is_finite_in_elapsed_frames(self):
        frames = [(number, empty_frame() if 13 <= number < 40 else head_and_shaft())
                  for number in range(10, 61)]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 61)))
        self.assertEqual(self.decision(run, 15).status, STATUS_LOST)
        for frame in range(40, 61):
            decision = self.decision(run, frame)
            self.assertEqual(decision.status, STATUS_LOST, frame)
            self.assertIsNone(decision.x, frame)
        self.assertTrue(run.policy_summary["terminal_lost"])
        self.assertEqual(run.summary()["automatic_observed_frames"], 2)
        self.assertGreater(pd.REACQ_WINDOW_FRAMES, 10)
        self.assertEqual(pd.REACQ_WINDOW_FRAMES, 24)

    def test_a_structural_cut_resets_and_is_never_crossed(self):
        frames = [(number, head_and_shaft()) for number in range(10, 21)]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 21)),
                                       cut_frames=[15])
        self.assertEqual(self.decision(run, 15).status, STATUS_MISSING)
        self.assertIsNone(self.decision(run, 15).x)
        self.assertEqual(self.decision(run, 16).status, STATUS_MISSING)
        self.assertIsNone(self.decision(run, 16).x)
        self.assertTrue(run.policy_summary["structural_cut_reset"])

    # -- evidence, coordinates, profile ----------------------------------

    def test_coordinates_are_converted_from_the_crop_back_to_native_pixels(self):
        self.assertEqual(pd.to_native(4.0, 6.0, origin_x=100, origin_y=50),
                         (104.0, 56.0))
        self.assertEqual(pd.to_native(8.0, 12.0, origin_x=100, origin_y=50, scale=2.0),
                         (104.0, 56.0))
        with self.assertRaises(ValueError):
            pd.to_native(1.0, 1.0, origin_x=0, origin_y=0, scale=0.0)
        frame = head_and_shaft(cx=95, cy=60)
        seed = self.seed_of(frame, 95, 60)
        self.assertEqual(seed["polarity"], pd.POLARITY_BRIGHTER)
        # A non-unit measurement scale is REFUSED: this arm never resizes a crop, so it
        # must not divide coordinates for one either.
        with self.assertRaises(ValueError):
            pd.detect(frame, self.request(11, (95, 60)), seed, scale=2.0)
        match, evidence = pd.detect(frame, self.request(11, (95, 60)), seed)
        self.assertTrue(match.found, evidence)
        self.assertNotEqual(evidence["crop_origin_native"], [0, 0])
        chosen = evidence["chosen"]
        self.assertEqual(chosen["bbox_native"][0],
                         evidence["crop_origin_native"][0] + chosen["bbox_crop"][0])
        self.assertLessEqual(chosen["bbox_native"][0], 95)
        self.assertGreaterEqual(chosen["bbox_native"][0] + chosen["bbox_native"][2], 95)
        self.assertLessEqual(abs(match.x - 95), 1.5, chosen)
        self.assertLessEqual(abs(match.y - 60), 1.5, chosen)

    def test_the_raw_record_keeps_seed_boxes_shaft_geometry_gates_and_segments(self):
        frames = [(number, empty_frame() if 13 <= number < 17 else head_and_shaft())
                  for number in range(10, 25)]
        run = pd.track_putter_sequence(frames, seed_frame=10, seed_x=70, seed_y=40,
                                       interval_frames=list(range(10, 25)))
        self.assertEqual(run.seed_evidence["reason"],
                         "seed_compact_head_evidence_accepted")
        self.assertTrue(run.seed_evidence["chosen"]["bbox_native"])
        claimed = [raw for raw in run.raw_frames
                   if raw.get("evidence", {}).get("chosen")]
        self.assertTrue(claimed)
        chosen = claimed[0]["evidence"]["chosen"]
        for key in ("bbox_native", "point_native", "shaft", "gates", "terms", "score",
                    "seed_identity", "colour", "core_px"):
            self.assertIn(key, chosen)
        for key in ("near_end", "far_end", "attach_distance_px", "supported", "reason"):
            self.assertIn(key, chosen["shaft"])
        refused = [raw for raw in run.raw_frames
                   if raw.get("evidence", {}).get("reason") == "no_admissible_component"]
        self.assertEqual(len(refused), 4)
        self.assertTrue(all("failed_gates" in raw["evidence"] for raw in refused))
        self.assertEqual({raw["decision"]["segment"] for raw in run.raw_frames
                          if raw["decision"].get("segment") is not None}, {0, 1})

    def test_the_profile_is_fixed_declared_and_distinct_from_the_other_candidate(self):
        self.assertEqual(pd.PROFILE_NAME, "putter-structural-redetect-v1")
        self.assertEqual(pd.PROFILE["name"], pd.PROFILE_NAME)
        self.assertIsNone(pd.PROFILE["appearance_template"])
        self.assertFalse(pd.PROFILE["template_following"])
        self.assertFalse(pd.PROFILE["velocity_extrapolation"])
        self.assertFalse(pd.PROFILE["global_bottom_ban"])
        self.assertNotEqual(pd.METHOD_DETECTOR, METHOD_FOREGROUND)
        self.assertNotEqual(pd.METHOD_REACQUIRED, FG_METHOD_REACQUIRED)
        for method in (pd.METHOD_DETECTOR, pd.METHOD_REACQUIRED):
            self.assertIn(method, bench.AUTOMATIC_METHODS)
        for method in (METHOD_FOREGROUND, FG_METHOD_REACQUIRED):
            self.assertIn(method, bench.AUTOMATIC_METHODS)
        self.assertIsInstance(json.loads(bench.canonical_json(pd.PROFILE)), dict)
        policy = pd.PutterPolicy(seed_frame=10, seed_x=1.0, seed_y=1.0)
        self.assertEqual(policy.reacq_window_frames, pd.REACQ_WINDOW_FRAMES)
        self.assertNotEqual(policy.reacq_window_frames, FG_REACQ_WINDOW_FRAMES)
        self.assertEqual(policy.min_score, pd.MIN_SCORE)
        self.assertEqual(policy.loss_frames, pd.LOSS_FRAMES)
        self.assertEqual(policy.reacq_confirm_frames, pd.REACQ_CONFIRM_FRAMES)
        self.assertFalse(policy.wants_appearance_update(
            pd.CandidateMatch(11, True, x=1.0, y=1.0, score=1.0, seed_score=1.0)))


    def test_every_inference_constant_is_declared_in_the_profile(self):
        """The report can only declare what actually ran if PROFILE declares every constant.

        A constant is accepted either by its own declared key or by its value appearing in
        the profile (a value coincidence could mask an omission, which is why the named
        checks above also pin the critical ones). Imported status/origin names and the
        profile's own bookkeeping are exempt.
        """
        declared = json.loads(bench.canonical_json(pd.PROFILE))
        flat = []
        for value in declared.values():
            flat.extend(value if isinstance(value, list) else [value])
        exempt = {
            "PROFILE", "PROFILE_NAME", "LIMITATIONS", "METHOD_SEED", "METHOD_DETECTOR",
            "METHOD_REACQUIRED", "POLARITY_BOTH", "LINE_ANTIPODAL_PAIRS",
            "STATUS_OBSERVED", "STATUS_LOST", "STATUS_MISSING", "ORIGIN_AUTOMATIC",
            "ORIGIN_DERIVED",
        }
        missing = sorted(
            name for name, value in vars(pd).items()
            if name.isupper() and name not in exempt
            and not isinstance(value, (dict, tuple))
            and name.lower() not in declared and value not in flat)
        self.assertEqual(missing, [], f"inference constants missing from PROFILE: {missing}")
        for key in ("seed_pick_px", "seed_min_contrast", "seed_min_pixels", "local_window_px",
                    "bright_sigma", "bright_min_delta", "min_local_std", "metal_min_value",
                    "metal_max_sat", "metal_min_l", "head_core_dist_px", "identical_diam_px",
                    "identical_colour_min", "colour_min_score", "polarity_source"):
            self.assertIn(key, declared)


@unittest.skipUnless(HAVE_CV, "the structural detector needs numpy and cv2")
class RunnerRoutingTests(unittest.TestCase):
    """The candidate may only touch a putt clubhead; everything else is unchanged."""

    def setUp(self):
        import tools.tiger_target_benchmark as tool
        self.tool = tool

    def test_only_a_putt_clubhead_is_routed_to_the_candidate(self):
        engine_for = self.tool.engine_for
        self.assertEqual(engine_for("clubhead", "putter_redetect", stroke_type="putt"),
                         "putter")
        self.assertEqual(engine_for("ball", "putter_redetect", stroke_type="putt"),
                         "classic")
        self.assertEqual(engine_for("clubhead", "putter_redetect", stroke_type="full_swing"),
                         "classic")
        self.assertEqual(engine_for("clubhead", "putter_redetect"), "classic")
        self.assertEqual(engine_for("clubhead", "baseline", stroke_type="putt"), "classic")
        self.assertEqual(engine_for("clubhead", "clubhead_foreground",
                                    stroke_type="putt"), "foreground")
        self.assertEqual(engine_for("ball", "clubhead_foreground", stroke_type="putt"),
                         "classic")

    def test_the_variant_tables_and_option_overrides_stay_consistent(self):
        self.assertIn("putter_redetect", self.tool.SUPPORTED_VARIANTS)
        self.assertEqual(set(self.tool.VARIANT_OPTION_OVERRIDES),
                         set(self.tool.SUPPORTED_VARIANTS))
        self.assertEqual(self.tool.VARIANT_OPTION_OVERRIDES["putter_redetect"], {})
        self.assertEqual(self.tool.VARIANT_OPTION_OVERRIDES["baseline"], {})
        for variant in self.tool.SUPPORTED_VARIANTS:
            self.assertIn(variant, self.tool.VARIANT_NOTES)
        block = {"max_step_px": 12, "lk_refine": False,
                 "interpolation": {"enabled": True, "max_gap_frames": 3}}
        self.assertEqual(self.tool.effective_options(block, "putter_redetect"),
                         self.tool.effective_options(block, "baseline"))
        self.assertEqual(self.tool.effective_options(block, "putter_redetect")["max_step_px"],
                         12)

    def test_the_declared_candidate_methods_exist_in_their_modules(self):
        for variant, path in self.tool.CANDIDATE_MODULE_PATHS.items():
            module = importlib.import_module(path)
            for key, attr in self.tool.CANDIDATE_METHOD_ATTRS[variant].items():
                self.assertTrue(hasattr(module, attr), f"{variant}: {attr} ({key})")
                value = getattr(module, attr)
                if key == "seed_anchor":
                    # The operator placement is not an automatic method and never scores.
                    self.assertNotIn(value, bench.AUTOMATIC_METHODS)
                    self.assertEqual(value, pd.METHOD_SEED)
                else:
                    self.assertIn(value, bench.AUTOMATIC_METHODS)

    def test_the_runtime_record_declares_the_stroke_gate_and_the_detector_methods(self):
        algorithms = {}
        record = self.tool._candidate_profile_record("putter_redetect", algorithms)
        profile = record["candidate_profile"]
        self.assertEqual(profile["name"], pd.PROFILE_NAME)
        self.assertEqual(profile["applies_when"], {"clip.stroke_type": "putt"})
        self.assertEqual(profile["engine_by_target"], {"ball": "classic",
                                                       "clubhead": "putter"})
        self.assertEqual(profile["methods"]["observation"], pd.METHOD_DETECTOR)
        self.assertEqual(profile["methods"]["reacquired"], pd.METHOD_REACQUIRED)
        self.assertEqual(profile["constants"], pd.PROFILE)
        self.assertIn("untrained STRUCTURAL detector", profile["note"])
        self.assertFalse(profile["driven_by_reference_options"])
        self.assertIn("putter_detector.py", algorithms)

    def clip(self, stroke_type):
        return {"id": "synthetic", "path": "synthetic.mp4", "sha256": "0" * 64,
                "width": FRAME_W, "height": FRAME_H, "frame_count": 61,
                "fps_rational": {"value": 30.0}, "stroke_type": stroke_type,
                "split": "development", "tags": [],
                "window": {"start_frame": 10, "end_frame": 20}, "cut_frames": [],
                "target_names": ["clubhead"],
                "targets": {"clubhead": {"seed": {"frame": 10, "x": 70.0, "y": 40.0,
                                                 "note": "operator seed"},
                                         "options": {}, "labels": []}}}

    def test_run_clip_sends_a_putt_clubhead_to_the_detector(self):
        seen = {}

        def recorder(*args, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(points=lambda: [],
                                   summary=lambda: {"target": "clubhead"},
                                   limitations=["synthetic"], raw_frames=[])

        with mock.patch.object(importlib.import_module(
                self.tool.CANDIDATE_MODULE_PATHS["putter_redetect"]), "track_putter", recorder):
            prediction, _, raw = self.tool.run_clip(object(), self.clip("putt"),
                                                    variant="putter_redetect")
        self.assertEqual(seen["target"], "clubhead")
        self.assertEqual(seen["seed_frame"], 10)
        self.assertEqual(seen["seed_x"], 70.0)
        self.assertEqual(seen["cut_frames"], [])
        self.assertEqual(list(raw), ["clubhead"])
        self.assertEqual(prediction["targets"]["clubhead"]["options"],
                         self.tool.effective_options({}, "baseline"))

    def test_run_clip_leaves_a_full_swing_clubhead_on_the_classic_path(self):
        try:
            from ghostcaddie.demo import tracking as tracking_mod
            from ghostcaddie.demo import worker as worker_mod
        except Exception as exc:                                # pragma: no cover
            self.skipTest(f"classic tracker modules unavailable: {exc}")
        calls = {"classic": 0, "detector": 0}

        def classic(*args, **kwargs):
            calls["classic"] += 1
            return {"raw": True}

        def detector(*args, **kwargs):
            calls["detector"] += 1
            raise AssertionError("the putt candidate must not run on a full swing")

        with mock.patch.object(worker_mod, "measure_target", classic), \
                mock.patch.object(tracking_mod, "resolve_track",
                                  lambda *a, **k: SimpleNamespace(points=[])), \
                mock.patch.object(importlib.import_module(
                    self.tool.CANDIDATE_MODULE_PATHS["putter_redetect"]), "track_putter", detector):
            self.tool.run_clip(object(), self.clip("full_swing"),
                               variant="putter_redetect")
        self.assertEqual(calls["classic"], 1)
        self.assertEqual(calls["detector"], 0)


if __name__ == "__main__":
    unittest.main()
