"""Benchmark-contract and evaluator tests for ``fairwayos-target-reference.v1``.

These run on plain dictionaries and a stdlib-only module: no video, no cv2, no model.
They cover the places where a target benchmark could otherwise flatter a tracker:
a sparse denominator read as a full one, the seed scored as an observation, a
derived point counted as evidence, an unlabelled frame treated as a negative, a
recovery time invented across sparse labels, a prediction whose origin disagrees
with its status, and a reference file whose hash no longer matches the clip it
describes.

The runner's own CLI is exercised through ``--validate-only``, which is the part of
it that needs no decoder.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostcaddie.demo import target_benchmark as bench  # noqa: E402
from ghostcaddie.demo.errors import DemoError  # noqa: E402

REFUSED = bench.TargetBenchmarkError


# --------------------------------------------------------------------------
# fixtures: plain dictionaries, native-pixel coordinates


def visible(frame: Any, x: Any = 50.0, y: Any = 40.0, tolerance: Any = 4.0,
            note: str = "read off native frames",
            reviewer: str = "primary native-frame review"):
    return {"frame": frame, "state": "visible", "x": x, "y": y,
            "tolerance_px": tolerance, "note": note, "reviewer": reviewer}


def negative(frame, state="occluded", note="ball behind the flagstick"):
    return {"frame": frame, "state": state, "x": None, "y": None, "tolerance_px": None,
            "note": note, "reviewer": "primary native-frame review"}


def uncertain(frame, note="pixel smear, cannot read a centre"):
    return {"frame": frame, "state": "uncertain", "x": None, "y": None,
            "tolerance_px": None, "note": note,
            "reviewer": "primary native-frame review"}


def seed(frame=12, x=50.0, y=40.0, note="reference seed"):
    return {"frame": frame, "x": x, "y": y, "note": note}


def clip_payload(**overrides):
    clip = {
        "id": "clip_a",
        "path": "/nonexistent/clip_a.mp4",
        "sha256": "0" * 64,
        "width": 200,
        "height": 100,
        "frame_count": 120,
        "fps_rational": "24000/1001",
        "stroke_type": "full_swing",
        "split": "development",
        "tags": ["tee"],
        "window": {"start_frame": 10, "end_frame": 60},
        "cut_frames": [],
        "targets": {"ball": {"seed": seed(), "options": {}, "labels": []}},
    }
    clip.update(overrides)
    return clip


def dataset_payload(clips=None, provenance=None, **overrides):
    payload = {"schema_version": bench.DATASET_SCHEMA,
               "provenance": dict(bench.REQUIRED_PROVENANCE),
               "clips": [clip_payload()] if clips is None else clips}
    if provenance:
        payload["provenance"].update(provenance)
    payload.update(overrides)
    return payload


def automatic(frame: int, x: Any = 50.0, y: Any = 40.0, score: Any = 0.9,
              status: str = "observed_template", method: str = "template"):
    return {"source_frame": frame, "x": x, "y": y, "status": status,
            "origin": "automatic", "method": method, "score": score}


def derived(frame, x=50.0, y=40.0):
    return {"source_frame": frame, "x": x, "y": y, "status": "interpolated",
            "origin": "derived", "method": "linear_interpolation", "score": None}


def manual(frame, x=50.0, y=40.0):
    return {"source_frame": frame, "x": x, "y": y, "status": "manual_anchor",
            "origin": "operator", "method": "operator_placement", "score": None}


def absent(frame, status="missing"):
    return {"source_frame": frame, "x": None, "y": None, "status": status,
            "origin": "derived", "method": "", "score": None}


def prediction_clip(clip_id, sha, targets):
    return {"id": clip_id, "sha256": sha, "targets": targets}


def predictions_payload(clips, variant="baseline"):
    return {"schema_version": bench.PREDICTIONS_SCHEMA, "variant": variant,
            "clips": clips}


def seed_echo(clip, target="ball"):
    """The seed a runner echoes back: the reference's own seed, frame and pixel."""
    reference = clip["targets"][target]["seed"]
    if reference is None:
        return None
    return {"frame": reference["frame"], "x": reference["x"], "y": reference["y"]}


def clip_block(points, seed_payload=None):
    return {"seed": seed_payload, "points": list(points)}


def full_prediction(clip, points_by_target):
    """A prediction clip whose every target echoes the reference seed."""
    targets = {}
    for target, points in points_by_target.items():
        targets[target] = clip_block(points, seed_echo(clip, target))
    return prediction_clip(clip["id"], clip["sha256"], targets)


def evaluate_payloads(clip=None, prediction=None, points=(), targets=None):
    clip = clip or clip_payload()
    dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
    payload = prediction or predictions_payload([
        full_prediction(clip, targets or {"ball": list(points)})])
    predictions = bench.validate_predictions(payload, dataset)
    return dataset, predictions, bench.evaluate(dataset, predictions)


def score(clip=None, points=(), **kwargs):
    clip = clip or clip_payload()
    dataset, predictions, results = evaluate_payloads(clip=clip,
                                                      points=list(points), **kwargs)
    return results["clips"][clip["id"]]["targets"]["ball"]


# --------------------------------------------------------------------------


class ProvenanceTests(unittest.TestCase):
    def test_ai_visual_reference_provenance_is_accepted(self):
        validated = bench.validate_dataset(dataset_payload(), verify_hashes=False)
        self.assertEqual(validated["schema_version"], bench.DATASET_SCHEMA)
        self.assertFalse(validated["provenance"]["ground_truth"])
        self.assertFalse(validated["provenance"]["human_confirmed"])
        self.assertEqual(validated["provenance"]["label_type"], "ai_visual_reference")
        self.assertTrue(validated["provenance"]["research_only"])
        self.assertFalse(validated["provenance"]["production_eligible"])

    def test_ground_truth_claim_is_refused(self):
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(provenance={"ground_truth": True}),
                                   verify_hashes=False)
        self.assertIn("ground_truth", str(ctx.exception))

    def test_human_confirmed_claim_is_refused(self):
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(
                dataset_payload(provenance={"human_confirmed": True}), verify_hashes=False)
        self.assertIn("human_confirmed", str(ctx.exception))

    def test_label_type_and_production_eligibility_are_pinned(self):
        for provenance in ({"label_type": "human"}, {"label_type": "pseudo"},
                           {"production_eligible": True}, {"research_only": False}):
            with self.subTest(provenance=provenance):
                with self.assertRaises(REFUSED):
                    bench.validate_dataset(dataset_payload(provenance=provenance),
                                           verify_hashes=False)

    def test_missing_provenance_field_is_refused(self):
        payload = dataset_payload()
        del payload["provenance"]["human_confirmed"]
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(payload, verify_hashes=False)
        self.assertIn("human_confirmed", str(ctx.exception))

    def test_wrong_schema_version_is_refused(self):
        payload = dataset_payload()
        payload["schema_version"] = "fairwayos-annotation-dataset.v1"
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(payload, verify_hashes=False)
        self.assertIn("schema_version", str(ctx.exception))

    def test_unknown_field_is_refused(self):
        payload = dataset_payload()
        payload["clips"][0]["fabricated"] = 1
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(payload, verify_hashes=False)
        self.assertIn("does not define", str(ctx.exception))


class LabelContractTests(unittest.TestCase):
    def test_visible_label_requires_coordinates_and_tolerance(self):
        for label in ({"frame": 20, "state": "visible", "x": None, "y": 40.0,
                       "tolerance_px": 4.0, "note": "n", "reviewer": "r"},
                      {"frame": 20, "state": "visible", "x": 50.0, "y": 40.0,
                       "tolerance_px": None, "note": "n", "reviewer": "r"},
                      {"frame": 20, "state": "visible", "x": 50.0, "y": 40.0,
                       "tolerance_px": 0.0, "note": "n", "reviewer": "r"}):
            with self.subTest(label=label):
                clip = clip_payload()
                clip["targets"]["ball"]["labels"] = [label]
                with self.assertRaises(REFUSED):
                    bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)

    def test_visible_label_coordinates_must_be_finite_and_in_bounds(self):
        for x, y in ((float("inf"), 40.0), (float("nan"), 40.0), (201.0, 40.0),
                     (50.0, -1.0), ("50", 40.0)):
            with self.subTest(x=x, y=y):
                clip = clip_payload()
                clip["targets"]["ball"]["labels"] = [visible(20, x=x, y=y)]
                with self.assertRaises(REFUSED):
                    bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)

    def test_non_visible_label_must_carry_null_coordinates(self):
        for field, value in (("x", 10.0), ("y", 10.0), ("tolerance_px", 4.0)):
            with self.subTest(field=field):
                label = negative(20)
                label[field] = value
                clip = clip_payload()
                clip["targets"]["ball"]["labels"] = [label]
                with self.assertRaises(REFUSED) as ctx:
                    bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
                self.assertIn(field, str(ctx.exception))

    def test_state_must_be_one_of_the_four_and_reviewer_and_note_are_required(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [dict(negative(20), state="absent")]
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("state", str(ctx.exception))
        for field in ("note", "reviewer"):
            clip = clip_payload()
            clip["targets"]["ball"]["labels"] = [dict(visible(20), **{field: ""})]
            with self.assertRaises(REFUSED) as ctx:
                bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
            self.assertIn(field, str(ctx.exception))

    def test_duplicate_and_non_integer_label_frames_are_refused(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20), visible(20, x=51.0)]
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("repeats frame", str(ctx.exception))
        for bad in (20.5, True, "20"):
            with self.subTest(frame=bad):
                clip = clip_payload()
                clip["targets"]["ball"]["labels"] = [dict(visible(20), frame=bad)]
                with self.assertRaises(REFUSED) as ctx:
                    bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
                self.assertIn("integer", str(ctx.exception))

    def test_duplicate_clip_ids_and_duplicate_cut_frames_are_refused(self):
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip_payload(), clip_payload()]),
                                   verify_hashes=False)
        self.assertIn("repeats id", str(ctx.exception))
        clip = clip_payload(cut_frames=[20, 20])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_labels_outside_the_window_are_retained_not_refused(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(61), negative(70)]
        validated = bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
        self.assertEqual(len(validated["clips"][0]["targets"]["ball"]["labels"]), 2)

    def test_a_label_past_the_clips_last_frame_is_still_refused(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(120)]
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("outside the clip's 120 frames", str(ctx.exception))

    def test_a_seed_outside_the_window_is_still_refused(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = seed(frame=9)
        with self.assertRaises(REFUSED):
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)

    def test_a_label_on_a_structural_cut_frame_is_retained_but_not_scored(self):
        clip = clip_payload(cut_frames=[20])
        clip["targets"]["ball"]["labels"] = [visible(20), visible(21)]
        result = score(clip, [automatic(21)])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["correct_within_tolerance"], 1)
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["on_cut_frame_labels_count"], 1)
        self.assertEqual(result["on_cut_frame_labels"]["frames"], [20])
        row = next(entry for entry in result["per_frame"] if entry["frame"] == 20)
        self.assertEqual(row["outcome"], "on_cut_frame_not_scored")

    def test_a_target_block_without_a_seed_key_is_accepted_as_unseeded(self):
        clip = clip_payload()
        clip["targets"]["ball"] = {"labels": [visible(20)]}
        validated = bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
        self.assertIsNone(validated["clips"][0]["targets"]["ball"]["seed"])
        self.assertEqual(validated["clips"][0]["targets"]["ball"]["options"]["mode"],
                         "template")

    def test_null_option_bounds_are_accepted_and_other_types_are_not(self):
        clip = clip_payload()
        clip["targets"]["ball"]["options"] = {"track_until_frame": None,
                                              "automatic_until_frame": None,
                                              "track_radius_frames": None}
        validated = bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
        options = validated["clips"][0]["targets"]["ball"]["options"]
        self.assertIsNone(options["track_until_frame"])
        self.assertIsNone(options["automatic_until_frame"])
        clip = clip_payload()
        clip["targets"]["ball"]["options"] = {"track_until_frame": 20.5}
        with self.assertRaises(REFUSED):
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)

    def test_options_unknown_key_and_manual_mode_are_refused(self):
        clip = clip_payload()
        clip["targets"]["ball"]["options"] = {"teleport": True}
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("teleport", str(ctx.exception))
        clip = clip_payload()
        clip["targets"]["ball"]["options"] = {"mode": "manual"}
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        self.assertIn("manual", str(ctx.exception))


class HashBindingTests(unittest.TestCase):
    def test_frozen_input_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            clip_file = Path(directory) / "clip_a.mp4"
            clip_file.write_bytes(b"a stand-in for the sealed clip bytes")
            clip = clip_payload(path=str(clip_file),
                               sha256=hashlib.sha256(b"something else").hexdigest())
            with self.assertRaises(REFUSED) as ctx:
                bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=True)
            self.assertIn("frozen input hash mismatch", str(ctx.exception))
            honest = clip_payload(
                path=str(clip_file),
                sha256=hashlib.sha256(clip_file.read_bytes()).hexdigest())
            validated = bench.validate_dataset(dataset_payload(clips=[honest]),
                                              verify_hashes=True)
            self.assertEqual(validated["clips"][0]["sha256"], honest["sha256"])

    def test_relative_clip_path_is_resolved_against_the_dataset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            clip_file = Path(directory) / "clip_a.mp4"
            clip_file.write_bytes(b"bytes")
            clip = clip_payload(
                path="clip_a.mp4",
                sha256=hashlib.sha256(b"bytes").hexdigest())
            validated = bench.validate_dataset(dataset_payload(clips=[clip]),
                                               base_dir=directory, verify_hashes=True)
            self.assertEqual(validated["clips"][0]["path"], str(clip_file.resolve()))

    def test_predictions_must_carry_the_datasets_frozen_clip_hash(self):
        dataset = bench.validate_dataset(dataset_payload(), verify_hashes=False)
        payload = predictions_payload([prediction_clip("clip_a", "1" * 64,
                                                       {"ball": clip_block([])})])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(payload, dataset)
        self.assertIn("frozen hash", str(ctx.exception))

    def test_unknown_clip_and_missing_target_are_refused(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_b", clip["sha256"], {"ball": clip_block([])})]),
                dataset)
        self.assertIn("not a clip of this dataset", str(ctx.exception))
        other = clip_payload()
        other["targets"]["clubhead"] = {"seed": seed(frame=14), "options": {},
                                        "labels": []}
        dataset = bench.validate_dataset(dataset_payload(clips=[other]), verify_hashes=False)
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_a", other["sha256"], {"ball": clip_block([])})]),
                dataset)
        self.assertIn("omits", str(ctx.exception))


class PredictionOriginTests(unittest.TestCase):
    def _refuse(self, point, clip=None, target="ball"):
        clip = clip or clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        payload = predictions_payload([full_prediction(clip, {target: [point]})])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(payload, dataset)
        return str(ctx.exception)

    def test_automatic_observation_with_operator_origin_is_refused(self):
        message = self._refuse(dict(automatic(20), origin="operator"))
        self.assertIn("origin", message)

    def test_derived_point_with_automatic_origin_is_refused(self):
        message = self._refuse(dict(derived(20), origin="automatic"))
        self.assertIn("origin", message)

    def test_operator_anchor_with_automatic_origin_is_refused(self):
        message = self._refuse(dict(manual(20), origin="automatic"))
        self.assertIn("origin", message)

    def test_missing_frame_with_coordinates_or_score_is_refused(self):
        message = self._refuse(dict(absent(20), x=10.0))
        self.assertIn("null coordinates", message)
        message = self._refuse(dict(absent(20), score=0.4))
        self.assertIn("null coordinates", message)

    def test_automatic_observation_at_the_seed_frame_is_refused(self):
        message = self._refuse(automatic(12))
        self.assertIn("seed frame", message)

    def test_point_outside_window_or_on_a_cut_frame_is_refused(self):
        message = self._refuse(automatic(61))
        self.assertIn("outside the clip window", message)
        message = self._refuse(automatic(20), clip=clip_payload(cut_frames=[20]))
        self.assertIn("cut frame", message)

    def test_duplicate_prediction_frames_are_refused(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        payload = predictions_payload([full_prediction(
            clip, {"ball": [automatic(20), automatic(20, x=51.0)]})])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(payload, dataset)
        self.assertIn("repeats frame", str(ctx.exception))

    def test_non_finite_prediction_coordinates_are_refused(self):
        message = self._refuse(dict(automatic(20), x=float("nan")))
        self.assertIn("finite", message)

    def test_a_valid_prediction_round_trips(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        payload = predictions_payload([full_prediction(
            clip, {"ball": [automatic(20), derived(21), absent(22)]})])
        validated = bench.validate_predictions(payload, dataset)
        points = validated["clips"][0]["targets"]["ball"]["points"]
        self.assertEqual([p["source_frame"] for p in points], [20, 21, 22])

    def test_the_seed_a_runner_used_must_be_the_reference_seed(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        moved = clip_block([], {"frame": 12, "x": 60.0, "y": 40.0})
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_a", clip["sha256"], {"ball": moved})]), dataset)
        self.assertIn("seeded elsewhere", str(ctx.exception))
        late = clip_block([], {"frame": 13, "x": 50.0, "y": 40.0})
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_a", clip["sha256"], {"ball": late})]), dataset)
        self.assertIn("seed frame", str(ctx.exception))
        dropped = clip_block([])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_a", clip["sha256"], {"ball": dropped})]), dataset)
        self.assertIn("is null", str(ctx.exception))

    def test_a_runner_may_not_place_an_operator_anchor_away_from_the_seed(self):
        message = self._refuse(manual(20))
        self.assertIn("operator anchor", message)
        self.assertIn("may not place operator anchors", message)

    def test_the_targets_own_seed_anchor_is_accepted(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        payload = predictions_payload([full_prediction(clip, {"ball": [manual(12)]})])
        validated = bench.validate_predictions(payload, dataset)
        self.assertEqual(validated["clips"][0]["targets"]["ball"]["points"][0]["status"],
                         "manual_anchor")

    def test_a_prediction_can_declare_the_effective_options_it_used(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        block = clip_block([automatic(20)], seed_echo(clip))
        block["options"] = {"lk_refine": True, "template_size": 31}
        predictions = bench.validate_predictions(predictions_payload([
            prediction_clip(clip["id"], clip["sha256"], {"ball": block})]), dataset)
        results = bench.evaluate(dataset, predictions)
        score_dict = results["clips"]["clip_a"]["targets"]["ball"]
        self.assertTrue(score_dict["effective_options"]["lk_refine"])
        self.assertEqual(score_dict["effective_options_source"], "prediction")
        self.assertFalse(score_dict["options"]["lk_refine"])
        undeclared = score(clip, [automatic(20)])
        self.assertEqual(undeclared["effective_options_source"], "reference")
        self.assertFalse(undeclared["effective_options"]["lk_refine"])

    def test_an_unknown_option_key_in_a_prediction_is_refused(self):
        clip = clip_payload()
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        block = clip_block([automatic(20)], seed_echo(clip))
        block["options"] = {"teleport": True}
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip(clip["id"], clip["sha256"], {"ball": block})]), dataset)
        self.assertIn("teleport", str(ctx.exception))

    def test_a_runner_may_not_invent_a_seed_the_reference_does_not_declare(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = None
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        invented = clip_block([], {"frame": 12, "x": 50.0, "y": 40.0})
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(predictions_payload([
                prediction_clip("clip_a", clip["sha256"], {"ball": invented})]), dataset)
        self.assertIn("may not invent", str(ctx.exception))
        self.assertIsNotNone(bench.validate_predictions(predictions_payload([
            prediction_clip("clip_a", clip["sha256"], {"ball": clip_block([])})]), dataset))


class DenominatorTests(unittest.TestCase):
    def test_sparse_labels_give_a_sparse_denominator_and_a_null_missing_run(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20), visible(30)]
        result = score(clip, [automatic(20), automatic(30, x=52.0)])
        self.assertEqual(result["evaluated_visible_frames"], 2)
        self.assertEqual(result["correct_within_tolerance"], 2)
        self.assertEqual(result["recall_within_tolerance"], 1.0)
        self.assertEqual(result["unlabelled_window_frames_absent_from_denominator"], 49)
        run = result["longest_missing_run_among_consecutive_visible_labels"]
        self.assertIsNone(run["frames"])
        self.assertIn("sparse", run["reason"])

    def test_unlabelled_frames_are_absent_from_the_denominator_and_never_negatives(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20)]
        result = score(clip, [automatic(20), automatic(21), automatic(22)])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["negative_labeled_frames"], 0)
        self.assertEqual(result["negative_fp_count"], 0)
        self.assertIsNone(result["negative_fp_rate"])
        self.assertEqual(result["accounting"]["automatic_points_emitted"], 3)

    def test_seed_frame_is_excluded_and_the_seed_is_never_evidence(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(12, x=50.0, y=40.0), visible(20)]
        result = score(clip, [manual(12), automatic(20)])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["correct_within_tolerance"], 1)
        self.assertTrue(result["seed_frame_excluded"])
        self.assertEqual(result["accounting"]["operator_anchor_points_emitted"], 1)
        self.assertEqual(result["accounting"]["automatic_points_emitted"], 1)
        scored = [entry for entry in result["per_frame"]
                  if entry["outcome"] in ("correct", "incorrect", "miss")]
        self.assertEqual([entry["frame"] for entry in scored], [20])
        # The label at the seed frame is RETAINED as pre-initialization, never scored.
        self.assertEqual(result["preseed_visible_count"], 1)
        seed_row = next(entry for entry in result["per_frame"] if entry["frame"] == 12)
        self.assertEqual(seed_row["outcome"], "preseed_not_scored")

    def test_negatives_are_only_what_the_reference_stated(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20), negative(21),
                                             negative(22, state="out_of_view"),
                                             uncertain(23)]
        result = score(clip, [automatic(20), automatic(21), automatic(22), automatic(23)])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["negative_labeled_frames"], 2)
        self.assertEqual(result["negative_fp_count"], 2)
        self.assertEqual(result["negative_fp_rate"], 1.0)
        self.assertEqual(result["uncertain_labels_excluded"], 1)
        counts = {entry["frame"]: entry["outcome"] for entry in result["per_frame"]
                  if entry["outcome"] in ("correct", "incorrect", "miss")}
        self.assertEqual(counts, {20: "correct"})
        negatives_rows = {entry["frame"]: entry["outcome"]
                          for entry in result["per_frame"]
                          if entry.get("state") in ("occluded", "out_of_view")}
        self.assertEqual(negatives_rows, {21: "negative", 22: "negative"})

    def test_uncertain_labels_are_neither_positive_nor_negative(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [uncertain(20), uncertain(21)]
        result = score(clip, [automatic(20), automatic(21)])
        self.assertEqual(result["evaluated_visible_frames"], 0)
        self.assertIsNone(result["recall_within_tolerance"])
        self.assertEqual(result["negative_labeled_frames"], 0)
        self.assertEqual(result["negative_fp_count"], 0)
        self.assertTrue(any("no denominator" in note for note in result["limitations"]))

    def test_recall_error_and_longest_missing_run_arithmetic(self):
        clip = clip_payload(window={"start_frame": 8, "end_frame": 60})
        clip["targets"]["ball"]["seed"] = seed(frame=9)
        clip["targets"]["ball"]["labels"] = [visible(10, tolerance=5.0),
                                             visible(11, tolerance=5.0),
                                             visible(12, tolerance=5.0),
                                             visible(13, tolerance=5.0),
                                             visible(14, tolerance=5.0)]
        result = score(clip, [automatic(10, x=50.0, y=40.0),
                              automatic(11, x=52.0, y=40.0),
                              automatic(12, x=70.0, y=40.0)])
        self.assertEqual(result["evaluated_visible_frames"], 5)
        self.assertEqual(result["correct_within_tolerance"], 2)
        self.assertEqual(result["incorrect_visible_count"], 1)
        self.assertEqual(result["miss_count"], 2)
        self.assertEqual(result["recall_within_tolerance"], 0.4)
        summary = result["error_summary"]
        self.assertEqual((summary["count"], summary["median_px"], summary["p90_px"],
                          summary["max_px"]), (3, 2.0, 20.0, 20.0))
        run = result["longest_missing_run_among_consecutive_visible_labels"]
        self.assertEqual(run["frames"], 2)
        self.assertEqual(run["run_frames"], [13, 14])

    def test_tolerance_boundary_is_inclusive_and_derived_points_are_not_evidence(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20, x=50.0, y=40.0, tolerance=4.0),
                                             visible(21)]
        result = score(clip, [automatic(20, x=54.0, y=40.0), derived(21), absent(22)])
        self.assertEqual(result["correct_within_tolerance"], 1)
        self.assertEqual(result["miss_count"], 1)
        self.assertEqual(result["accounting"]["derived_points_at_visible_labels"], 1)
        self.assertEqual(result["accounting"]["derived_points_emitted"], 1)
        self.assertEqual(result["accounting"]["missing_or_lost_points_emitted"], 1)
        self.assertEqual(result["accounting"]["automatic_points_emitted"], 1)

    def test_tracker_score_is_never_used_as_accuracy(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20), visible(21, x=200.0)]
        high = score(clip, [automatic(20, score=0.99), automatic(21, score=0.99)])
        low = score(clip, [automatic(20, score=0.01), automatic(21, score=0.01)])
        none = score(clip, [automatic(20, score=None), automatic(21, score=None)])
        for field in ("correct_within_tolerance", "recall_within_tolerance",
                      "incorrect_visible_count", "error_summary",
                      "negative_fp_count"):
            self.assertEqual(high[field], low[field])
            self.assertEqual(high[field], none[field])
        self.assertFalse(high["accounting"]["score_used_in_any_metric"])


class RecoveryTests(unittest.TestCase):
    def test_latency_is_measured_from_the_first_visible_frame(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22),
                                             visible(23)]
        result = score(clip, [automatic(23)])
        recovery = result["recovery"]
        self.assertEqual(recovery["occlusion_to_visible_transitions"], 1)
        self.assertEqual(recovery["evaluated_transitions"], 1)
        self.assertEqual(recovery["excluded_before_acquisition"], 0)
        self.assertEqual(recovery["recovered_within_labelled_run"], 1)
        self.assertIn("first_visible_frame", recovery["latency_definition"])
        case = recovery["cases"][0]
        self.assertEqual(case["occlusion_frame"], 20)
        self.assertEqual(case["first_visible_frame"], 21)
        self.assertEqual(case["recovered_frame"], 23)
        self.assertEqual(case["latency_frames"], 2)
        self.assertEqual(recovery["median_latency_frames"], 2.0)
        self.assertEqual(recovery["mean_latency_frames"], 2.0)
        self.assertEqual(recovery["latencies_frames"], [2])

    def test_re_acquiring_on_the_first_visible_frame_is_zero_latency(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22)]
        result = score(clip, [automatic(21)])
        self.assertEqual(result["recovery"]["cases"][0]["latency_frames"], 0)
        self.assertEqual(result["recovery"]["median_latency_frames"], 0)

    def test_even_median_latency_averages_middle_values(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22),
                                             visible(23), visible(24),
                                             negative(30), visible(31), visible(32)]
        recovery = score(clip, [automatic(24), automatic(31)])["recovery"]
        self.assertEqual(recovery["latencies_frames"], [0, 3])
        self.assertEqual(recovery["median_latency_frames"], 1.5)

    def test_median_latency_is_a_median_not_a_mean(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = (
            [negative(20), visible(21), visible(22)] +
            [negative(30), visible(31), visible(32)] +
            [negative(40)] + [visible(f) for f in range(41, 51)])
        recovery = score(clip, [automatic(21), automatic(32), automatic(50)])["recovery"]
        self.assertEqual(recovery["latencies_frames"], [0, 1, 9])
        self.assertEqual(recovery["median_latency_frames"], 1)
        self.assertEqual(recovery["mean_latency_frames"], 3.333)

    def test_a_transition_at_or_before_the_seed_is_excluded_not_counted(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = seed(frame=20)
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22)]
        result = score(clip, [automatic(21)])
        recovery = result["recovery"]
        self.assertEqual(recovery["occlusion_to_visible_transitions"], 1)
        self.assertEqual(recovery["evaluated_transitions"], 0)
        self.assertEqual(recovery["excluded_before_acquisition"], 1)
        self.assertIsNone(recovery["median_latency_frames"])
        self.assertIn("at or before", recovery["cases"][0]["reason"])
        self.assertIsNotNone(recovery["null_reason"])

    def test_a_transition_without_any_seed_is_excluded(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = None
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22)]
        result = score(clip, [])
        recovery = result["recovery"]
        self.assertEqual(recovery["excluded_before_acquisition"], 1)
        self.assertEqual(recovery["evaluated_transitions"], 0)
        self.assertIn("no seed at all", recovery["cases"][0]["reason"])

    def test_sparse_labels_after_the_occlusion_return_null_with_a_reason(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(30), visible(31), visible(35)]
        result = score(clip, [automatic(31)])
        recovery = result["recovery"]
        self.assertEqual(recovery["evaluated_transitions"], 0)
        self.assertEqual(recovery["recovered_within_labelled_run"], 0)
        self.assertIsNone(recovery["median_latency_frames"])
        self.assertIsNone(recovery["mean_latency_frames"])
        self.assertIn("consecutive", recovery["null_reason"])
        self.assertFalse(recovery["cases"][0]["evaluated"])
        self.assertIn("sparse", recovery["cases"][0]["reason"])

    def test_a_non_adjacent_occlusion_is_not_a_transition(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(20), visible(22), visible(23)]
        result = score(clip, [automatic(22)])
        recovery = result["recovery"]
        self.assertEqual(recovery["occlusion_to_visible_transitions"], 0)
        self.assertIsNone(recovery["median_latency_frames"])
        self.assertIsNotNone(recovery["null_reason"])

    def test_visible_to_occlusion_is_not_a_recovery(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [visible(20), negative(21)]
        result = score(clip, [automatic(20)])
        self.assertEqual(result["recovery"]["occlusion_to_visible_transitions"], 0)

    def test_not_recovered_inside_the_labelled_run_is_stated(self):
        clip = clip_payload()
        clip["targets"]["ball"]["labels"] = [negative(20), visible(21), visible(22)]
        result = score(clip, [automatic(21, x=100.0, y=40.0)])
        case = result["recovery"]["cases"][0]
        self.assertTrue(case["evaluated"])
        self.assertIsNone(case["recovered_frame"])
        self.assertIsNone(case["latency_frames"])
        self.assertIn("no correct automatic observation", case["reason"])


class SeededScopeTests(unittest.TestCase):
    """One forward leg per target: pre-seed labels are pre-initialization, and an
    unseeded target has no accuracy denominator at all. Both are retained, never
    silently dropped."""

    def test_visible_labels_before_the_seed_are_excluded_from_the_denominator(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = seed(frame=20)
        clip["targets"]["ball"]["labels"] = [visible(11), visible(19), visible(20),
                                             visible(21), visible(22)]
        result = score(clip, [automatic(21), automatic(22)])
        self.assertEqual(result["evaluated_visible_frames"], 2)
        self.assertEqual(result["correct_within_tolerance"], 2)
        self.assertEqual(result["recall_within_tolerance"], 1.0)
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["preseed_visible_count"], 3)
        self.assertEqual(result["preseed_labels"]["total"], 3)
        self.assertEqual(result["preseed_labels"]["frames"], [11, 19, 20])
        outcomes = {entry["frame"]: entry["outcome"] for entry in result["per_frame"]}
        self.assertEqual(outcomes[11], "preseed_not_scored")
        self.assertEqual(outcomes[19], "preseed_not_scored")
        self.assertEqual(outcomes[20], "preseed_not_scored")
        self.assertEqual(outcomes[21], "correct")
        self.assertIn("pre-initialization", result["recall_denominator_note"])
        self.assertTrue(any("pre-initialization" in note
                            for note in result["limitations"]))

    def test_pre_seed_labels_are_not_counted_as_misses_or_negatives(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = seed(frame=20)
        clip["targets"]["ball"]["labels"] = [visible(12), negative(13), uncertain(14),
                                             visible(21)]
        result = score(clip, [])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["miss_count"], 1)
        self.assertEqual(result["negative_labeled_frames"], 0)
        self.assertEqual(result["negative_fp_count"], 0)
        self.assertEqual(result["uncertain_labels_excluded"], 0)
        self.assertEqual(result["preseed_labels"]["total"], 3)
        self.assertEqual(result["preseed_labels"]["visible"], 1)
        self.assertEqual(result["preseed_labels"]["occluded"], 1)
        self.assertEqual(result["preseed_labels"]["uncertain"], 1)

    def test_an_unseeded_target_has_no_accuracy_denominator(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = None
        clip["targets"]["ball"]["labels"] = [visible(20), visible(21), negative(22)]
        result = score(clip, [])
        self.assertEqual(result["evaluated_visible_frames"], 0)
        self.assertIsNone(result["recall_within_tolerance"])
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["incorrect_visible_count"], 0)
        self.assertEqual(result["negative_labeled_frames"], 0)
        self.assertIsNone(result["error_summary"]["median_px"])
        self.assertEqual(result["unseeded_visible_count"], 2)
        self.assertEqual(result["unseeded_labels"]["total"], 3)
        self.assertIn("unseeded", result["recall_denominator_note"])
        self.assertEqual(result["recovery"]["evaluated_transitions"], 0)
        self.assertIsNotNone(result["recovery"]["null_reason"])
        outcomes = {entry["frame"]: entry["outcome"] for entry in result["per_frame"]}
        self.assertEqual(outcomes[20], "unseeded_not_scored")
        self.assertTrue(any("no accuracy denominator" in note
                            for note in result["limitations"]))

    def test_backward_or_both_directions_are_refused(self):
        for direction in ("backward", "both"):
            with self.subTest(direction=direction):
                clip = clip_payload()
                clip["targets"]["ball"]["options"] = {"direction": direction}
                with self.assertRaises(REFUSED) as ctx:
                    bench.validate_dataset(dataset_payload(clips=[clip]),
                                           verify_hashes=False)
                self.assertIn("forward", str(ctx.exception))

    def test_forward_direction_is_accepted(self):
        clip = clip_payload()
        clip["targets"]["ball"]["options"] = {"direction": "forward"}
        validated = bench.validate_dataset(dataset_payload(clips=[clip]),
                                          verify_hashes=False)
        self.assertEqual(validated["clips"][0]["targets"]["ball"]["options"]["direction"],
                         "forward")

    def test_outside_window_labels_are_retained_and_excluded_from_the_denominators(self):
        """The unit regression the primary asked for: a visible label and a negative
        label outside the window are both retained, both counted, and neither reaches
        the recall denominator or the negative-FP denominator."""
        clip = clip_payload()   # window [10, 60], seed at 12
        clip["targets"]["ball"]["labels"] = [
            visible(20),            # in window, post-seed: the only scored label
            negative(21),           # in window, post-seed: the only scored negative
            visible(70),            # AFTER the window: retained, excluded
            negative(75, state="out_of_view"),   # AFTER the window: retained, excluded
            negative(5),            # BEFORE the window: retained, excluded
        ]
        result = score(clip, [automatic(20, x=54.0, y=40.0)])
        self.assertEqual(result["evaluated_visible_frames"], 1)
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["incorrect_visible_count"], 0)
        self.assertEqual(result["negative_labeled_frames"], 1)
        self.assertEqual(result["negative_fp_count"], 0)
        self.assertEqual(result["negative_fp_rate"], 0.0)
        self.assertEqual(result["out_of_window_labels_count"], 3)
        self.assertEqual(result["out_of_window_labels"]["visible"], 1)
        self.assertEqual(result["out_of_window_labels"]["out_of_view"], 1)
        self.assertEqual(result["out_of_window_labels"]["occluded"], 1)
        self.assertEqual(result["out_of_window_labels"]["frames"], [5, 70, 75])
        rows = {entry["frame"]: entry["outcome"] for entry in result["per_frame"]}
        self.assertEqual(rows[20], "correct")
        self.assertEqual(rows[21], "negative")
        self.assertEqual(rows[5], "out_of_window_not_scored")
        self.assertEqual(rows[70], "out_of_window_not_scored")
        self.assertEqual(rows[75], "out_of_window_not_scored")
        self.assertTrue(any("OUTSIDE the clip window" in note
                            for note in result["limitations"]))
        # The point of the exclusion: had those outside-window negatives been counted,
        # the negative denominator would be 3 instead of 1 and the FP rate would look
        # artificially easy, because no prediction is allowed outside the window in the
        # first place (the predictions contract refuses such a point outright).
        self.assertEqual(result["accounting"]["automatic_points_emitted"], 1)
        dataset = bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)
        payload = predictions_payload([full_prediction(clip, {"ball": [automatic(70)]})])
        with self.assertRaises(REFUSED) as ctx:
            bench.validate_predictions(payload, dataset)
        self.assertIn("outside the clip window", str(ctx.exception))


class StructuralSuppressionTests(unittest.TestCase):
    def test_window_and_cut_frames_are_suppressed_without_bridging(self):
        window = {"start_frame": 10, "end_frame": 60}
        points = [automatic(8), automatic(10), derived(30), automatic(40),
                  automatic(45), derived(50), automatic(55), automatic(70)]
        # a cut at 20 resets the source: the derived point at 30 would bridge it.
        kept, summary = bench.suppress_structural(points, window=window, cut_frames=[20])
        self.assertEqual([point["source_frame"] for point in kept],
                         [10, 40, 45, 50, 55])
        self.assertEqual(summary["dropped_outside_window"], 2)
        self.assertEqual(summary["dropped_bridging_a_cut"], 1)
        self.assertEqual(summary["dropped_at_cut_frames"], 0)

    def test_a_point_on_the_cut_frame_itself_is_dropped(self):
        window = {"start_frame": 10, "end_frame": 60}
        kept, summary = bench.suppress_structural(
            [automatic(19), automatic(20), automatic(21)], window=window,
            cut_frames=[20])
        self.assertEqual([point["source_frame"] for point in kept], [19, 21])
        self.assertEqual(summary["dropped_at_cut_frames"], 1)

    def test_without_cuts_nothing_is_suppressed_inside_the_window(self):
        window = {"start_frame": 10, "end_frame": 60}
        points = [automatic(10), derived(11), absent(12), automatic(13)]
        kept, summary = bench.suppress_structural(points, window=window, cut_frames=[])
        self.assertEqual(len(kept), 4)
        self.assertEqual(summary["dropped_bridging_a_cut"], 0)


class SplitAndReportTests(unittest.TestCase):
    def _two_clip_results(self):
        first = clip_payload()
        first["targets"]["ball"]["labels"] = [visible(20), visible(21)]
        second = clip_payload(id="clip_b", split="evaluation", tags=["putt"])
        second["targets"]["ball"]["labels"] = [visible(30)]
        dataset = bench.validate_dataset(
            dataset_payload(clips=[first, second]), verify_hashes=False)
        payload = predictions_payload([
            full_prediction(first, {"ball": [automatic(20), automatic(21)]}),
            full_prediction(second, {"ball": [automatic(30, x=90.0)]}),
        ])
        predictions = bench.validate_predictions(payload, dataset)
        return bench.evaluate(dataset, predictions)

    def test_development_and_evaluation_are_reported_separately(self):
        results = self._two_clip_results()
        self.assertEqual(results["splits"]["development"]["clips"], ["clip_a"])
        self.assertEqual(results["splits"]["evaluation"]["clips"], ["clip_b"])
        developed = results["splits"]["development"]["targets"]["ball"]
        evaluated = results["splits"]["evaluation"]["targets"]["ball"]
        self.assertEqual(developed["evaluated_visible_frames"], 2)
        self.assertEqual(evaluated["evaluated_visible_frames"], 1)
        self.assertEqual(evaluated["correct_within_tolerance"], 0)
        self.assertEqual(evaluated["incorrect_visible_count"], 1)

    def test_tags_are_pooled_per_split(self):
        results = self._two_clip_results()
        tags = results["splits"]["development"]["tags"]
        self.assertIn("tee", tags)
        self.assertEqual(tags["tee"]["ball"]["evaluated_visible_frames"], 2)
        self.assertNotIn("putt", tags)

    def test_the_pooled_all_splits_block_refuses_to_be_a_holdout(self):
        results = self._two_clip_results()
        note = results["pooled_all_splits"]["note"]
        self.assertIn("NOT a held-out", note)
        self.assertIn("not a statistical holdout", note)

    def test_markdown_report_states_what_the_numbers_are_and_are_not(self):
        markdown = bench.render_markdown(self._two_clip_results())
        self.assertIn("not a statistical holdout", markdown)
        self.assertIn("family-independence", markdown)
        self.assertIn("sparse AI-visual references", markdown)
        self.assertIn("## Development split", markdown)
        self.assertIn("## Evaluation split", markdown)
        self.assertIn("## Pooled all splits (bookkeeping only, NOT held out)", markdown)
        self.assertIn("seed frame is excluded from every score", markdown)
        self.assertIn("used in no metric", markdown)

    def test_json_report_is_serialisable_and_hashable(self):
        results = self._two_clip_results()
        text = bench.canonical_json(results)
        self.assertIn("fairwayos-target-benchmark-report.v1", text)
        self.assertEqual(json.loads(text)["schema_version"], bench.REPORT_SCHEMA)

    def test_a_target_without_a_seed_scores_nothing_and_says_so(self):
        clip = clip_payload()
        clip["targets"]["ball"]["seed"] = None
        clip["targets"]["ball"]["labels"] = [visible(20)]
        result = score(clip, [])
        self.assertEqual(result["evaluated_visible_frames"], 0)
        self.assertIsNone(result["recall_within_tolerance"])
        self.assertEqual(result["unseeded_visible_count"], 1)
        self.assertTrue(any("unseeded" in note or "no accuracy denominator" in note
                            for note in result["limitations"]))

    def test_error_message_types_are_demo_errors(self):
        self.assertTrue(issubclass(REFUSED, DemoError))


class StdlibOnlyTests(unittest.TestCase):
    """The evaluator must run where cv2/numpy do not exist."""

    def test_module_imports_and_evaluates_with_cv2_blocked(self):
        program = textwrap.dedent(
            """
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in {"cv2", "numpy"}:
                        raise ImportError(f"{name} is blocked in this test")
                    return None

            sys.meta_path.insert(0, Blocker())
            from ghostcaddie.demo import target_benchmark as bench

            clip = {"id": "c", "path": "c.mp4", "sha256": "0" * 64, "width": 200,
                    "height": 100, "frame_count": 40, "fps_rational": "30/1",
                    "stroke_type": "putt", "split": "development", "tags": [],
                    "window": {"start_frame": 1, "end_frame": 20}, "cut_frames": [],
                    "targets": {"ball": {"seed": {"frame": 2, "x": 5.0, "y": 5.0,
                                                  "note": "seed"},
                                         "options": {},
                                         "labels": [{"frame": 5, "state": "visible",
                                                     "x": 10.0, "y": 10.0,
                                                     "tolerance_px": 3.0,
                                                     "note": "n", "reviewer": "r"}]}}}
            dataset = bench.validate_dataset(
                {"schema_version": bench.DATASET_SCHEMA,
                 "provenance": dict(bench.REQUIRED_PROVENANCE), "clips": [clip]},
                verify_hashes=False)
            predictions = bench.validate_predictions(
                {"schema_version": bench.PREDICTIONS_SCHEMA, "variant": "baseline",
                 "clips": [{"id": "c", "sha256": "0" * 64, "targets": {"ball": {
                     "seed": {"frame": 2, "x": 5.0, "y": 5.0},
                     "points": [{"source_frame": 5, "x": 11.0, "y": 10.0,
                                 "status": "observed_template", "origin": "automatic",
                                 "method": "template", "score": 0.8}]}}}]}, dataset)
            results = bench.evaluate(dataset, predictions)
            assert results["splits"]["development"]["targets"]["ball"][
                "correct_within_tolerance"] == 1
            assert "cv2" not in sys.modules and "numpy" not in sys.modules
            print("stdlib-only-ok")
            """)
        completed = subprocess.run([sys.executable, "-c", program], cwd=str(ROOT),
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("stdlib-only-ok", completed.stdout)


class VariantAblationTests(unittest.TestCase):
    """``lk_agreement`` changes lk_refine and nothing else; baseline is untouched."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tiger_target_benchmark_tool",
            ROOT / "tools" / "tiger_target_benchmark.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tool = module

    def test_baseline_leaves_the_reference_options_exactly_as_they_are(self):
        block = {"template_size": 31, "search_radius": 40, "lk_refine": False,
                 "interpolation": {"enabled": False, "max_gap_frames": 0}}
        baseline = self.tool.effective_options(block, "baseline")
        expected = {key: value for key, value in block.items()
                    if key != "interpolation"}
        self.assertEqual({key: value for key, value in baseline.items()
                          if key != "interpolation"}, expected)
        self.assertEqual(baseline["interpolation"],
                         {"enabled": False, "max_gap_frames": 0})
        self.assertNotIn("interpolation", expected)

    def test_lk_agreement_sets_lk_refine_true_and_nothing_else(self):
        block = {"template_size": 31, "search_radius": 44, "min_score": 0.55,
                 "loss_frames": 3, "direction": "forward", "lk_refine": False,
                 "max_jump_px": 60.0,
                 "interpolation": {"enabled": True, "max_gap_frames": 2}}
        baseline = self.tool.effective_options(block, "baseline")
        agreement = self.tool.effective_options(block, "lk_agreement")
        self.assertTrue(agreement["lk_refine"])
        self.assertFalse(baseline["lk_refine"])
        differences = {key for key in set(baseline) | set(agreement)
                       if baseline.get(key) != agreement.get(key)}
        self.assertEqual(differences, {"lk_refine"})
        self.assertEqual(agreement["interpolation"], baseline["interpolation"])

    def test_the_override_table_names_only_implemented_variants(self):
        self.assertEqual(set(self.tool.VARIANT_OPTION_OVERRIDES),
                         set(self.tool.SUPPORTED_VARIANTS))
        self.assertEqual(self.tool.VARIANT_OPTION_OVERRIDES["baseline"], {})
        for variant in self.tool.SUPPORTED_VARIANTS:
            with self.subTest(variant=variant):
                self.assertIn(variant, self.tool.VARIANT_NOTES)

    def test_an_unknown_variant_is_refused_before_any_option_is_composed(self):
        with self.assertRaises(self.tool.RunnerError):
            self.tool.effective_options({}, "candidate-2")

    def test_the_tool_imports_without_cv2(self):
        self.assertNotIn("cv2", dir(self.tool))


class ManifestBindingTests(unittest.TestCase):
    """The runner's scope binding to the showcase batch manifest."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tiger_target_benchmark_tool",
            ROOT / "tools" / "tiger_target_benchmark.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tool = module

    def _dataset(self, clip_id, clip_sha):
        clip = clip_payload(id=clip_id, sha256=clip_sha)
        return bench.validate_dataset(dataset_payload(clips=[clip]), verify_hashes=False)

    def test_a_clip_outside_the_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"clips": [
                {"id": "clip_b", "clip": "/x/clip_b.mp4", "clip_sha256": "1" * 64}]}),
                encoding="utf-8")
            dataset = self._dataset("clip_a", "0" * 64)
            with self.assertRaises(self.tool.RunnerError) as ctx:
                self.tool._manifest_check(dataset, manifest)
            self.assertIn("not in the manifest", str(ctx.exception))

    def test_a_manifest_clip_the_dataset_omits_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"clips": [
                {"id": "clip_a", "clip": "/x/clip_a.mp4", "clip_sha256": "0" * 64},
                {"id": "clip_b", "clip": "/x/clip_b.mp4", "clip_sha256": "1" * 64}]}),
                encoding="utf-8")
            dataset = self._dataset("clip_a", "0" * 64)
            with self.assertRaises(self.tool.RunnerError) as ctx:
                self.tool._manifest_check(dataset, manifest)
            self.assertIn("omits manifest clip", str(ctx.exception))

    def test_a_hash_that_disagrees_with_the_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"clips": [
                {"id": "clip_a", "clip": "/x/clip_a.mp4", "clip_sha256": "2" * 64}]}),
                encoding="utf-8")
            dataset = self._dataset("clip_a", "0" * 64)
            with self.assertRaises(self.tool.RunnerError) as ctx:
                self.tool._manifest_check(dataset, manifest)
            self.assertIn("sha256 differs", str(ctx.exception))

    def test_a_matching_clip_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"clips": [
                {"id": "clip_a", "clip": "/x/clip_a.mp4", "clip_sha256": "0" * 64}]}),
                encoding="utf-8")
            dataset = self._dataset("clip_a", "0" * 64)
            bound = self.tool._manifest_check(dataset, manifest)
            self.assertEqual(bound["clips_checked"], 1)


class RunnerCliTests(unittest.TestCase):
    """The runner's validation path, which is the part that needs no decoder."""

    def _dataset_file(self, directory: Path) -> Path:
        clip_file = directory / "clip_a.mp4"
        clip_file.write_bytes(b"stand-in clip bytes")
        clip = clip_payload(path=str(clip_file),
                            sha256=hashlib.sha256(clip_file.read_bytes()).hexdigest())
        clip["targets"]["ball"]["labels"] = [visible(20)]
        dataset_file = directory / "dataset.json"
        dataset_file.write_text(json.dumps(dataset_payload(clips=[clip])), encoding="utf-8")
        return dataset_file

    def _run(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "tiger_target_benchmark.py"),
             *arguments],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180)

    def test_validate_only_accepts_a_bound_dataset_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = self._dataset_file(Path(directory))
            out = Path(directory) / "predictions.json"
            completed = self._run("--dataset", str(dataset_file), "--out", str(out),
                                  "--variant", "baseline", "--validate-only")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("validate-only: dataset accepted", completed.stdout)
            self.assertFalse(out.exists())

    def test_hash_bypass_cannot_start_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = self._dataset_file(Path(directory))
            out = Path(directory) / "p.json"
            completed = self._run("--dataset", str(dataset_file), "--out", str(out),
                                  "--no-hash-check")
            self.assertEqual(completed.returncode, 3)
            self.assertIn("allowed only with --validate-only", completed.stderr)
            self.assertFalse(out.exists())

    def test_a_tampered_dataset_is_refused_with_a_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = self._dataset_file(Path(directory))
            payload = json.loads(dataset_file.read_text())
            payload["provenance"]["ground_truth"] = True
            dataset_file.write_text(json.dumps(payload), encoding="utf-8")
            completed = self._run("--dataset", str(dataset_file),
                                  "--out", str(Path(directory) / "p.json"),
                                  "--validate-only")
            self.assertEqual(completed.returncode, 2)
            self.assertIn("REFUSED", completed.stderr)
            self.assertIn("ground_truth", completed.stderr)

    def test_unsupported_variant_fails_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = self._dataset_file(Path(directory))
            completed = self._run("--dataset", str(dataset_file),
                                  "--out", str(Path(directory) / "p.json"),
                                  "--variant", "candidate-2", "--validate-only")
            self.assertEqual(completed.returncode, 3)
            self.assertIn("unsupported variant", completed.stderr)


if __name__ == "__main__":
    unittest.main()
