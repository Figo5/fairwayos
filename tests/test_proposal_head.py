"""Synthetic CV behaviour contracts for proposal_head, not golf-footage accuracy.

At least one positive and one negative path exercise real cv2 pixels (the frozen
shaft detector and the frozen native-contrast core measurement); one isolated
competition test patches the shared ribbon helper.
"""
import copy
import json
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ghostcaddie.demo import proposal_head as h
from ghostcaddie.demo import shaft_head


def body(wrist=(100, 300), score=0.8, roi=(40, 320, 120, 300)):
    return {"roi_xywh": list(roi),
            "wrists": {node: {"x": float(wrist[0]), "y": float(wrist[1]),
                              "score": float(score)}
                       for node in ("wr_l", "wr_r")}}


def det(x1, y1, x2, y2, conf=0.5, cls=0):
    return {"xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf), "class_id": cls}


def image(head=True, shaft=True, bright=False, second=False):
    im = np.full((720, 400, 3), 170, np.uint8)
    color = (255,) * 3 if bright else (30,) * 3
    if shaft:
        cv2.line(im, (100, 300), (265, 300), (30,) * 3, 2)
    if head:
        if bright:
            cv2.circle(im, (265, 300), 7, color, -1)
        else:
            cv2.ellipse(im, (265, 300), (8, 5), 0, 0, 360, color, -1)
    if second:
        cv2.line(im, (100, 300), (230, 200), (30,) * 3, 2)
        cv2.ellipse(im, (230, 200), (8, 5), 0, 0, 360, (30,) * 3, -1)
    return im


_DEFAULT_BODY = object()


def run(im, dets, b=_DEFAULT_BODY, origin=(0, 0), size=(400, 720)):
    return h.measure_proposals(im, origin, size, body() if b is _DEFAULT_BODY else b, dets)


HEAD_BOX = det(240, 283, 272, 317)


class ProposalHeadTests(unittest.TestCase):
    def test_real_pixels_positive_exports_measured_core_not_box_centre(self):
        res = run(image(), [HEAD_BOX])
        self.assertEqual(res["refusal_reason"], None)
        selected = res["selected"]
        self.assertIsNotNone(selected)
        centroid = np.mean(selected["support_pixels_native"], axis=0)
        np.testing.assert_allclose(selected["centroid_native"], centroid, atol=1e-9)
        box_centre = [(240 + 272) / 2.0, (283 + 317) / 2.0]
        self.assertNotEqual(list(selected["centroid_native"]), box_centre)
        self.assertNotEqual(list(selected["centroid_native"]), selected["distal"])
        self.assertEqual(len(selected["support_sha256"]), 64)
        self.assertEqual(res["cluster_count"], 1)
        self.assertEqual(selected["proposal_indices"], [0])

    def test_bg_branch_box_disconnected_from_wrists_is_vetoed(self):
        im = image()
        cv2.ellipse(im, (330, 580), (8, 5), 0, 0, 360, (30,) * 3, -1)
        cv2.line(im, (300, 580), (330, 580), (30,) * 3, 2)
        res = run(im, [HEAD_BOX, det(310, 562, 352, 600)])
        self.assertIsNotNone(res["selected"])
        self.assertLess(res["selected"]["centroid_native"][0], 300)
        background = res["proposals"][1]
        self.assertEqual(background["supporting_line_count"], 0)
        self.assertIn("no_supporting_segment_near_proposal_box",
                      background["refusal_reasons"])

    def test_shaft_edge_without_compact_core_is_refused(self):
        res = run(image(head=False), [HEAD_BOX])
        self.assertIsNone(res["selected"])
        self.assertEqual(res["cluster_count"], 0)
        self.assertIn(res["refusal_reason"], ("no_measured_proposal_core", "no_qualifying_proposal_core"))
        self.assertTrue(all(not r['core_accepted'] for r in res['refusals']))

    def test_two_competing_measured_cores_are_withheld(self):
        res = run(image(second=True), [det(210, 182, 250, 218), HEAD_BOX])
        self.assertIsNone(res["selected"])
        self.assertEqual(res["cluster_count"], 2)
        self.assertEqual(res["refusal_reason"],
                         "ambiguous_multiple_proposal_core_clusters")

    def test_low_confidence_can_propose_but_never_selects_by_confidence(self):
        im = image()
        cv2.ellipse(im, (330, 580), (8, 5), 0, 0, 360, (30,) * 3, -1)
        low = det(240, 283, 272, 317, conf=0.001)
        high = det(312, 562, 348, 598, conf=0.99)
        res = run(im, [high, low])
        self.assertEqual(low["confidence"], h.PROPOSAL_CONFIDENCE_FLOOR)
        self.assertIsNotNone(res["selected"])
        self.assertLess(res["selected"]["centroid_native"][0], 300)
        self.assertEqual(res["selected"]["proposal_index"], 1)

    def test_absent_or_unreliable_current_wrists_are_withheld(self):
        res = run(image(), [HEAD_BOX], b=None)
        self.assertIsNone(res["selected"])
        self.assertEqual(res["refusal_reason"], "body_absent")
        weak = body(score=0.1)
        res = run(image(), [HEAD_BOX], b=weak)
        self.assertIsNone(res["selected"])
        self.assertTrue(res["refusal_reason"].startswith("no_reliable_current_wrist"))
        no_wrist = body()
        no_wrist["wrists"] = {}
        self.assertIsNone(run(image(), [HEAD_BOX], b=no_wrist)["selected"])

    def test_geometry_and_native_bounds_are_validated(self):
        im = image()
        with self.assertRaises(ValueError):
            h.measure_proposals(im.astype(np.float32), (0, 0), (400, 720), body(), [HEAD_BOX])
        with self.assertRaises(ValueError):
            h.measure_proposals(im[:, :, 0], (0, 0), (400, 720), body(), [HEAD_BOX])
        with self.assertRaises(ValueError):
            h.measure_proposals(im, (10, 0), (400, 720), body(), [HEAD_BOX])
        with self.assertRaises(ValueError):
            h.measure_proposals(im, (0.0, 0.0), (400, 720), body(), [HEAD_BOX])
        with self.assertRaises(ValueError):
            h.measure_proposals(im, (0, 0), (400.0, 720), body(), [HEAD_BOX])
        with self.assertRaises(ValueError):
            h.measure_proposals(im, (-1, 0), (400, 720), body(), [HEAD_BOX])
        # no frame argument exists: only geometry provenance is checked, not staleness
    def test_malformed_detections_and_body_are_refused(self):
        im = image()
        for bad in (
            {"confidence": 0.5, "class_id": 0},                                # no box
            det(0, 0, 10, 10, cls=True),                                       # bool class
            det(0, 0, 10, 10, cls="0"),                                        # str class
            {"xyxy": [0, 0, 10, 10], "class_id": 0},                           # no confidence
            det(0, 0, 10, 10, conf=float("nan")),                              # nan conf
            det(0, 0, 10, 10, conf=1.5),                                       # conf > 1
            det(0, 0, 10, 10, conf=-0.1),                                      # conf < 0
            {"xyxy": [0, 0, 10], "confidence": 0.5, "class_id": 0},            # bad arity
            det(10, 0, 0, 10),                                                 # non-increasing x
            det(0, 10, 10, 0),                                                 # non-increasing y
            det(-1, 0, 10, 10),                                                # outside crop
            det(0, 0, 401, 10),                                                # outside crop
            det(0, 0, float("inf"), 10),                                       # non-finite box
        ):
            with self.subTest(det=bad):
                with self.assertRaises(ValueError):
                    h.measure_proposals(im, (0, 0), (400, 720), body(), [bad])
        for bad_body in (
            {**body(), "normalized": True},
            {**body(), "roi_xywh": [0, 0, -5, 10]},
            {**body(), "roi_xywh": [300, 200, 200, 200]},
            {**body(), "roi_xywh": [0, 0, 10]},
            {**body(), "wrists": "nope"},
            {**body(), "wrists": {"wr_l": {"x": float("nan"), "y": 3.0, "score": 0.9}}},
            {**body(), "wrists": {"wr_l": {"nx": 0.1, "ny": 0.2, "score": 0.9}}},
            {**body(), "wrists": {"wr_l": {"x": 500.0, "y": 3.0, "score": 0.9}}},
        ):
            with self.subTest(body=bad_body):
                with self.assertRaises(ValueError):
                    h.measure_proposals(im, (0, 0), (400, 720), bad_body, [HEAD_BOX])

    def test_empty_proposals_are_withheld_and_wrong_model_class_is_refused(self):
        im = image()
        self.assertIsNone(run(im, [])["selected"])
        self.assertEqual(run(im, [])["refusal_reason"], "no_class0_proposal")
        other = [det(248, 282, 292, 318, cls=1)]
        with self.assertRaises(ValueError):run(im, other)

    def test_high_confidence_is_considered_but_never_used_to_choose(self):
        # A single box above the floor but far below any operating threshold still
        # proposes, and its confidence magnitude never enters the decision.
        for confidence in (0.001, 0.05, 0.9):
            res = run(image(), [det(240, 283, 272, 317, conf=confidence)])
            self.assertIsNotNone(res["selected"])
            self.assertEqual(res["selected"]["support_sha256"],
                             run(image(), [det(240, 283, 272, 317, conf=0.001)])["selected"]["support_sha256"])

    def test_core_support_outside_proposal_box_is_refused(self):
        # The narrow strip passes through the core: the centroid is inside but the
        # measured support pixels are not, so the bbox cannot be the answer.
        real=run(image(), [HEAD_BOX])['selected']
        core={**real,'accepted':True,'refusal_reasons':[]}
        x,y=core['centroid_native']
        with patch.object(shaft_head,'_measure_patch',return_value=[core]):
            res = run(image(), [det(x+.5-.05, y-8, x+.5+.05, y+8)])
        self.assertIsNone(res["selected"])
        self.assertIn("core_support_not_inside_proposal_box",
                      res["refusals"][0]["refusal_reasons"])
        self.assertTrue(res["refusals"][0]["core_accepted"])

    def test_core_accepted_but_no_thin_shaft_ribbon_is_reported(self):
        false_ribbon = {"accepted": False, "same_sign_fraction": 0.0,
                        "sign_counts": {"dark": 0, "bright": 0},
                        "offsets_native_px": [], "profiles": []}
        with patch.object(shaft_head, "_shaft_ribbon", return_value=false_ribbon):
            res = run(image(), [HEAD_BOX])
        self.assertIsNone(res["selected"])
        self.assertTrue(any("core_accepted_but_no_thin_shaft_ribbon"
                            in entry["refusal_reasons"] for entry in res["refusals"]))
        self.assertTrue(all(entry["core_accepted"] for entry in res["refusals"]
                            if "core_accepted_but_no_thin_shaft_ribbon"
                            in entry["refusal_reasons"]))

    def test_bright_round_ball_is_refused_and_patch_clip_is_refused(self):
        self.assertIsNone(run(image(bright=True), [det(250, 285, 285, 315)])["selected"])
        near_edge = np.full((720, 400, 3), 170, np.uint8)
        cv2.line(near_edge, (10, 350), (10, 250), (30,) * 3, 2)
        cv2.ellipse(near_edge, (10, 250), (6, 8), 0, 0, 360, (30,) * 3, -1)
        res = run(near_edge, [det(0, 240, 22, 262)], b=body(wrist=(10, 350)))
        self.assertIsNone(res["selected"])
        self.assertIn("patch_clipped_by_crop", res["proposals"][0]["refusal_reasons"])

    def test_no_input_is_mutated_and_evidence_is_json_serialisable(self):
        im = image()
        b = body()
        dets = [HEAD_BOX, det(310, 562, 352, 600, conf=0.9)]
        im_before, body_before, dets_before = im.copy(), copy.deepcopy(b), copy.deepcopy(dets)
        res = h.measure_proposals(im, (0, 0), (400, 720), b, dets)
        np.testing.assert_array_equal(im, im_before)
        self.assertEqual(b, body_before)
        self.assertEqual(dets, dets_before)
        json.dumps(res, allow_nan=False, sort_keys=True)

    def test_profile_records_research_only_scope_and_dependencies(self):
        self.assertTrue(h.PROFILE["research_only"])
        self.assertFalse(h.PROFILE["production_eligible"])
        self.assertFalse(h.PROFILE["confidence_based_selection"])
        self.assertFalse(h.PROFILE["temporal_policy"])
        self.assertFalse(h.PROFILE["line_extrapolation"])
        self.assertIsNone(h.PROFILE["exported_confidence"])
        self.assertFalse(h.PROFILE["physical_identity_established"])
        self.assertEqual(h.PROFILE["proposal_confidence_floor"], 0.001)
        self.assertTrue(h.PROFILE["dependencies"])

    def test_proposal_cap_fails_closed_before_measurement(self):
        dets = [det(248, 282, 292, 318) for _ in range(h.MAX_PROPOSALS + 1)]
        res = run(image(), dets)
        self.assertTrue(res["cap_exceeded"])
        self.assertEqual(res["refusal_reason"], "proposal_cap_exceeded")
        self.assertIsNone(res["selected"])
        self.assertEqual(res["segment_count"], 0)
        below_floor=[{**d,'confidence':0.0001} for d in dets]
        self.assertEqual(run(image(),below_floor)['refusal_reason'],'proposal_cap_exceeded')

    def test_centroid_outside_box_is_refused_even_when_most_support_is_inside(self):
        real=run(image(),[HEAD_BOX])['selected']
        # Isolate containment with support whose mean lies outside a narrow box
        # even though eight of ten pixels are inside it.
        pixels=[[265,296+j] for j in range(8)]+[[270,299],[270,300]]
        centroid=np.mean(pixels,axis=0).tolist()
        core={**real,'support_pixels_native':pixels,'centroid_native':centroid,'accepted':True,'refusal_reasons':[]}
        with patch.object(shaft_head,'_measure_patch',return_value=[core]):
            res=run(image(),[det(264,290,265.8,310)])
        self.assertIsNone(res['selected'])
        self.assertTrue(any('core_centroid_outside_proposal_box' in r['refusal_reasons'] for r in res['refusals']))

    def test_native_crop_origin_is_added_and_not_treated_as_box_centre(self):
        full = np.full((720, 500, 3), 170, np.uint8)
        cv2.line(full, (200, 300), (365, 300), (30,) * 3, 2)
        cv2.ellipse(full, (365, 300), (8, 5), 0, 0, 360, (30,) * 3, -1)
        crop = full[:, 100:400]
        res = h.measure_proposals(crop, (100, 0), (500, 720), body(wrist=(200, 300)),
                                  [HEAD_BOX])
        self.assertIsNotNone(res["selected"])
        expected=run(image(),[HEAD_BOX])['selected']['centroid_native']
        np.testing.assert_allclose(res["selected"]["centroid_native"], [expected[0]+100,expected[1]], atol=1e-9)
        self.assertEqual(res["selected"]["proposal_box_native_xyxy"],
                         [340.0, 283.0, 372.0, 317.0])


if __name__ == "__main__":
    unittest.main()
