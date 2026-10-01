"""Behaviour tests for the isolated masked-foreground clubhead candidate.

Two kinds of test live here and they must not be confused:

* **policy tests** -- pure, no cv2, no video. They feed synthetic ``CandidateMatch``
  numbers to ``ForegroundPolicy`` and check what the policy does with them: the
  identity gate, the ambiguity gate, the bounded step, the loss, the bounded
  re-acquisition window and its consecutive confirmation, the structural-cut reset,
  and the invariant that a frame with no accepted measurement never carries a
  coordinate. Synthetic numbers prove the *policy*, never a tracker's accuracy.
* **synthetic-scene tests** -- generated frames in a temporary directory, written
  from code. They check that the measurement layer follows the small moving
  foreground past a larger static neighbour, abstains when two candidates are
  indistinguishable, refuses a flat or edge-clipped seed, clamps (rather than
  rejects) a search window at a frame edge, and re-acquires after a genuine
  disappearance. They are evidence of BEHAVIOUR on synthetic patterns: they are not
  a performance claim about any real clip.

The runner is exercised end to end on a synthetic clip as well, and the one promise
worth testing hardest is checked there: with ``--variant clubhead_foreground`` the
BALL leg is byte-identical to ``--variant baseline``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostcaddie.demo import foreground_tracker as fg  # noqa: E402
from ghostcaddie.demo import target_benchmark as bench  # noqa: E402

try:  # pragma: no cover - import guard, mirrors tests/demo_fixtures.py
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    CV2_AVAILABLE = False
    np = None

REQUIRES_CV2 = unittest.skipUnless(CV2_AVAILABLE,
                                  "this interpreter cannot decode/encode video")

TOOL = ROOT / "tools" / "tiger_target_benchmark.py"
MODULE = ROOT / "ghostcaddie" / "demo" / "foreground_tracker.py"


# --------------------------------------------------------------------------
# pure-policy helpers

def match(frame, x, y, *, score=0.9, seed_score=0.9, second=0.1,
          transform="rot+0.0/scale1.00"):
    return fg.CandidateMatch(int(frame), True, x=float(x), y=float(y),
                             score=float(score), seed_score=float(seed_score),
                             second_score=float(second), transform=transform)


SEED_FRAME = 10
SEED_X, SEED_Y = 50.0, 40.0


def new_policy(**overrides):
    return fg.ForegroundPolicy(seed_frame=SEED_FRAME, seed_x=SEED_X, seed_y=SEED_Y,
                               **overrides)


def drive(policy, frames, observation_for=None, *, cuts=()):
    """Step a policy through frames, asking it where to look each time."""
    requests = {}
    for frame in frames:
        if frame in cuts:
            policy.step(frame, cut=True)
            continue
        request = policy.request_for(frame)
        requests[frame] = request
        observation = observation_for(frame) if observation_for else None
        policy.step(frame, observation)
    return requests


def run_after_loss(policy, start_frame, *, confirmations, offset=(3.0, 0.0),
                   score=0.9, seed_score=0.9, second=0.1, frames=None):
    """Produce a loss, then the requested number of confirming candidates."""
    frames = frames if frames is not None else range(start_frame, start_frame + 20)
    requests = {}
    for index, frame in enumerate(frames):
        request = policy.request_for(frame)
        requests[frame] = request
        if index < 3:
            policy.step(frame, None)                      # three refusals -> loss
        elif index < 3 + confirmations:
            base = policy.reacq_anchor or (SEED_X, SEED_Y)
            policy.step(frame, match(frame, base[0] + offset[0], base[1] + offset[1],
                                     score=score, seed_score=seed_score, second=second))
        else:
            policy.step(frame, None)
    return requests


def null_coordinates(decisions):
    """Every non-observation must carry no coordinate and no score, without exception."""
    bad = [d.source_frame for d in decisions
           if d.status in ("missing", "lost")
           and (d.x is not None or d.y is not None or d.score is not None)]
    return bad


# --------------------------------------------------------------------------
# the predictions contract, exercised on the candidate's own output

def contract_payloads(points, *, labels=(), seed_frame=SEED_FRAME, window=(0, 40),
                      cuts=(), frame=64, seed_x=SEED_X, seed_y=SEED_Y,
                      target="clubhead"):
    """Build a validated dataset and a prediction payload for the given points."""
    clip = {"id": "clip_a", "path": "/nonexistent/clip_a.mp4", "sha256": "0" * 64,
            "width": frame, "height": frame, "frame_count": 64, "fps_rational": "30/1",
            "stroke_type": "full_swing", "split": "development", "tags": [],
            "window": {"start_frame": window[0], "end_frame": window[1]},
            "cut_frames": list(cuts),
            "targets": {target: {"seed": {"frame": seed_frame, "x": seed_x,
                                          "y": seed_y, "note": "synthetic seed"},
                                 "options": {}, "labels": list(labels)}}}
    dataset = bench.validate_dataset(
        {"schema_version": bench.DATASET_SCHEMA,
         "provenance": dict(bench.REQUIRED_PROVENANCE), "clips": [clip]},
        verify_hashes=False)
    payload = {"schema_version": bench.PREDICTIONS_SCHEMA, "variant": "candidate",
               "clips": [{"id": "clip_a", "sha256": "0" * 64,
                          "targets": {target: {"seed": {"frame": seed_frame,
                                                        "x": seed_x, "y": seed_y},
                                               "points": list(points)}}}]}
    return dataset, payload


def contract_points(points, **kwargs):
    """Validate candidate points through the benchmark's real contract.

    Nothing here is stubbed: the evaluator's own ``validate_predictions`` decides
    whether the candidate's statuses, origins, methods and coordinates are legal.
    """
    target = kwargs.get("target", "clubhead")
    dataset, payload = contract_payloads(points, **kwargs)
    validated = bench.validate_predictions(payload, dataset)
    return validated["clips"][0]["targets"][target]["points"]


def visible_label(frame: int, x: float = 50.0, y: float = 40.0) -> dict:
    return {"frame": int(frame), "state": "visible", "x": x, "y": y,
            "tolerance_px": 4.0, "note": "synthetic visible label",
            "reviewer": "synthetic fixture"}


def occluded_label(frame: int) -> dict:
    return {"frame": int(frame), "state": "occluded", "x": None, "y": None,
            "tolerance_px": None, "note": "synthetic occlusion",
            "reviewer": "synthetic fixture"}


# --------------------------------------------------------------------------
# synthetic scenes (generated from code; no source footage is used)

TURF = 70
HEAD = 210


def _turf(height, width):
    image = np.full((height, width), TURF, np.uint8)
    image += (np.indices((height, width)).sum(axis=0) % 3).astype(np.uint8)
    return image


def _draw_head(image, x, y):
    image[y - 6:y + 7, x - 6:x + 7] = HEAD
    image[y - 3:y + 4, x - 1:x + 2] = 60
    image[y - 6:y + 7, x - 6:x - 4] = 40
    return image


def _draw_ball(image, x, y):
    cv2.circle(image, (int(x), int(y)), 3, 245, -1)
    return image


def head_scene(heads, *, leg=None, size=(160, 200), colour=True):
    """Grey turf, the given heads, and optionally one large static bright structure."""
    image = _turf(*size)
    if leg is not None:
        lx, ly, lw, lh = leg
        image[ly:ly + lh, lx:lx + lw] = 150
        image[ly:ly + lh, lx + 8:lx + 12] = 90
        image[ly:ly + lh, lx + 25:lx + 29] = 90
    for x, y in heads:
        _draw_head(image, x, y)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if colour else image


def head_frames(count, *, start=(60, 60), step=(3, 2), leg=None, head_from=None,
                absent=None, size=(160, 200)):
    """Ascending ``(frame, image)`` pairs with the head moving by ``step``."""
    frames = []
    for index in range(count):
        x = start[0] + step[0] * index
        y = start[1] + step[1] * index
        heads = [] if (absent and absent[0] <= index <= absent[1]) else [(x, y)]
        frames.append((index, head_scene(heads, leg=leg, size=size)))
    return frames


def track(frames, *, seed_frame=0, seed_x=60.0, seed_y=60.0, cuts=()):
    return fg.track_sequence(frames, seed_frame=seed_frame, seed_x=seed_x,
                             seed_y=seed_y,
                             interval_frames=[frame for frame, _ in frames],
                             cut_frames=cuts)


# --------------------------------------------------------------------------
# the policy

class PolicyTests(unittest.TestCase):
    def test_the_seed_is_the_only_operator_anchor(self):
        policy = new_policy()
        decision = policy.step(SEED_FRAME)
        self.assertEqual(decision.status, "manual_anchor")
        self.assertEqual(decision.origin, "operator")
        self.assertEqual(decision.method, "operator_placement")
        self.assertEqual((decision.x, decision.y), (SEED_X, SEED_Y))
        contract_points([decision.to_point()])

    def test_a_frames_before_the_seed_are_missing_and_say_why(self):
        policy = new_policy()
        self.assertEqual(policy.request_for(SEED_FRAME), None)
        decision = policy.step(SEED_FRAME - 1)
        self.assertEqual(decision.status, "missing")
        self.assertIn("before the seed", decision.note)
        self.assertEqual(decision.x, None)

    def test_three_consecutive_refusals_declare_the_loss_at_the_third_frame(self):
        """Causal: the loss starts at the third refusal, and nothing is rewritten."""
        policy = new_policy()
        policy.step(SEED_FRAME)
        decisions = [policy.step(frame) for frame in (11, 12, 13)]
        self.assertEqual([decision.status for decision in decisions],
                         ["missing", "missing", "lost"])
        self.assertEqual([decision.source_frame for decision in policy.decisions
                          if decision.status == "lost"], [13])
        self.assertIn("loss_frames=3", decisions[-1].note)
        self.assertIn("not rewritten", decisions[-1].note)
        self.assertIn("short gap", decisions[0].note)
        self.assertEqual(null_coordinates(policy.decisions), [])
        contract_points([d.to_point() for d in policy.decisions])

    def test_a_refused_frame_is_never_a_point_and_never_a_stale_marker(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        policy.step(11, match(11, 52.0, 40.0))
        refused = policy.step(12, match(12, 90.0, 40.0))       # step gate
        self.assertEqual(refused.status, "missing")
        self.assertIsNone(refused.x)
        self.assertIsNone(refused.y)
        self.assertIsNone(refused.score)
        self.assertIn("step", refused.note)

    def test_each_gate_refuses_on_its_own(self):
        cases = {
            "score": dict(score=0.10),
            "seed_identity": dict(seed_score=0.10),
            "unambiguous": dict(second=0.895),
            "step": dict(x=SEED_X + 40.0),
        }
        for gate, overrides in cases.items():
            with self.subTest(gate=gate):
                policy = new_policy()
                policy.step(SEED_FRAME)
                overrides = dict(overrides)
                x = overrides.pop("x", SEED_X + 2.0)
                decision = policy.step(11, match(11, x, SEED_Y, **overrides))
                self.assertEqual(decision.status, "missing")
                self.assertIn("refused: " + gate, decision.note)
                self.assertEqual(decision.gates.get(gate), False)
                self.assertIsNone(decision.x)
                if gate == "step":
                    # while tracking, the consecutive gate and the global gate are the
                    # same bound, and both must refuse
                    self.assertEqual(decision.gates.get("global_step"), False)

    def test_the_extrapolated_search_centre_is_never_reported_as_a_point(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        policy.step(11, match(11, 54.0, SEED_Y))
        request = policy.request_for(12)
        self.assertEqual(request.anchor_x, 58.0)                # last + velocity
        self.assertFalse(request.reacquiring)
        decision = policy.step(12)                              # nothing measured
        self.assertEqual(decision.status, "missing")
        self.assertIsNone(decision.x)
        self.assertNotIn(58.0, [d.x for d in policy.decisions if d.x is not None])

    def test_square_search_corner_is_still_limited_by_global_distance(self):
        """A 48 px search half-width admits corners farther than the 52 px bound."""
        policy = new_policy()
        policy.step(SEED_FRAME)
        for frame in (11, 12, 13):
            policy.step(frame)
        x, y = policy.reacq_anchor
        first = policy.step(14, match(14, x + 30, y + 30))
        second = policy.step(15, match(15, x + 45, y + 45))
        self.assertTrue(first.gates['global_step'])
        self.assertTrue(second.gates['step'])
        self.assertFalse(second.gates['global_step'])
        self.assertEqual(second.status, 'lost')
        self.assertIsNone(second.x)

    def test_re_acquisition_never_walks_away_from_the_physical_anchor(self):
        """The inbox's audit case: pending steps pass, the global distance must not.

        Anchor at the seed: candidate one at +40 (inside the bounded radius, passes the
        consecutive gate), candidate two at +75 from the anchor (its step from candidate
        one is only 35, so consecutive compatibility alone would accept it). The global
        gate must refuse it: 75 > REACQ_MAX_STEP_PX.
        """
        policy = new_policy()
        policy.step(SEED_FRAME)
        for frame in (11, 12, 13):
            policy.step(frame, None)                      # loss at 13
        anchor = policy.reacq_anchor
        self.assertEqual(anchor, (SEED_X, SEED_Y))
        requests = {}
        for frame, offset in ((14, 40.0), (15, 75.0), (16, 78.0)):
            requests[frame] = policy.request_for(frame)
            policy.step(frame, match(frame, anchor[0] + offset, anchor[1]))
        first, second, third = policy.decisions[-3:]
        self.assertEqual(first.status, "lost")
        self.assertIn("confirmation 1/2", first.note)
        self.assertEqual(first.gates["step"], True)
        self.assertEqual(first.gates["global_step"], True)
        self.assertEqual(second.gates["step"], True)          # 35 px from the first
        self.assertEqual(second.gates["global_step"], False)  # 75 px from the anchor
        self.assertIn("global_step", second.note)
        self.assertIsNone(second.x)
        self.assertEqual([d for d in policy.decisions if d.reacquired], [])
        # ...and the search stayed centred on the physical anchor the whole time
        for request in requests.values():
            self.assertEqual((request.anchor_x, request.anchor_y), anchor)
        self.assertEqual(null_coordinates(policy.decisions), [])

    def test_the_re_acquisition_search_is_not_re_centred_on_a_candidate(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        for frame in (11, 12, 13):
            policy.step(frame, None)
        anchor = policy.reacq_anchor
        policy.step(14, match(14, anchor[0] + 30.0, anchor[1]))
        self.assertEqual(policy.reacq_confirmations, 1)
        request = policy.request_for(15)
        self.assertEqual((request.anchor_x, request.anchor_y), anchor)
        self.assertIn("never re-centred", request.because)
        # the NEXT confirmation must still be compatible with the previous one
        self.assertEqual(policy._pending_origin(), (anchor[0] + 30.0, anchor[1]))

    def test_the_search_radius_and_threshold_change_inside_the_window(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        tracking = policy.request_for(11)
        self.assertEqual(tracking.radius_px, fg.SEARCH_RADIUS_PX)
        requests = run_after_loss(policy, 11, confirmations=0, frames=range(11, 17))
        inside = [request for frame, request in requests.items()
                  if request is not None and request.reacquiring]
        self.assertTrue(inside, "no re-acquisition request was ever made")
        self.assertEqual(inside[0].radius_px, fg.REACQ_RADIUS_PX)
        self.assertIn("last OBSERVED physical position", inside[0].because)

    def test_a_candidate_between_the_two_thresholds_cannot_re_acquire(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        run_after_loss(policy, 11, confirmations=1, score=0.60)
        refused = [d for d in policy.decisions
                   if d.status == "lost" and d.measurement.get("found")]
        self.assertTrue(refused, "no measured candidate was recorded inside the window")
        for decision in refused:
            self.assertEqual(decision.gates.get("score"), False)
            self.assertIn("score", decision.note)
            self.assertIsNone(decision.x)
        self.assertEqual([d for d in policy.decisions if d.reacquired], [])

    def test_re_acquisition_needs_consecutive_confirmation_frames(self):
        # one candidate is not a re-acquisition: the frame stays lost and is never
        # back-filled by a later frame that completes the run
        policy = new_policy()
        policy.step(SEED_FRAME)
        run_after_loss(policy, 11, confirmations=1)
        held = [d for d in policy.decisions if "confirmation 1/2" in d.note]
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0].status, "lost")
        self.assertIsNone(held[0].x)
        self.assertIsNone(held[0].y)
        self.assertEqual([d for d in policy.decisions if d.reacquired], [])

        # two consecutive measured confirmations: the SECOND frame is the re-acquired
        # one, the first stays lost (no back-fill)
        policy = new_policy()
        policy.step(SEED_FRAME)
        run_after_loss(policy, 11, confirmations=2)
        reacquired = [d for d in policy.decisions if d.reacquired]
        self.assertEqual(len(reacquired), 1)
        confirmed = reacquired[0]
        self.assertEqual(confirmed.source_frame, 15)
        self.assertEqual(confirmed.status, "observed_template")
        self.assertEqual(confirmed.origin, "automatic")
        self.assertEqual(confirmed.method, fg.METHOD_REACQUIRED)
        self.assertIn("re-acquired AT THIS FRAME", confirmed.note)
        self.assertIn("not backfilled", confirmed.note)
        first_candidate = [d for d in policy.decisions if d.source_frame == 14][0]
        self.assertEqual(first_candidate.status, "lost")
        self.assertIsNone(first_candidate.x)
        self.assertFalse(first_candidate.reacquired)
        # a re-acquired run opens a NEW segment: nothing bridges the loss
        self.assertIsNotNone(confirmed.segment)
        self.assertTrue(int(confirmed.segment or 0) > 0)
        for decision in policy.decisions:
            if not decision.observed:
                self.assertIsNone(decision.segment)
        self.assertEqual(null_coordinates(policy.decisions), [])
        contract_points([d.to_point() for d in policy.decisions])

    def test_reported_frames_do_not_change_when_later_frames_are_appended(self):
        """Prefix invariance: the report for frames [0..N] must not depend on N."""

        def scripted(stop_frame):
            policy = new_policy()
            policy.step(SEED_FRAME)
            for frame in range(11, stop_frame + 1):
                if frame in (11, 12, 13):
                    policy.step(frame, None)              # -> loss at frame 13
                elif frame in (14, 15):
                    anchor = policy.reacq_anchor
                    policy.step(frame, match(frame, anchor[0] + 2.0, anchor[1]))
                else:
                    policy.step(frame, None)
            return policy

        short = scripted(16)
        long = scripted(24)
        short_raw = [decision.to_raw() for decision in short.decisions]
        long_raw = [decision.to_raw() for decision in long.decisions[:len(short_raw)]]
        self.assertEqual(short_raw, long_raw)
        self.assertEqual([d.source_frame for d in short.decisions],
                         list(range(SEED_FRAME, 17)))

    def test_a_short_confirmation_run_is_broken_without_rewriting_its_frames(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        frames = list(range(11, 25))
        for index, frame in enumerate(frames):
            if index < 3:
                policy.step(frame, None)                      # loss at 13
            elif index == 3:
                anchor = policy.reacq_anchor
                policy.step(frame, match(frame, anchor[0] + 3.0, anchor[1]))
            else:
                policy.step(frame, None)
        reacquired = [d for d in policy.decisions if d.reacquired]
        self.assertEqual(reacquired, [])
        self.assertEqual(null_coordinates(policy.decisions), [])
        first_candidate = [d for d in policy.decisions if d.source_frame == 14][0]
        self.assertEqual(first_candidate.status, "lost")
        self.assertIsNone(first_candidate.x)
        self.assertIn("confirmation 1/2", first_candidate.note)
        self.assertTrue(any("confirmation run of 1 frame(s) was broken" in d.note
                            for d in policy.decisions))
        self.assertTrue(any("no rewrite" in d.note for d in policy.decisions))

    def test_the_re_acquisition_window_is_finite_and_terminal(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        frames = list(range(11, 11 + 3 + fg.REACQ_WINDOW_FRAMES + 4))
        requests = run_after_loss(policy, 11, confirmations=0, frames=frames)
        inside = [frame for frame, request in requests.items()
                  if request is not None and request.reacquiring]
        self.assertEqual(len(inside), fg.REACQ_WINDOW_FRAMES)
        self.assertEqual(inside, list(range(inside[0], inside[0] + fg.REACQ_WINDOW_FRAMES)))
        self.assertTrue(all(requests[frame] is None
                            for frame in frames if frame > inside[-1]))
        tail = [d for d in policy.decisions if d.status == "lost"]
        self.assertTrue(all(d.x is None and d.y is None for d in tail))
        self.assertIn("window is over", tail[-1].note)
        self.assertEqual([d for d in policy.decisions if d.reacquired], [])
        contract_points([d.to_point() for d in policy.decisions])

    def test_a_structural_cut_resets_and_nothing_re_acquires_across_it(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        policy.step(11, match(11, 53.0, 40.0))
        run_after_loss(policy, 12, confirmations=0, frames=range(12, 16))
        self.assertTrue(policy.lost)
        cut = policy.step(16, cut=True)
        self.assertEqual(cut.status, "missing")
        self.assertIn("cut", cut.note)
        self.assertFalse(policy.lost)
        for frame in range(17, 24):
            self.assertIsNone(policy.request_for(frame))
            decision = policy.step(frame, match(frame, 53.0, 40.0,
                                                score=0.99, seed_score=0.99))
            self.assertEqual(decision.status, "missing")
            self.assertIsNone(decision.x)
        self.assertTrue(policy.summary()["structural_cut_reset"])
        self.assertEqual(null_coordinates(policy.decisions), [])

    def test_a_refused_seed_measures_nothing_and_never_invents_a_point(self):
        policy = new_policy()
        policy.refuse_seed("seed_patch_is_flat_low_texture")
        policy.step(SEED_FRAME)
        for frame in range(11, 18):
            self.assertIsNone(policy.request_for(frame))
            decision = policy.step(frame, match(frame, 52.0, 40.0, score=0.99,
                                                seed_score=0.99))
            self.assertEqual(decision.status, "missing")
            self.assertIn("seed_patch_is_flat_low_texture", decision.note)
            self.assertIsNone(decision.x)
        contract_points([d.to_point() for d in policy.decisions])

    def test_observed_points_are_only_ever_measured_positions(self):
        policy = new_policy()
        policy.step(SEED_FRAME)
        for index, frame in enumerate(range(11, 16)):
            policy.step(frame, match(frame, 52.0 + 2.0 * index, 40.0))
        observed = [d for d in policy.decisions if d.observed]
        self.assertEqual([d.x for d in observed[1:]],
                         [52.0, 54.0, 56.0, 58.0, 60.0])
        self.assertEqual(null_coordinates(policy.decisions), [])


class PolicyContractTests(unittest.TestCase):
    """The candidate's vocabulary must be the evaluator's vocabulary."""

    def test_the_two_foreground_methods_are_accepted_by_the_contract(self):
        self.assertIn(fg.METHOD_FOREGROUND, bench.AUTOMATIC_METHODS)
        self.assertIn(fg.METHOD_REACQUIRED, bench.AUTOMATIC_METHODS)

    def test_the_candidate_statuses_are_the_contract_statuses(self):
        self.assertEqual(fg.STATUS_OBSERVED, "observed_template")
        self.assertEqual(fg.STATUS_MANUAL, "manual_anchor")
        self.assertEqual(fg.STATUS_MISSING, "missing")
        self.assertEqual(fg.STATUS_LOST, "lost")

    def test_the_contract_accepts_a_foreground_observation_and_a_reacquired_one(self):
        points = [
            {"source_frame": 10, "status": "manual_anchor", "origin": "operator",
             "method": "operator_placement", "x": 50.0, "y": 40.0, "score": None},
            {"source_frame": 11, "status": "observed_template", "origin": "automatic",
             "method": fg.METHOD_FOREGROUND, "x": 52.0, "y": 40.0, "score": 0.9},
            {"source_frame": 12, "status": "lost", "origin": "derived", "method": "",
             "x": None, "y": None, "score": None},
            {"source_frame": 13, "status": "observed_template", "origin": "automatic",
             "method": fg.METHOD_REACQUIRED, "x": 55.0, "y": 40.0, "score": 0.9},
        ]
        validated = contract_points(points, window=(0, 40))
        self.assertEqual([p["method"] for p in validated],
                         ["operator_placement", fg.METHOD_FOREGROUND, "",
                          fg.METHOD_REACQUIRED])

    def test_an_invented_method_is_still_refused(self):
        points = [{"source_frame": 11, "status": "observed_template",
                   "origin": "automatic", "method": "neural_magic", "x": 52.0,
                   "y": 40.0, "score": 0.9}]
        with self.assertRaises(bench.TargetBenchmarkError) as ctx:
            contract_points(points)
        self.assertIn("is not a tracker method", str(ctx.exception))

    def test_a_foreground_observation_may_not_claim_operator_origin(self):
        points = [{"source_frame": 11, "status": "observed_template",
                   "origin": "operator", "method": fg.METHOD_FOREGROUND, "x": 52.0,
                   "y": 40.0, "score": 0.9}]
        with self.assertRaises(bench.TargetBenchmarkError):
            contract_points(points)

    def test_a_cut_frame_point_is_refused_by_the_contract(self):
        points = [{"source_frame": 11, "status": "observed_template",
                   "origin": "automatic", "method": fg.METHOD_FOREGROUND, "x": 52.0,
                   "y": 40.0, "score": 0.9}]
        with self.assertRaises(bench.TargetBenchmarkError) as ctx:
            contract_points(points, cuts=(11,))
        self.assertIn("cut frame", str(ctx.exception))

    def test_the_contract_counts_the_confirmation_delay_in_recovery_latency(self):
        """The causal rule must show up as honest reported latency.

        Labels: occluded at 20, visible 21..24. The candidate's first measured
        confirmation lands at 21 but is reported LOST (confirmation 1/2); the frame it
        is allowed to report is 22. The evaluator therefore measures a latency of 1,
        which is the confirmation delay -- a back-filling policy would have reported a
        point at 21 and claimed zero latency.
        """
        labels = [occluded_label(20)] + [visible_label(frame) for frame in (21, 22, 23, 24)]
        points = [
            {"source_frame": 10, "status": "manual_anchor", "origin": "operator",
             "method": "operator_placement", "x": 50.0, "y": 40.0, "score": None},
            {"source_frame": 21, "status": "lost", "origin": "derived", "method": "",
             "x": None, "y": None, "score": None},
            {"source_frame": 22, "status": "observed_template", "origin": "automatic",
             "method": fg.METHOD_REACQUIRED, "x": 50.0, "y": 40.0, "score": 0.9},
            {"source_frame": 23, "status": "observed_template", "origin": "automatic",
             "method": fg.METHOD_FOREGROUND, "x": 50.5, "y": 40.0, "score": 0.9},
            {"source_frame": 24, "status": "observed_template", "origin": "automatic",
             "method": fg.METHOD_FOREGROUND, "x": 51.0, "y": 40.0, "score": 0.9},
        ]
        dataset, payload = contract_payloads(points, labels=labels)
        predictions = bench.validate_predictions(payload, dataset)
        result = bench.evaluate(dataset, predictions)["clips"]["clip_a"]["targets"]["clubhead"]
        recovery = result["recovery"]
        self.assertEqual(recovery["evaluated_transitions"], 1)
        case = recovery["cases"][0]
        self.assertEqual(case["occlusion_frame"], 20)
        self.assertEqual(case["first_visible_frame"], 21)
        self.assertEqual(case["recovered_frame"], 22)
        self.assertEqual(case["latency_frames"], 1)
        # the unreported confirmation frame is not evidence of anything
        self.assertEqual(result["accounting"]["missing_or_lost_points_emitted"], 1)
        deviated = [entry for entry in result["per_frame"]
                    if entry["outcome"] in ("correct", "incorrect")]
        self.assertEqual([entry["frame"] for entry in deviated], [22, 23, 24])


class ProfileTests(unittest.TestCase):
    #: Names re-exported from ``demo.tracking``: the candidate uses the tracker's own
    #: vocabulary, so they are asserted against that module instead of PROFILE.
    REEXPORTED = {"ORIGIN_AUTOMATIC", "ORIGIN_DERIVED", "ORIGIN_OPERATOR",
                  "STATUS_MANUAL", "STATUS_OBSERVED", "STATUS_MISSING", "STATUS_LOST"}
    #: Declared per run (and in the report), not a tunable constant.
    DECLARED_ELSEWHERE = {"LIMITATIONS"}

    def test_every_module_constant_is_declared_in_the_profile(self):
        special = {"PROFILE", "TRANSFORMS"} | self.REEXPORTED | self.DECLARED_ELSEWHERE
        missing = []
        for name in dir(fg):
            if not name.isupper() or name in special:
                continue
            value = getattr(fg, name)
            key = name.lower()
            if key in fg.PROFILE or any(key in existing for existing in fg.PROFILE):
                continue
            if value in fg.PROFILE.values():
                continue
            missing.append(name)
        self.assertEqual(missing, [])

    def test_the_reexported_vocabulary_is_the_trackers_own(self):
        from ghostcaddie.demo import tracking

        for name in self.REEXPORTED:
            self.assertIs(getattr(fg, name), getattr(tracking, name))
        self.assertEqual(fg.PROFILE["statuses"],
                         [fg.STATUS_MANUAL, fg.STATUS_OBSERVED, fg.STATUS_MISSING,
                          fg.STATUS_LOST])

    def test_the_profile_carries_the_transform_list_it_runs(self):
        self.assertEqual(fg.PROFILE["transforms"],
                         [fg.transform_label(angle, scale)
                          for angle, scale in fg.TRANSFORMS])
        self.assertEqual(len(fg.TRANSFORMS), len(fg.ROTATIONS_DEG) * len(fg.SCALES))

    def test_the_profile_states_that_it_is_not_driven_by_reference_options(self):
        self.assertFalse(fg.PROFILE["driven_by_reference_options"])
        self.assertTrue(fg.PROFILE["one_forward_leg_only"])


class ProvenanceTests(unittest.TestCase):
    """The candidate reads a seed, source structure and nothing else."""

    def test_the_module_never_subscripts_a_label_block(self):
        forbidden = {"labels", "label", "manual_segments", "corrections",
                     "reference_labels", "pseudo_labels"}
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str) \
                    and node.slice.value in forbidden:
                offenders.append((node.lineno, node.slice.value))
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append((node.lineno, node.attr))
        self.assertEqual(offenders, [])

    def test_the_runners_foreground_path_reads_no_label_block(self):
        tree = ast.parse(TOOL.read_text(encoding="utf-8"))
        function = next((node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "run_clip"),
                        None)
        self.assertIsNotNone(function, "run_clip is missing from the runner")
        offenders = []
        for node in ast.walk(function):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and node.slice.value in {"labels", "manual_segments",
                                             "corrections"}:
                offenders.append((node.lineno, node.slice.value))
        self.assertEqual(offenders, [])

    def test_the_candidate_api_takes_a_seed_structure_and_target_only(self):
        import inspect

        allowed = {"frames", "cap", "seed_frame", "seed_x", "seed_y",
                   "interval_frames", "cut_frames", "target",
                   "start_frame", "end_frame"}
        for function in (fg.track_sequence, fg.track_foreground):
            parameters = set(inspect.signature(function).parameters)
            self.assertTrue(parameters <= allowed, parameters - allowed)


# --------------------------------------------------------------------------
# the measurement layer on synthetic scenes

@REQUIRES_CV2
class SyntheticSceneTests(unittest.TestCase):
    def test_the_masked_search_follows_the_small_moving_foreground(self):
        leg = (120, 40, 50, 60)
        frames = head_frames(12, leg=leg)
        run = track(frames)
        self.assertTrue(run.available, run.reason)
        observed = [d for d in run.decisions if d.observed and d.segment is not None]
        self.assertEqual(len(observed), 12)
        for index, decision in enumerate(observed):
            truth = (60 + 3 * index, 60 + 2 * index)
            self.assertLess(abs(decision.x - truth[0]), 2.0)
            self.assertLess(abs(decision.y - truth[1]), 2.0)
        # the search really had the larger static structure in view
        last = [entry for entry in run.raw_frames if entry["frame"] == 11][0]
        window = last["request"]["anchor"]
        radius = last["request"]["radius_px"]
        self.assertGreater(leg[0], window[0] + radius)          # leg is to the right
        self.assertLess(leg[0], window[0] + radius + fg.TEMPLATE_SIZE)
        self.assertEqual(last["decision"]["method"], fg.METHOD_FOREGROUND)
        self.assertEqual(run.summary()["reacquired_frames"], 0)
        self.assertEqual(null_coordinates(run.decisions), [])

    def test_the_seed_mask_covers_the_target_evidence_and_not_the_background(self):
        image = head_scene([(60, 60)])
        grey = fg._grey(image)
        patch = fg._patch(grey, 60.0, 60.0)
        mask, stats = fg.seed_foreground_mask(patch)
        self.assertIsNotNone(mask, stats)
        self.assertEqual(stats["reason"], "")
        self.assertLess(stats["mask_fraction"], fg.MASK_FOREGROUND_MAX_FRACTION)
        self.assertGreaterEqual(stats["mask_pixels"], fg.MASK_MIN_PIXELS)
        inside = patch[mask > 0]
        self.assertGreater(float(np.median(inside)),
                           stats["background_grey"] + fg.MASK_BACKGROUND_DELTA)
        centroid = np.argwhere(mask > 0).mean(axis=0)
        half = (fg.TEMPLATE_SIZE - 1) / 2.0
        self.assertLess(abs(centroid[0] - half), 3.0)
        self.assertLess(abs(centroid[1] - half), 3.0)

    def test_the_map_and_the_direct_score_agree_at_every_position(self):
        image = head_scene([(60, 60)])
        grey = fg._grey(image)
        patch = fg._patch(grey, 60.0, 60.0)
        mask, _ = fg.seed_foreground_mask(patch)
        window = grey[45:45 + 79, 21:21 + 79]
        score_map = fg._masked_ncc_map(window, patch, mask)
        for row, column in ((0, 0), (15, 24), (24, 24), (30, 40)):
            direct = fg._masked_score(window[row:row + fg.TEMPLATE_SIZE,
                                            column:column + fg.TEMPLATE_SIZE],
                                      patch, mask)
            self.assertAlmostEqual(direct, float(score_map[row, column]), places=4)

    def test_the_transformed_mask_stays_binary(self):
        image = head_scene([(60, 60)])
        grey = fg._grey(image)
        patch = fg._patch(grey, 60.0, 60.0)
        mask, _stats = fg.seed_foreground_mask(patch)
        self.assertEqual(sorted(set(np.unique(mask).tolist())), [0.0, 1.0])
        for angle, scale in fg.TRANSFORMS:
            _template, warped_mask = fg._transform(patch, mask, angle, scale)
            values = sorted(set(np.unique(warped_mask).tolist()))
            self.assertTrue(set(values) <= {0.0, 1.0}, (angle, scale, values))
        template, warped_mask = fg._transform(patch, mask, 0.0, 1.0)
        self.assertIs(template, patch)
        self.assertIs(warped_mask, mask)

    def test_two_indistinguishable_candidates_abstain(self):
        frames = [(index, head_scene([(60, 60), (78, 60)])) for index in range(8)]
        run = track(frames, seed_x=60.0)
        self.assertTrue(run.available)
        self.assertTrue(any(decision.gates.get("unambiguous") is False
                            for decision in run.decisions))
        self.assertIn("unambiguous", " ".join(decision.note
                                              for decision in run.decisions))
        self.assertEqual(null_coordinates(run.decisions), [])
        self.assertEqual([d for d in run.decisions if d.observed], [run.decisions[0]])

    def test_a_flat_seed_is_refused_and_nothing_is_reported(self):
        frames = [(index, head_scene([(60, 60)])) for index in range(6)]
        frames = [(frame, np.full_like(image, TURF)) for frame, image in frames]
        run = track(frames)
        self.assertFalse(run.available)
        self.assertEqual(run.reason, "seed_patch_is_flat_low_texture")
        # the seed anchor is still the operator's own point; every frame after it is
        # missing, and nothing is measured or re-acquired from an uninformative seed
        self.assertEqual(run.decisions[0].status, "manual_anchor")
        self.assertEqual([d.status for d in run.decisions[1:]], ["missing"] * 5)
        self.assertEqual(run.decisions[0].to_raw()["gates"], {})
        self.assertEqual(null_coordinates(run.decisions), [])

    def test_a_seed_mask_of_pure_background_texture_is_refused(self):
        checker = 100.0 + 40.0 * ((np.indices((31, 31)).sum(axis=0)) % 2)
        mask, stats = fg.seed_foreground_mask(checker.astype("float32"))
        self.assertIsNone(mask)
        self.assertEqual(stats["reason"], "seed_mask_covers_too_few_pixels")
        self.assertGreaterEqual(stats["patch_std"], fg.MASK_MIN_CONTRAST)
        self.assertGreater(stats["foreground_threshold"], 40.0)

    def test_an_uninformative_seed_scene_is_refused_with_its_reason(self):
        image = head_scene([(90, 90)])                  # the target is not at the seed
        run = track([(index, image) for index in range(4)])
        self.assertFalse(run.available)
        self.assertTrue(run.reason.startswith(("seed_patch_is_flat", "seed_mask_")),
                        run.reason)
        self.assertEqual(null_coordinates(run.decisions), [])
        self.assertEqual([d for d in run.decisions if d.observed][0].status,
                         "manual_anchor")

    def test_a_seed_patch_at_the_frame_edge_is_refused(self):
        frames = [(index, head_scene([(60, 60)])) for index in range(6)]
        run = track(frames, seed_x=4.0, seed_y=60.0)
        self.assertFalse(run.available)
        self.assertEqual(run.reason, "seed_patch_not_fully_inside_frame")
        self.assertEqual(null_coordinates(run.decisions), [])

    def test_a_search_window_at_the_frame_edge_is_clamped_not_rejected(self):
        frames = [(index, head_scene([(max(6, 20 - 3 * index), 60)]))
                  for index in range(6)]
        run = track(frames, seed_x=20.0, seed_y=60.0)
        self.assertTrue(run.available)
        observed = [d for d in run.decisions if d.observed]
        self.assertGreaterEqual(len(observed), 3)
        clipped = [d for d in run.decisions
                   if d.measurement.get("search_window_clipped") is True]
        self.assertTrue(clipped, "the clamped window was never reported as clipped")
        # Both methods clamp search margins while retaining a whole target patch.
        from ghostcaddie.demo.config import TargetOptions
        from ghostcaddie.demo.worker import _match
        grey = fg._grey(head_scene([(20, 60)]))
        template = fg._patch(grey, 20.0, 60.0)
        point, _score, reason = _match(fg._grey(head_scene([(17, 60)])), (17.0, 60.0),
                                       template, TargetOptions())
        self.assertIsNotNone(point)
        self.assertEqual(reason, "")
        self.assertLess(abs(point[0] - 17.0), 2.0)

    def test_a_genuine_disappearance_is_reacquired_inside_the_window(self):
        frames = []
        for index in range(15):
            if 4 <= index <= 6:
                heads = []
            elif index <= 3:
                heads = [(60 + 3 * index, 60)]
            else:
                heads = [(75, 60)]
            frames.append((index, head_scene(heads)))
        run = track(frames)
        self.assertTrue(run.available)
        # causal confirmation: frame 7 is the first measured confirmation and stays
        # lost; frame 8 completes the run and is the re-acquired point, with its own
        # method, a fresh segment and no back-fill of frame 7.
        reacquired = [d for d in run.decisions if d.reacquired]
        self.assertEqual([d.source_frame for d in reacquired], [8])
        self.assertEqual(reacquired[0].method, fg.METHOD_REACQUIRED)
        self.assertLess(abs(float(reacquired[0].x or 0.0) - 75.0), 2.0)
        losses = [d for d in run.decisions if d.status == "lost"]
        self.assertEqual([d.source_frame for d in losses], [6, 7])
        self.assertEqual(null_coordinates(run.decisions), [])
        segments = {d.segment for d in reacquired}
        tracked_before = {d.segment for d in run.decisions
                          if d.observed and d.source_frame < 4}
        self.assertEqual(len(segments), 1)
        self.assertFalse(segments & tracked_before)
        after = [d for d in run.decisions if d.observed and d.source_frame > max(
            d.source_frame for d in reacquired)]
        self.assertTrue(after)
        self.assertTrue(all(d.method == fg.METHOD_FOREGROUND for d in after))

    def test_a_disappearance_longer_than_the_window_is_never_reacquired(self):
        frames = []
        for index in range(25):
            if 4 <= index <= 20:
                heads = []
            elif index <= 3:
                heads = [(60 + 3 * index, 60)]
            else:
                heads = [(75, 60)]
            frames.append((index, head_scene(heads)))
        run = track(frames)
        self.assertEqual(run.summary()["reacquired_frames"], 0)
        self.assertEqual(null_coordinates(run.decisions), [])
        self.assertTrue(any("window is over" in d.note for d in run.decisions))

    def test_the_run_is_prefix_invariant_across_a_re_acquisition(self):
        """Appending frames must not change a single already-reported frame.

        The candidate is asked for frames [0..12] and then for [0..14] of the same
        synthetic disappearance; the frames the shorter run reported must be identical
        in the longer run (status, method, coordinates, score, segment and note).
        """

        def frames(stop):
            return [(index, head_scene([] if 4 <= index <= 6 else
                                       ([(60 + 3 * index, 60)] if index <= 3
                                        else [(75, 60)])))
                    for index in range(stop + 1)]

        short = track(frames(12))
        long = track(frames(14))
        short_raw = [decision.to_raw() for decision in short.decisions]
        long_raw = [decision.to_raw() for decision in long.decisions[:len(short_raw)]]
        self.assertEqual(short_raw, long_raw)
        self.assertTrue([d for d in short.decisions if d.reacquired])
        self.assertEqual([d.source_frame for d in long.decisions
                          if d.reacquired], [8])

    def test_a_structural_cut_stops_the_run_and_is_never_crossed(self):
        frames = head_frames(12)
        run = track(frames, cuts=(6,))
        after = [d for d in run.decisions if d.source_frame >= 6]
        self.assertTrue(after)
        self.assertTrue(all(d.status == "missing" for d in after))
        self.assertTrue(all(d.x is None for d in after))
        self.assertTrue(run.summary()["policy"]["structural_cut_reset"])
        self.assertEqual(run.summary()["reacquired_frames"], 0)


# --------------------------------------------------------------------------
# the runner, end to end on a synthetic clip

@REQUIRES_CV2
class RunnerTests(unittest.TestCase):
    """The clubhead variant must move the clubhead leg and nothing else."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _synthetic_clip(self):
        height, width, frames_count = 160, 200, 30
        path = self.root / "synthetic.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                                 (width, height))
        ball_truth, head_truth = {}, {}
        for index in range(frames_count):
            image = _turf(height, width)
            ball_x, ball_y = 30.0 + 2.0 * index, 100.0
            head_x, head_y = 60.0 + 3.0 * index, 60.0 + 2.0 * index
            _draw_ball(image, ball_x, ball_y)
            _draw_head(image, int(head_x), int(head_y))
            ball_truth[index] = (ball_x, ball_y)
            head_truth[index] = (float(int(head_x)), float(int(head_y)))
            writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        writer.release()
        cap = cv2.VideoCapture(str(path))
        width_decoded = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height_decoded = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        decoded_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        self.assertEqual((width_decoded, height_decoded), (width, height))
        self.assertEqual(decoded_frames, frames_count)
        return path, ball_truth, head_truth, (width, height, decoded_frames, fps)

    def _dataset(self):
        path, ball_truth, head_truth, media = self._synthetic_clip()
        width, height, frames_count, fps = media
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        labels = [{"frame": frame, "state": "visible", "x": ball_truth[frame][0],
                   "y": ball_truth[frame][1], "tolerance_px": 4.0,
                   "note": "synthetic ball position", "reviewer": "synthetic fixture"}
                  for frame in (8, 12, 16, 20)]
        head_labels = [{"frame": frame, "state": "visible", "x": head_truth[frame][0],
                        "y": head_truth[frame][1], "tolerance_px": 4.0,
                        "note": "synthetic clubhead position",
                        "reviewer": "synthetic fixture"}
                       for frame in (8, 12, 16, 20)]
        clip = {"id": "synthetic_clip", "path": str(path), "sha256": sha,
                "width": width, "height": height, "frame_count": frames_count,
                "fps_rational": f"{int(round(fps))}/1", "stroke_type": "full_swing",
                "split": "development", "tags": ["synthetic"],
                "window": {"start_frame": 0, "end_frame": frames_count - 1},
                "cut_frames": [],
                "targets": {
                    "ball": {"seed": {"frame": 2, "x": ball_truth[2][0],
                                      "y": ball_truth[2][1], "note": "synthetic"},
                             "options": {}, "labels": labels},
                    "clubhead": {"seed": {"frame": 2, "x": head_truth[2][0],
                                          "y": head_truth[2][1], "note": "synthetic"},
                                 "options": {}, "labels": head_labels}}}
        payload = {"schema_version": bench.DATASET_SCHEMA,
                   "provenance": dict(bench.REQUIRED_PROVENANCE), "clips": [clip]}
        dataset_path = self.root / "dataset.json"
        dataset_path.write_text(json.dumps(payload), encoding="utf-8")
        return dataset_path, clip

    def _run(self, variant, *, raw_out=None):
        dataset_path, clip = self._dataset()
        out = self.root / f"{variant}.predictions.json"
        command = [sys.executable, str(TOOL), "--dataset", str(dataset_path),
                   "--out", str(out), "--variant", variant]
        if raw_out is not None:
            command += ["--raw-out", str(raw_out)]
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                   text=True, timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return out, clip, completed

    def test_the_ball_leg_is_byte_identical_to_the_baseline_variant(self):
        baseline, clip, _ = self._run("baseline")
        candidate, _, completed = self._run("clubhead_foreground")
        baseline_payload = json.loads(baseline.read_text())
        candidate_payload = json.loads(candidate.read_text())
        self.assertEqual(baseline_payload["variant"], "baseline")
        self.assertEqual(candidate_payload["variant"], "clubhead_foreground")
        baseline_ball = baseline_payload["clips"][0]["targets"]["ball"]
        candidate_ball = candidate_payload["clips"][0]["targets"]["ball"]
        self.assertEqual(baseline_ball, candidate_ball)
        self.assertTrue(candidate_ball["points"], "the synthetic ball produced no points")
        # ...and the two variants are not the same run on the clubhead leg
        baseline_head = baseline_payload["clips"][0]["targets"]["clubhead"]
        candidate_head = candidate_payload["clips"][0]["targets"]["clubhead"]
        self.assertNotEqual(baseline_head, candidate_head)
        self.assertIn("clubhead_foreground", completed.stdout)

    def test_the_candidate_leg_reports_its_own_method_and_stays_in_contract(self):
        baseline, _, _ = self._run("baseline")
        candidate, clip, _ = self._run("clubhead_foreground")
        payload = json.loads(candidate.read_text())
        held = payload["clips"][0]["targets"]["clubhead"]
        methods = {point["method"] for point in held["points"]}
        self.assertIn(fg.METHOD_FOREGROUND, methods)
        self.assertTrue(methods <= {fg.METHOD_FOREGROUND, fg.METHOD_REACQUIRED,
                                    "operator_placement", ""}, methods)
        # the declared options stay the classic tracker's options: exactly the ones the
        # baseline run declares for the same target, never the candidate's profile.
        reference = json.loads(baseline.read_text())["clips"][0]["targets"]["clubhead"]
        self.assertEqual(held["options"], reference["options"])
        self.assertEqual(held["seed"], reference["seed"])
        dataset, _ = bench.load_dataset(self.root / "dataset.json", verify_hashes=True)
        validated = bench.validate_predictions(payload, dataset)
        self.assertEqual(
            len(validated["clips"][0]["targets"]["clubhead"]["points"]),
            len(held["points"]))

    def test_the_report_declares_the_candidate_profile_and_its_constants_hash(self):
        candidate, _, _ = self._run("clubhead_foreground")
        report = json.loads(candidate.with_suffix(".json.report.json").read_text())
        profile = report["runtime"]["candidate_profile"]
        self.assertEqual(profile["name"], fg.PROFILE_NAME)
        self.assertEqual(profile["constants"], fg.PROFILE)
        self.assertEqual(profile["constants_sha256"],
                         bench.sha256_text(bench.canonical_json(fg.PROFILE)))
        checksum = report["runtime"]["algorithm_code_sha256"]["foreground_tracker.py"]
        self.assertEqual(checksum["sha256"], bench.sha256_file(MODULE))
        self.assertEqual(profile["engine_by_target"],
                         {"ball": "classic", "clubhead": "foreground"})
        self.assertIn("clubhead", report["candidate_raw_summary"]["synthetic_clip"])
        self.assertNotIn("baseline", json.dumps({"candidate_profile": profile}))

    def test_the_baseline_report_carries_no_candidate_profile(self):
        baseline, _, _ = self._run("baseline")
        report = json.loads(baseline.with_suffix(".json.report.json").read_text())
        self.assertNotIn("candidate_profile", report["runtime"])
        self.assertNotIn("foreground_tracker.py",
                         report["runtime"]["algorithm_code_sha256"])

    def test_the_raw_metadata_explains_every_frame_that_carried_no_point(self):
        raw_path = self.root / "raw.json"
        candidate, _, completed = self._run("clubhead_foreground", raw_out=raw_path)
        self.assertTrue(raw_path.exists())
        raw = json.loads(raw_path.read_text())
        self.assertEqual(raw["schema_version"], "fairwayos-target-foreground-raw.v1")
        block = raw["clips"]["synthetic_clip"]["clubhead"]
        self.assertTrue(block["frames"])
        predictions = json.loads(candidate.read_text())
        held = predictions["clips"][0]["targets"]["clubhead"]["points"]
        by_frame = {entry["frame"]: entry for entry in block["frames"]}
        for point in held:
            entry = by_frame[point["source_frame"]]
            self.assertEqual(entry["decision"]["status"], point["status"])
            self.assertEqual(entry["decision"]["method"], point["method"])
        for entry in block["frames"]:
            if entry["decision"]["status"] in ("missing", "lost"):
                self.assertIsNone(entry["decision"]["x"])
                self.assertIn("note", entry["decision"])
        self.assertIn("candidate raw:", completed.stdout)

    def test_validate_only_needs_no_candidate_and_decodes_nothing(self):
        dataset_path, _ = self._dataset()
        out = self.root / "never-written.json"
        completed = subprocess.run(
            [sys.executable, str(TOOL), "--dataset", str(dataset_path),
             "--out", str(out), "--variant", "clubhead_foreground", "--validate-only"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
