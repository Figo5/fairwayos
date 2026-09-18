import unittest

from ghostcaddie.video.ball_evaluation import aggregate, evaluate_clip, match_tolerance


def annotation(frames, width=1000, height=1000):
    return {"clip_id": "clip", "width": width, "height": height, "frames": frames}


def visible(x, y, radius=None, ambiguous=False):
    row = {"visible": True, "x": x, "y": y, "ambiguous": ambiguous}
    if radius is not None:
        row["radius"] = radius
    return row


def absent(ambiguous=False):
    return {"visible": False, "reason": "gone", "ambiguous": ambiguous}


def observed(frame, x, y, state="observed"):
    return {"frame_index": frame, "state": state, "x": x, "y": y}


def silent(frame, state="unavailable"):
    return {"frame_index": frame, "state": state, "x": None, "y": None}


class ToleranceTests(unittest.TestCase):
    def test_tolerance_scales_with_the_frame_and_with_the_ball(self):
        self.assertAlmostEqual(match_tolerance(width=1920, height=1080, radius=None), 21.6)
        self.assertAlmostEqual(match_tolerance(width=4096, height=2160, radius=None), 43.2)
        # A big ball widens the tolerance; a small one never narrows it below
        # the resolution floor.
        self.assertAlmostEqual(match_tolerance(width=1920, height=1080, radius=100), 150.0)
        self.assertAlmostEqual(match_tolerance(width=1920, height=1080, radius=2), 21.6)


class FrameScoringTests(unittest.TestCase):
    def test_saying_nothing_when_the_ball_is_absent_is_a_success(self):
        result = evaluate_clip(annotation({"1": absent(), "2": absent()}),
                               [silent(1), silent(2)])
        self.assertEqual(result["true_negative"], 2)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["fp_rate_on_absent"], 0.0)

    def test_a_prediction_on_an_absent_frame_is_a_false_positive(self):
        result = evaluate_clip(annotation({"1": absent()}), [observed(1, 10, 10)])
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["fp_rate_on_absent"], 1.0)
        self.assertEqual(result["wrong_object_rate_on_visible"], 0.0)

    def test_a_wrong_object_lock_is_a_false_positive_not_a_near_miss(self):
        # The truth is at (10, 10) and the prediction is 400 px away on some
        # other bright thing. Scoring that as "a miss" would let a tracker that
        # follows the flagstick look like a tracker with poor precision.
        result = evaluate_clip(annotation({"1": visible(10.0, 10.0, 5.0)}),
                               [observed(1, 410, 10)])
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["wrong_object_rate_on_visible"], 1.0)
        self.assertEqual(result["recall"], 0.0)

    def test_a_hit_inside_tolerance_counts_and_records_its_error(self):
        result = evaluate_clip(annotation({"1": visible(100.0, 100.0, 5.0)}),
                               [observed(1, 110, 100)])
        self.assertEqual(result["true_positive"], 1)
        self.assertAlmostEqual(result["median_error_px"], 10.0)
        self.assertEqual(result["precision"], 1.0)

    def test_silence_on_a_visible_frame_is_a_false_negative(self):
        result = evaluate_clip(annotation({"1": visible(10.0, 10.0)}), [silent(1)])
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["recall"], 0.0)

    def test_ambiguous_frames_are_excluded_from_scoring_and_counted_separately(self):
        result = evaluate_clip(
            annotation({"1": visible(10.0, 10.0, ambiguous=True), "2": absent(ambiguous=True)}),
            [observed(1, 900, 900), observed(2, 5, 5)])
        self.assertEqual(result["ambiguous_frames"], 2)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["true_positive"], 0)

    def test_only_observed_states_may_earn_credit(self):
        # An interpolated point is not an observation. Letting it score would be
        # precisely the stale-coordinate credit this evaluation exists to refuse.
        for state in ("interpolated", "ambiguous", "occluded", "lost", "unavailable"):
            result = evaluate_clip(annotation({"1": visible(100.0, 100.0)}),
                                   [{"frame_index": 1, "state": state, "x": 100.0, "y": 100.0}])
            self.assertEqual(result["true_positive"], 0, state)
            self.assertEqual(result["false_negative"], 1, state)


class TrackScoringTests(unittest.TestCase):
    def test_a_long_run_pinned_to_one_wrong_spot_is_reported_as_a_static_lock(self):
        frames = {str(index): absent() for index in range(1, 7)}
        rows = [observed(index, 500.0, 500.0) for index in range(1, 7)]
        result = evaluate_clip(annotation(frames), rows)
        self.assertEqual(result["longest_static_lock"], 6)

    def test_scattered_errors_do_not_look_like_a_lock(self):
        frames = {str(index): absent() for index in range(1, 7)}
        rows = [observed(index, 100.0 * index, 500.0) for index in range(1, 7)]
        result = evaluate_clip(annotation(frames), rows)
        self.assertEqual(result["longest_static_lock"], 1)

    def test_longest_correct_run_counts_correct_silence_as_well_as_correct_hits(self):
        frames = {"1": visible(10.0, 10.0), "2": absent(), "3": absent(),
                  "4": visible(20.0, 20.0)}
        rows = [observed(1, 10, 10), silent(2), silent(3), observed(4, 900, 900)]
        result = evaluate_clip(annotation(frames), rows)
        self.assertEqual(result["longest_correct_run"], 3)

    def test_identity_switches_count_transitions_between_right_and_wrong(self):
        frames = {"1": visible(10.0, 10.0), "2": visible(10.0, 10.0), "3": visible(10.0, 10.0)}
        rows = [observed(1, 10, 10), observed(2, 900, 900), observed(3, 10, 10)]
        result = evaluate_clip(annotation(frames), rows)
        self.assertEqual(result["identity_switches"], 2)


class AggregateTests(unittest.TestCase):
    def test_pooling_keeps_the_two_false_positive_families_apart(self):
        strong = evaluate_clip(annotation({"1": visible(10.0, 10.0)}), [observed(1, 10, 10)])
        weak = evaluate_clip(annotation({"1": absent(), "2": visible(10.0, 10.0)}),
                             [observed(1, 500, 500), observed(2, 800, 800)])
        totals = aggregate([strong, weak])
        self.assertEqual(totals["clips"], 2)
        self.assertEqual(totals["true_positive"], 1)
        self.assertEqual(totals["false_positive"], 2)
        self.assertEqual(totals["wrong_object_frames"], 1)
        self.assertEqual(totals["fp_frames_on_absent"], 1)
        self.assertAlmostEqual(totals["fp_rate_on_absent"], 1.0)


if __name__ == "__main__":
    unittest.main()
