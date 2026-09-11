import unittest

from ghostcaddie.video.siwoo_layers import (
    LayerState,
    choose_golfer_candidate,
    object_visibility_for_frame,
    renderable_trail,
    reset_layers_on_cut,
    source_to_display_frame,
)


class SiWooLayerContractTests(unittest.TestCase):
    def test_unavailable_state_hides_marker_and_clears_trail(self):
        history = [(10, 10), (12, 11)]
        state = LayerState(name="ball", visible=False, point=None,
                           reason="unresolved_against_cloud")
        self.assertEqual(renderable_trail(state, history), [])

    def test_cut_reset_clears_each_layer_independently(self):
        layers = {
            "body": LayerState("body", True, (100, 100), reason="pose"),
            "clubhead": LayerState("clubhead", False, None, reason="impact_unresolvable"),
            "ball": LayerState("ball", True, (200, 120), reason="observed"),
        }
        reset = reset_layers_on_cut(layers, cut_between=(215, 216))
        self.assertFalse(reset["body"].visible)
        self.assertFalse(reset["clubhead"].visible)
        self.assertFalse(reset["ball"].visible)
        self.assertIn("cut", reset["body"].reason)
        self.assertIn("impact_unresolvable", layers["clubhead"].reason)

    def test_golfer_selection_prefers_main_player_not_caddie_bib(self):
        candidates = [
            {"box": (840, 380, 925, 650), "confidence": 0.88, "white_bib_fraction": 0.55},
            {"box": (1010, 345, 1110, 665), "confidence": 0.72, "white_bib_fraction": 0.08},
        ]
        picked = choose_golfer_candidate(candidates, preferred_x=1060)
        self.assertEqual(picked["box"], (1010, 345, 1110, 665))

    def test_source_frame_mapping_is_one_to_one_for_realtime_pass(self):
        self.assertEqual(source_to_display_frame(40, pass_start=40), 0)
        self.assertEqual(source_to_display_frame(149, pass_start=40), 109)
        self.assertEqual(source_to_display_frame(339, pass_start=40), 299)

    def test_visibility_map_keeps_body_after_prior_cutoff_until_reviewed_exit(self):
        visibility = {
            "body": [(40, 118, "native_review_visible_setup_to_partial_followthrough")],
            "ball": [(149, 208, "observed_flight_track")],
        }
        self.assertEqual(object_visibility_for_frame(98, "body", visibility)[0], "visible")
        self.assertEqual(object_visibility_for_frame(115, "body", visibility)[0], "visible")
        self.assertEqual(object_visibility_for_frame(119, "body", visibility)[0], "unavailable")

    def test_visibility_map_resets_after_cut_even_inside_interval(self):
        visibility = {"ball": [(149, 230, "candidate_visible")], "cuts": [216]}
        self.assertEqual(object_visibility_for_frame(208, "ball", visibility)[0], "visible")
        state, reason = object_visibility_for_frame(216, "ball", visibility)
        self.assertEqual(state, "unavailable")
        self.assertIn("camera_cut", reason)

    def test_shift_box_clamps_to_frame_and_preserves_size(self):
        from ghostcaddie.video.siwoo_layers import shift_box
        self.assertEqual(shift_box((10, 20, 110, 220), 5, -10, 1280, 720),
                         (15, 10, 115, 210))
        self.assertEqual(shift_box((1200, 600, 1280, 720), 50, 50, 1280, 720),
                         (1200, 600, 1280, 720))

    def test_body_continuation_budget_stops_unbounded_flow_ghosting(self):
        from ghostcaddie.video.siwoo_layers import can_continue_without_detection
        self.assertTrue(can_continue_without_detection(0, max_misses=5))
        self.assertTrue(can_continue_without_detection(4, max_misses=5))
        self.assertFalse(can_continue_without_detection(5, max_misses=5))

    def test_golfer_candidate_gate_rejects_caddie_and_scene_wide_boxes(self):
        from ghostcaddie.video.siwoo_layers import is_golfer_body_candidate
        self.assertTrue(is_golfer_body_candidate((606, 256, 699, 613), frame_width=1280, frame_height=720))
        self.assertFalse(is_golfer_body_candidate((840, 380, 925, 650), frame_width=1280, frame_height=720))
        self.assertFalse(is_golfer_body_candidate((83, 12, 1280, 712), frame_width=1280, frame_height=720))

    def test_track_ranges_preserve_separate_segments_without_bridging(self):
        from ghostcaddie.video.siwoo_layers import contiguous_visible_ranges
        states = {
            98: {"visible": True},
            99: {"visible": True},
            100: {"visible": False},
            149: {"visible": True},
            150: {"visible": True},
            209: {"visible": True},
        }
        self.assertEqual(contiguous_visible_ranges(states), [(98, 99), (149, 150), (209, 209)])


if __name__ == "__main__":
    unittest.main()
