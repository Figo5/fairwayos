"""TDD: display->native coordinate mapping and exact frame binding.

A seed clicked on a scaled preview must land on the right NATIVE pixel, and must
be bound to the exact decoded frame and the source hash. Getting this wrong would
silently produce plausible-looking but wrong coordinates.
"""
import unittest
from ghostcaddie.upload.uimap import (
    DisplayTransform, MappingError, map_display_to_native, map_native_to_display,
)


class TransformTests(unittest.TestCase):
    def test_identity_when_display_matches_native(self):
        t = DisplayTransform(native_w=1920, native_h=1080, display_w=1920, display_h=1080)
        self.assertEqual(map_display_to_native(t, 100, 200), (100.0, 200.0))

    def test_half_scale_doubles_coordinates(self):
        t = DisplayTransform(1920, 1080, 960, 540)
        self.assertEqual(map_display_to_native(t, 100, 200), (200.0, 400.0))

    def test_letterboxed_display_removes_padding(self):
        # 16:9 native shown in a 4:3 box -> vertical padding
        t = DisplayTransform(1920, 1080, 800, 800)
        # content is 800x450 centred -> pad_y = 175
        x, y = map_display_to_native(t, 400, 400)
        self.assertAlmostEqual(x, 960.0, places=3)
        self.assertAlmostEqual(y, 540.0, places=3)

    def test_click_in_padding_is_rejected_not_clamped(self):
        t = DisplayTransform(1920, 1080, 800, 800)
        with self.assertRaises(MappingError):
            map_display_to_native(t, 400, 10)   # inside the letterbox bar

    def test_out_of_bounds_click_is_rejected(self):
        t = DisplayTransform(1920, 1080, 960, 540)
        for bad in ((-1, 10), (10, -1), (961, 10), (10, 541)):
            with self.assertRaises(MappingError):
                map_display_to_native(t, *bad)

    def test_round_trip_is_stable(self):
        t = DisplayTransform(1280, 720, 640, 360)
        nx, ny = map_display_to_native(t, 321, 187)
        dx, dy = map_native_to_display(t, nx, ny)
        self.assertAlmostEqual(dx, 321, places=3)
        self.assertAlmostEqual(dy, 187, places=3)

    def test_transform_rejects_nonpositive_sizes(self):
        for bad in ((0, 1080, 10, 10), (1920, 0, 10, 10), (1920, 1080, 0, 10)):
            with self.assertRaises(MappingError):
                DisplayTransform(*bad).validate()


class SeedBindingTests(unittest.TestCase):
    def test_seed_records_exact_frame_and_source_hash(self):
        from ghostcaddie.upload.uimap import build_seed_payload
        t = DisplayTransform(1920, 1080, 960, 540)
        p = build_seed_payload(source_sha256="a"*64, frame=310, target="ball",
                               transform=t, display_points=[(100, 200)])
        self.assertEqual(p["source_sha256"], "a"*64)
        self.assertEqual(p["frame"], 310)
        self.assertEqual(p["point_xy"], [200.0, 400.0])
        self.assertEqual(p["initialization"], "assisted")
        self.assertFalse(p["ground_truth"])

    def test_box_needs_two_points_in_order(self):
        from ghostcaddie.upload.uimap import build_seed_payload
        t = DisplayTransform(1920, 1080, 960, 540)
        p = build_seed_payload("a"*64, 12, "clubhead", t, [(300, 400), (100, 200)])
        self.assertEqual(p["box_xyxy"], [200.0, 400.0, 600.0, 800.0])

    def test_body_seed_is_refused_because_body_is_automatic(self):
        from ghostcaddie.upload.uimap import build_seed_payload
        t = DisplayTransform(1920, 1080, 960, 540)
        with self.assertRaises(MappingError):
            build_seed_payload("a"*64, 1, "body", t, [(10, 10)])


if __name__ == "__main__":
    unittest.main()
