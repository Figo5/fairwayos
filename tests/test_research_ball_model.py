import unittest

from ghostcaddie.video.research_ball_model import normalize_box, normalize_point


class TestResearchBallModel(unittest.TestCase):
    def test_scales_normalized_coordinates_to_pixels(self):
        self.assertEqual(normalize_point((0.25, 0.5), 600, 480), (150.0, 240.0))
        self.assertEqual(normalize_box((0.1, 0.2, 0.4, 0.6), 600, 480), (60.0, 96.0, 240.0, 288.0))

    def test_preserves_pixel_coordinates(self):
        self.assertEqual(normalize_point((150.0, 240.0), 600, 480), (150.0, 240.0))
        self.assertEqual(normalize_box((60.0, 96.0, 240.0, 288.0), 600, 480), (60.0, 96.0, 240.0, 288.0))

    def test_rejects_invalid_dimensions_or_shape(self):
        with self.assertRaises(ValueError):
            normalize_point((0.2, 0.3), 0, 480)
        with self.assertRaises(ValueError):
            normalize_box((0.1, 0.2, 0.3), 600, 480)


if __name__ == "__main__":
    unittest.main()
