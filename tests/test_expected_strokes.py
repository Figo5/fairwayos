import unittest

from ghostcaddie.config import ExpectedStrokesConfig
from ghostcaddie.expected_strokes import BaselineTourExpectedStrokesModel
from ghostcaddie.models import RegionType


class TestExpectedStrokes(unittest.TestCase):
    def setUp(self):
        self.model = BaselineTourExpectedStrokesModel(ExpectedStrokesConfig())

    def test_flat_band_lookup(self):
        # Band edges from the default table.
        self.assertEqual(self.model.strokes_from_lie(RegionType.FAIRWAY, 50), 2.5)
        self.assertEqual(self.model.strokes_from_lie(RegionType.FAIRWAY, 75), 2.7)
        self.assertEqual(self.model.strokes_from_lie(RegionType.FAIRWAY, 150), 2.9)
        self.assertEqual(self.model.strokes_from_lie(RegionType.FAIRWAY, 999), 3.6)  # past all bands
        self.assertEqual(self.model.strokes_from_lie(RegionType.ROUGH, 75), 2.9)
        self.assertEqual(self.model.strokes_from_lie(RegionType.BUNKER, 25), 2.8)

    def test_water_penalty_composition(self):
        cfg = ExpectedStrokesConfig(water_penalty_strokes=1.0)
        model = BaselineTourExpectedStrokesModel(cfg)
        dist = 90.0
        expected = cfg.water_penalty_strokes + model.strokes_from_lie(RegionType.ROUGH, dist)
        self.assertEqual(
            model.expected_strokes_for_landing(RegionType.WATER, dist, 200.0), expected
        )

    def test_ob_penalty_uses_original_distance(self):
        cfg = ExpectedStrokesConfig(ob_penalty_strokes=2.0)
        model = BaselineTourExpectedStrokesModel(cfg)
        landing_dist, original_dist = 5.0, 160.0
        expected = cfg.ob_penalty_strokes + model.strokes_from_lie(RegionType.FAIRWAY, original_dist)
        self.assertEqual(
            model.expected_strokes_for_landing(RegionType.OUT_OF_BOUNDS, landing_dist, original_dist),
            expected,
        )
        # And it must NOT use the (tiny) landing distance.
        self.assertNotEqual(
            model.expected_strokes_for_landing(RegionType.OUT_OF_BOUNDS, landing_dist, original_dist),
            cfg.ob_penalty_strokes + model.strokes_from_lie(RegionType.FAIRWAY, landing_dist),
        )

    def test_rough_worse_than_fairway(self):
        for d in (40, 100, 180, 300):
            self.assertGreater(
                self.model.strokes_from_lie(RegionType.ROUGH, d),
                self.model.strokes_from_lie(RegionType.FAIRWAY, d),
            )


if __name__ == "__main__":
    unittest.main()
