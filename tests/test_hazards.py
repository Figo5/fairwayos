import unittest

from ghostcaddie.geometry import CoordinateSystem, Point2D
from ghostcaddie.hazards import classify_landing
from ghostcaddie.models import CourseModel, RegionType

FAIRWAY = [[Point2D(0, -20), Point2D(300, -20), Point2D(300, 20), Point2D(0, 20)]]
GREEN = [[Point2D(280, -15), Point2D(300, -15), Point2D(300, 15), Point2D(280, 15)]]
BUNKER = [[Point2D(150, 5), Point2D(170, 5), Point2D(170, 20), Point2D(150, 20)]]
WATER = [[Point2D(150, -30), Point2D(200, -30), Point2D(200, -10), Point2D(150, -10)]]
OB = [[Point2D(240, 30), Point2D(320, 30), Point2D(320, 60), Point2D(240, 60)]]


def _course(**kwargs):
    defaults = dict(
        name="T", par=4, coordinate_system=CoordinateSystem(), fairway=FAIRWAY,
        green=GREEN, bunkers=BUNKER, water_hazards=WATER, out_of_bounds=OB,
        pin_position=Point2D(290, 0),
    )
    defaults.update(kwargs)
    return CourseModel(**defaults)


class TestClassifyLanding(unittest.TestCase):
    def test_region_returns(self):
        course = _course()
        self.assertEqual(classify_landing(Point2D(100, 0), course), RegionType.FAIRWAY)
        self.assertEqual(classify_landing(Point2D(290, 0), course), RegionType.GREEN)
        self.assertEqual(classify_landing(Point2D(160, 12), course), RegionType.BUNKER)
        self.assertEqual(classify_landing(Point2D(170, -20), course), RegionType.WATER)
        self.assertEqual(classify_landing(Point2D(280, 45), course), RegionType.OUT_OF_BOUNDS)

    def test_none_returns_rough(self):
        course = _course()
        self.assertEqual(classify_landing(Point2D(100, 100), course), RegionType.ROUGH)

    def test_priority_order_for_overlaps(self):
        # Overlap every hazard over the same point and verify priority:
        # OB > water > bunker > green > fairway.
        overlapping = [Point2D(150, 0), Point2D(160, 0)]
        # All five lists share the same small polygon around (150, 0).
        shared = [[Point2D(145, -5), Point2D(160, -5), Point2D(160, 5), Point2D(145, 5)]]
        course = _course(
            fairway=shared, green=shared, bunkers=shared,
            water_hazards=shared, out_of_bounds=shared,
        )
        self.assertEqual(classify_landing(Point2D(150, 0), course), RegionType.OUT_OF_BOUNDS)
        course2 = _course(fairway=shared, green=shared, bunkers=shared, water_hazards=shared)
        self.assertEqual(classify_landing(Point2D(150, 0), course2), RegionType.WATER)
        course3 = _course(fairway=shared, green=shared, bunkers=shared)
        self.assertEqual(classify_landing(Point2D(150, 0), course3), RegionType.BUNKER)
        course4 = _course(fairway=shared, green=shared)
        self.assertEqual(classify_landing(Point2D(150, 0), course4), RegionType.GREEN)


if __name__ == "__main__":
    unittest.main()
