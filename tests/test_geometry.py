import unittest

from ghostcaddie.geometry import (
    CoordinateMapper,
    CoordinateSystem,
    Point2D,
    bearing_deg,
    distance,
    point_in_polygon,
)

SQUARE = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]


class TestPointInPolygon(unittest.TestCase):
    def test_inside(self):
        self.assertTrue(point_in_polygon(Point2D(5, 5), SQUARE))

    def test_outside(self):
        self.assertFalse(point_in_polygon(Point2D(15, 5), SQUARE))
        self.assertFalse(point_in_polygon(Point2D(-1, 5), SQUARE))
        self.assertFalse(point_in_polygon(Point2D(5, 15), SQUARE))

    def test_on_boundary_adjacent(self):
        # A point slightly inside just off an edge counts as inside; just
        # outside counts as outside. (Exact on-edge is ambiguous by design.)
        self.assertTrue(point_in_polygon(Point2D(0.1, 5), SQUARE))
        self.assertFalse(point_in_polygon(Point2D(-0.1, 5), SQUARE))


class TestDistanceAndBearing(unittest.TestCase):
    def test_distance_3_4_5(self):
        self.assertAlmostEqual(distance(Point2D(0, 0), Point2D(3, 4)), 5.0)

    def test_bearing_cardinals(self):
        self.assertAlmostEqual(bearing_deg(Point2D(0, 0), Point2D(10, 0)), 0.0)
        self.assertAlmostEqual(bearing_deg(Point2D(0, 0), Point2D(0, 10)), 90.0)
        self.assertAlmostEqual(bearing_deg(Point2D(0, 0), Point2D(-10, 0)), 180.0)
        self.assertAlmostEqual(bearing_deg(Point2D(0, 0), Point2D(0, -10)), -90.0)


class TestCoordinateMapper(unittest.TestCase):
    def test_round_trip_nonzero_origin(self):
        cs = CoordinateSystem(origin=Point2D(250.0, 100.0))
        mapper = CoordinateMapper(cs)
        raw = {"x": 320.5, "y": 40.25}
        engine = mapper.to_engine(raw)
        self.assertAlmostEqual(engine.x, 70.5)
        self.assertAlmostEqual(engine.y, -59.75)
        back = mapper.from_engine(engine)
        self.assertAlmostEqual(back["x"], raw["x"])
        self.assertAlmostEqual(back["y"], raw["y"])
        # And engine -> raw -> engine.
        point = Point2D(12.5, -3.25)
        self.assertAlmostEqual(mapper.to_engine(mapper.from_engine(point)).x, point.x)
        self.assertAlmostEqual(mapper.to_engine(mapper.from_engine(point)).y, point.y)


# --- Four-point calibration ---

# Known projective mapping used to generate correspondences:
#   X = (2u + 0.5v + 10) / (0.001u + 0.002v + 1)
#   Y = (-0.25u + 3v + 20) / (0.001u + 0.002v + 1)
def _projective(u, v):
    denom = 0.001 * u + 0.002 * v + 1.0
    return (2.0 * u + 0.5 * v + 10.0) / denom, (-0.25 * u + 3.0 * v + 20.0) / denom


SOURCE_CORNERS = [
    Point2D(100.0, 80.0),
    Point2D(900.0, 80.0),
    Point2D(900.0, 620.0),
    Point2D(100.0, 620.0),
]
ENGINE_CORNERS = [Point2D(*_projective(p.x, p.y)) for p in SOURCE_CORNERS]


def _four_point_cs():
    return CoordinateSystem(
        mode="four_point",
        units="yards",
        source_units="pixels",
        source_points=tuple(SOURCE_CORNERS),
        engine_points=tuple(ENGINE_CORNERS),
    )


class TestFourPointSchema(unittest.TestCase):
    def test_stores_fields_without_manual_semantics(self):
        cs = _four_point_cs()
        self.assertEqual(cs.mode, "four_point")
        self.assertEqual(cs.units, "yards")
        self.assertEqual(cs.source_units, "pixels")
        self.assertEqual(cs.source_points, tuple(SOURCE_CORNERS))
        self.assertEqual(cs.engine_points, tuple(ENGINE_CORNERS))

    def test_manual_defaults_unchanged(self):
        cs = CoordinateSystem()
        self.assertEqual(cs.mode, "manual")
        self.assertEqual(cs.origin, Point2D(0.0, 0.0))
        self.assertEqual(cs.units, "yards")
        self.assertEqual(cs.source_points, ())
        self.assertEqual(cs.engine_points, ())

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            CoordinateSystem(mode="bogus")

    def test_four_point_requires_exactly_four_pairs(self):
        with self.assertRaises(ValueError):
            CoordinateSystem(
                mode="four_point",
                source_points=tuple(SOURCE_CORNERS[:3]),
                engine_points=tuple(ENGINE_CORNERS[:3]),
            )
        with self.assertRaises(ValueError):
            CoordinateSystem(
                mode="four_point",
                source_points=tuple(SOURCE_CORNERS),
                engine_points=tuple(ENGINE_CORNERS[:3]),
            )


class TestFourPointMapping(unittest.TestCase):
    def test_corners_map_within_tolerance(self):
        mapper = CoordinateMapper(_four_point_cs())
        for src, eng in zip(SOURCE_CORNERS, ENGINE_CORNERS):
            got = mapper.to_engine({"x": src.x, "y": src.y})
            self.assertAlmostEqual(got.x, eng.x, places=8)
            self.assertAlmostEqual(got.y, eng.y, places=8)

    def test_interior_points_map_within_tolerance(self):
        mapper = CoordinateMapper(_four_point_cs())
        for u, v in [(300.0, 200.0), (700.0, 500.0)]:
            ex, ey = _projective(u, v)
            got = mapper.to_engine({"x": u, "y": v})
            self.assertAlmostEqual(got.x, ex, places=8)
            self.assertAlmostEqual(got.y, ey, places=8)

    def test_round_trips_within_tolerance(self):
        mapper = CoordinateMapper(_four_point_cs())
        for u, v in [(100.0, 80.0), (500.0, 350.0), (900.0, 620.0)]:
            back = mapper.from_engine(mapper.to_engine({"x": u, "y": v}))
            self.assertAlmostEqual(back["x"], u, places=8)
            self.assertAlmostEqual(back["y"], v, places=8)
        for eng in ENGINE_CORNERS:
            got = mapper.to_engine(mapper.from_engine(eng))
            self.assertAlmostEqual(got.x, eng.x, places=8)
            self.assertAlmostEqual(got.y, eng.y, places=8)


class TestFourPointValidation(unittest.TestCase):
    def _assert_rejected(self, source, engine):
        with self.assertRaises(ValueError):
            CoordinateMapper(
                CoordinateSystem(
                    mode="four_point",
                    source_points=tuple(source),
                    engine_points=tuple(engine),
                )
            )

    def test_duplicate_source_points_rejected(self):
        self._assert_rejected(
            [Point2D(100, 80), Point2D(100, 80), Point2D(900, 620), Point2D(100, 620)],
            ENGINE_CORNERS,
        )

    def test_duplicate_engine_points_rejected(self):
        self._assert_rejected(
            SOURCE_CORNERS,
            [ENGINE_CORNERS[0], ENGINE_CORNERS[0], ENGINE_CORNERS[2], ENGINE_CORNERS[3]],
        )

    def test_near_collinear_source_rejected(self):
        # Three points nearly on one line (tiny y deviation) -> near-singular.
        self._assert_rejected(
            [
                Point2D(100, 80),
                Point2D(500, 80.0001),
                Point2D(900, 80),
                Point2D(100, 620),
            ],
            ENGINE_CORNERS,
        )

    def test_near_collinear_engine_rejected(self):
        # Three engine points nearly on one line (tiny y deviation).
        self._assert_rejected(
            SOURCE_CORNERS,
            [
                Point2D(0, 0),
                Point2D(300, 0),
                Point2D(300, 0.0001),
                Point2D(0, 200),
            ],
        )

    def test_valid_quadrilateral_accepted(self):
        mapper = CoordinateMapper(_four_point_cs())
        self.assertIsNotNone(mapper)

    def test_mixed_scale_valid_calibration_accepted(self):
        # Source in pixels (0..4000), engine in yards (5x3): the two point
        # sets differ by ~3 orders of magnitude. A single combined scale would
        # falsely reject the small engine quadrilateral as degenerate.
        source = [
            Point2D(0, 0),
            Point2D(4000, 0),
            Point2D(4000, 3000),
            Point2D(0, 3000),
        ]
        engine = [
            Point2D(0, 0),
            Point2D(5, 0),
            Point2D(5, 3),
            Point2D(0, 3),
        ]
        mapper = CoordinateMapper(
            CoordinateSystem(
                mode="four_point",
                source_points=tuple(source),
                engine_points=tuple(engine),
            )
        )
        # The affine mapping is exact: u/800 -> X, v/1000 -> Y.
        got = mapper.to_engine({"x": 1000.0, "y": 500.0})
        self.assertAlmostEqual(got.x, 1.25)
        self.assertAlmostEqual(got.y, 0.5)
        back = mapper.from_engine(got)
        self.assertAlmostEqual(back["x"], 1000.0)
        self.assertAlmostEqual(back["y"], 500.0)


class TestHomographyFiniteBoundary(unittest.TestCase):
    def _mapper(self):
        return CoordinateMapper(_four_point_cs())

    def test_non_finite_forward_input_rejected(self):
        mapper = self._mapper()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                mapper.to_engine({"x": bad, "y": 200.0})
            with self.assertRaises(ValueError):
                mapper.to_engine({"x": 200.0, "y": bad})

    def test_non_finite_reverse_input_rejected(self):
        mapper = self._mapper()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                mapper.from_engine(Point2D(bad, 0.0))
            with self.assertRaises(ValueError):
                mapper.from_engine(Point2D(0.0, bad))

    def test_non_finite_calibration_input_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            for i in range(4):
                source = list(SOURCE_CORNERS)
                source[i] = Point2D(bad, source[i].y)
                with self.assertRaises(ValueError):
                    CoordinateMapper(
                        CoordinateSystem(
                            mode="four_point",
                            source_points=tuple(source),
                            engine_points=tuple(ENGINE_CORNERS),
                        )
                    )
                engine = list(ENGINE_CORNERS)
                engine[i] = Point2D(engine[i].x, bad)
                with self.assertRaises(ValueError):
                    CoordinateMapper(
                        CoordinateSystem(
                            mode="four_point",
                            source_points=tuple(SOURCE_CORNERS),
                            engine_points=tuple(engine),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
