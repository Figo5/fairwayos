import math
import random
import statistics
import unittest

from ghostcaddie.config import SimulationConfig
from ghostcaddie.dispersion import GaussianDispersionModel
from ghostcaddie.geometry import Point2D
from ghostcaddie.models import (
    ClubProfile,
    LiePerformanceModifier,
    ShotEvent,
)

MODEL = GaussianDispersionModel()
CLUB = ClubProfile(
    club="7i", carry_mean_yd=165.0, carry_stddev_yd=7.0,
    lateral_stddev_yd=9.0, miss_bias_yd=3.0, sample_size=100,
)
START = Point2D(0, 0)
AIM = Point2D(160, 80)


class TestWindContract(unittest.TestCase):
    """ShotEvent.wind trust-boundary: documented toward-direction schema."""

    def _shot(self, wind):
        return ShotEvent(
            event_id="E", player_id="P", tournament_id="T",
            hole_number=1, shot_number=1,
            start_position=START, target_position=AIM,
            actual_landing_position=AIM, lie="tee", club="iron",
            distance_to_pin=160.0, wind=wind, timestamp="2026-06-14T00:00:00Z",
        )

    def test_zero_wind_accepted(self):
        shot = self._shot({"speed_mph": 0.0, "direction_deg": 90})
        self.assertEqual(shot.wind["speed_mph"], 0.0)
        self.assertEqual(shot.wind["direction_deg"], 90)

    def test_negative_speed_rejected(self):
        with self.assertRaises(ValueError):
            self._shot({"speed_mph": -1.0, "direction_deg": 90})

    def test_missing_keys_rejected(self):
        with self.assertRaises(ValueError):
            self._shot({"speed_mph": 5.0})
        with self.assertRaises(ValueError):
            self._shot({"direction_deg": 90})

    def test_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            self._shot({"speed_mph": "fast", "direction_deg": 90})

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            self._shot({"speed_mph": float("nan"), "direction_deg": 90})
        with self.assertRaises(ValueError):
            self._shot({"speed_mph": 5.0, "direction_deg": float("inf")})

    def test_equivalent_directions_accepted_unbounded(self):
        # -90 and 270 are the same physical wind; direction is intentionally
        # unbounded so both stay usable by sin/cos.
        self._shot({"speed_mph": 5.0, "direction_deg": -90})
        self._shot({"speed_mph": 5.0, "direction_deg": 450})


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_sequence(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        seq1 = MODEL.sample_many(START, AIM, CLUB, None, rng1, 200)
        seq2 = MODEL.sample_many(START, AIM, CLUB, None, rng2, 200)
        self.assertEqual(seq1, seq2)
        # And a different seed diverges.
        rng3 = random.Random(99)
        seq3 = MODEL.sample_many(START, AIM, CLUB, None, rng3, 200)
        self.assertNotEqual(seq1, seq3)


class TestLieEffect(unittest.TestCase):
    def test_rough_lie_widens_spread(self):
        fairway_mod = LiePerformanceModifier(1.0, 1.0)
        rough_mod = LiePerformanceModifier(0.95, 1.3)
        n = 1000
        fairway_spread = self._lateral_spread(fairway_mod, n)
        rough_spread = self._lateral_spread(rough_mod, n)
        self.assertGreater(rough_spread, fairway_spread * 1.1)

    def _lateral_spread(self, lie_mod, n):
        rng = random.Random(7)
        samples = MODEL.sample_many(START, AIM, CLUB, lie_mod, rng, n)
        return statistics.pstdev(s.y for s in samples)


class TestCarryMean(unittest.TestCase):
    def test_mean_carry_close_to_club(self):
        # Aim straight along +x so the sampled x IS the along-axis carry.
        aim_x = Point2D(160, 0)
        rng = random.Random(11)
        n = 1000
        samples = MODEL.sample_many(START, aim_x, CLUB, None, rng, n)
        mean_along = statistics.mean(s.x for s in samples)
        self.assertAlmostEqual(mean_along, CLUB.carry_mean_yd, delta=1.5)


class TestWindShifts(unittest.TestCase):
    """Cardinal wind vector shifts on a straight +x aim.

    Wind contract: 0 deg = tailwind, 180 = headwind, 90 = crosswind toward
    +y, 270 = crosswind toward -y, for a shot aimed along +x. The strike-frame
    projection must raise/lower the along or lateral Gaussian mean by
    speed * coefficient, leaving the other axis unchanged.
    """

    AIM_X = Point2D(165, 0)  # straight +x aim so x IS along, y IS lateral
    SPEED = 10.0
    ALONG_COEF = 1.5  # must match SimulationConfig default
    LATERAL_COEF = 1.0

    def _mean_shift(self, wind, n=2000):
        rng = random.Random(7)
        base = MODEL.sample_many(START, self.AIM_X, CLUB, None, rng, n)
        mean_x0 = statistics.mean(p.x for p in base)
        mean_y0 = statistics.mean(p.y for p in base)
        rng = random.Random(7)  # identical draw stream, only the mean shifts
        windy = MODEL.sample_many(START, self.AIM_X, CLUB, None, rng, n,
                                  wind=wind)
        return (statistics.mean(p.x for p in windy) - mean_x0,
                statistics.mean(p.y for p in windy) - mean_y0)

    def test_tailwind_0_raises_x_leaves_y(self):
        dx, dy = self._mean_shift({"speed_mph": self.SPEED, "direction_deg": 0.0})
        self.assertAlmostEqual(dx, self.SPEED * self.ALONG_COEF, delta=0.6)
        self.assertAlmostEqual(dy, 0.0, delta=0.6)

    def test_headwind_180_lowers_x_leaves_y(self):
        dx, dy = self._mean_shift({"speed_mph": self.SPEED, "direction_deg": 180.0})
        self.assertAlmostEqual(dx, -self.SPEED * self.ALONG_COEF, delta=0.6)
        self.assertAlmostEqual(dy, 0.0, delta=0.6)

    def test_crosswind_90_raises_y_leaves_x(self):
        dx, dy = self._mean_shift({"speed_mph": self.SPEED, "direction_deg": 90.0})
        self.assertAlmostEqual(dx, 0.0, delta=0.6)
        self.assertAlmostEqual(dy, self.SPEED * self.LATERAL_COEF, delta=0.6)

    def test_crosswind_270_lowers_y_leaves_x(self):
        dx, dy = self._mean_shift({"speed_mph": self.SPEED, "direction_deg": 270.0})
        self.assertAlmostEqual(dx, 0.0, delta=0.6)
        self.assertAlmostEqual(dy, -self.SPEED * self.LATERAL_COEF, delta=0.6)

    def test_zero_speed_exactly_equals_none(self):
        # Zero-mph wind must produce the EXACT pre-wind landing sequence.
        rng_a = random.Random(1234)
        no_wind = MODEL.sample_many(START, self.AIM_X, CLUB, None, rng_a, 50)
        rng_b = random.Random(1234)
        zero_wind = MODEL.sample_many(
            START, self.AIM_X, CLUB, None, rng_b, 50,
            wind={"speed_mph": 0.0, "direction_deg": 90},
        )
        self.assertEqual(no_wind, zero_wind)

    def test_crosswind_does_not_change_draw_order(self):
        # Wind shifts only Gaussian MEANS; the random stream and spread are
        # untouched. Under an identical draw stream the windy spread equals
        # the calm spread (pointwise identical gauss draws).
        rng_a = random.Random(5)
        no_wind = MODEL.sample_many(START, self.AIM_X, CLUB, None, rng_a, 200)
        rng_b = random.Random(5)
        windy = MODEL.sample_many(
            START, self.AIM_X, CLUB, None, rng_b, 200,
            wind={"speed_mph": 15.0, "direction_deg": 90.0},
        )
        sx = statistics.pstdev(p.x for p in no_wind)
        wx = statistics.pstdev(p.x for p in windy)
        self.assertAlmostEqual(sx, wx, delta=0.3)
        self.assertAlmostEqual(
            statistics.mean(p.y for p in windy) - statistics.mean(p.y for p in no_wind),
            15.0 * self.LATERAL_COEF,
            delta=1.2,
        )

    def test_non_cardinal_wind_matches_direct_vector_math(self):
        # 135 deg wind on a +x shot: along = w*cos(135), lateral = w*sin(135).
        rng = random.Random(21)
        wind = {"speed_mph": 10.0, "direction_deg": 135.0}
        shift_x, shift_y = self._mean_shift(wind)
        along = 10.0 * math.cos(math.radians(135.0)) * self.ALONG_COEF
        lateral = 10.0 * math.sin(math.radians(135.0)) * self.LATERAL_COEF
        # Rotate the strike-frame shift back to engine axes (aim bearing 0).
        self.assertAlmostEqual(shift_x, along, delta=0.6)
        self.assertAlmostEqual(shift_y, lateral, delta=0.6)

    def test_configurable_coefficients_scale_the_shift(self):
        custom = GaussianDispersionModel(SimulationConfig(
            along_wind_carry_yd_per_mph=3.0,
            crosswind_lateral_drift_yd_per_mph=2.0,
        ))
        rng = random.Random(7)
        base = custom.sample_many(START, self.AIM_X, CLUB, None, rng, 2000)
        mean_y0 = statistics.mean(p.y for p in base)
        rng = random.Random(7)
        windy = custom.sample_many(
            START, self.AIM_X, CLUB, None, rng, 2000,
            wind={"speed_mph": 10.0, "direction_deg": 90.0},
        )
        self.assertAlmostEqual(
            statistics.mean(p.y for p in windy) - mean_y0,
            10.0 * 2.0, delta=0.6,
        )


if __name__ == "__main__":
    unittest.main()
