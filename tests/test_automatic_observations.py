"""The automatic pipeline's output contract, and the speed layer that reads it.

Two rules this pins down, both learned from reading the reference repos rather
than their READMEs:

  * GhostBall's pipeline_integration.py falls back to `[52.0, 34.0]` -- the
    centre of a 105x68 m pitch -- whenever a ball position is missing, so every
    downstream number on those frames is computed from a coordinate nobody
    measured. Our equivalent must FAIL CLOSED: a missing target is unavailable,
    never a default coordinate.
  * Metric speed needs BOTH a spatial calibration and real capture time. The
    Rory clip carries only its playback rate (30000/1001); its capture rate is
    unknown, so mph is not derivable and must be refused, not estimated.
"""
import math
import unittest

from ghostcaddie.upload.observations import (
    Calibration, MetricSpeedUnavailable, PointObservation, Timebase,
    image_speed_px_s, metric_speed_mps, observations_from_records,
)


class ContractTests(unittest.TestCase):
    def test_a_missing_target_is_unavailable_not_a_default_coordinate(self):
        recs = [{"source_frame": 10, "visible": True, "point_xy": [100.0, 50.0]},
                {"source_frame": 11, "visible": False, "point_xy": None}]
        obs = observations_from_records(recs, target="ball", fps=30000 / 1001,
                                        method="test", source_sha256="a" * 64)
        self.assertEqual(len(obs), 1, "a non-visible frame must emit no observation")
        self.assertEqual(obs[0].frame_index, 10)

    def test_no_observation_is_ever_synthesised_for_an_empty_record_set(self):
        self.assertEqual(observations_from_records([], target="ball",
                                                   fps=30.0, method="t",
                                                   source_sha256="a" * 64), [])

    def test_timestamps_come_from_the_frame_index_and_the_real_rate(self):
        recs = [{"source_frame": 3062, "visible": True, "point_xy": [1.0, 2.0]}]
        o = observations_from_records(recs, target="ball", fps=30000 / 1001,
                                      method="t", source_sha256="a" * 64)[0]
        self.assertAlmostEqual(o.t_seconds, 3062 * 1001 / 30000, places=9)


class ImageSpeedTests(unittest.TestCase):
    def _obs(self, f, x, y):
        return PointObservation(target="ball", frame_index=f, x_px=x, y_px=y,
                                t_seconds=f * 1001 / 30000, method="t",
                                source_sha256="a" * 64)

    def test_image_speed_uses_real_elapsed_time(self):
        s = image_speed_px_s(self._obs(10, 0.0, 0.0), self._obs(11, 30.0, 40.0))
        self.assertAlmostEqual(s.distance_px, 50.0)
        self.assertAlmostEqual(s.dt_seconds, 1001 / 30000, places=9)
        self.assertAlmostEqual(s.px_per_second, 50.0 / (1001 / 30000), places=6)
        self.assertEqual(s.frame_gap, 1)

    def test_a_frame_gap_is_preserved_not_hidden(self):
        s = image_speed_px_s(self._obs(10, 0.0, 0.0), self._obs(14, 0.0, 10.0))
        self.assertEqual(s.frame_gap, 4)
        self.assertTrue(s.spans_gap)

    def test_out_of_order_observations_are_refused(self):
        with self.assertRaises(ValueError):
            image_speed_px_s(self._obs(11, 0.0, 0.0), self._obs(10, 1.0, 1.0))

    def test_speed_between_different_targets_is_refused(self):
        a = self._obs(10, 0.0, 0.0)
        b = PointObservation(target="clubhead", frame_index=11, x_px=1.0, y_px=1.0,
                             t_seconds=11 * 1001 / 30000, method="t",
                             source_sha256="a" * 64)
        with self.assertRaises(ValueError):
            image_speed_px_s(a, b)


class MetricSpeedTests(unittest.TestCase):
    def _pair(self):
        mk = lambda f, x: PointObservation(target="ball", frame_index=f, x_px=x,
                                           y_px=0.0, t_seconds=f * 1001 / 30000,
                                           method="t", source_sha256="a" * 64)
        return mk(10, 0.0), mk(11, 100.0)

    def test_mph_is_refused_without_a_calibration(self):
        a, b = self._pair()
        with self.assertRaises(MetricSpeedUnavailable) as e:
            metric_speed_mps(a, b, calibration=None,
                             timebase=Timebase(real_seconds_per_playback_second=1.0,
                                               method="t"))
        self.assertIn("calibration", str(e.exception).lower())

    def test_mph_is_refused_without_a_known_capture_rate(self):
        """The Rory case exactly: playback rate known, capture rate unknown."""
        a, b = self._pair()
        cal = Calibration(meters_per_pixel=0.01, uncertainty=0.001, method="t")
        with self.assertRaises(MetricSpeedUnavailable) as e:
            metric_speed_mps(a, b, calibration=cal,
                             timebase=Timebase(real_seconds_per_playback_second=None,
                                               method="unknown capture rate"))
        self.assertIn("capture", str(e.exception).lower())

    def test_metric_speed_is_produced_only_when_both_inputs_are_explicit(self):
        a, b = self._pair()
        cal = Calibration(meters_per_pixel=0.01, uncertainty=0.001, method="t")
        # 4x slow motion: one playback second holds 0.25 real seconds
        tb = Timebase.from_slowmo_multiple(4.0, method="declared 4x")
        m = metric_speed_mps(a, b, calibration=cal, timebase=tb)
        # 100 px * 0.01 m/px = 1 m, over a real dt four times SHORTER than playback
        self.assertAlmostEqual(m.meters, 1.0, places=9)
        self.assertAlmostEqual(m.real_dt_seconds, (1001 / 30000) / 4.0, places=12)
        self.assertAlmostEqual(m.meters_per_second, 1.0 / ((1001 / 30000) / 4.0),
                               places=6)
        self.assertTrue(m.calibration_method)
        self.assertGreater(m.uncertainty_fraction, 0.0)

    def test_a_zero_or_negative_slowmo_factor_is_refused(self):
        a, b = self._pair()
        cal = Calibration(meters_per_pixel=0.01, uncertainty=0.001, method="t")
        for bad in (0.0, -2.0, float("nan")):
            with self.assertRaises((MetricSpeedUnavailable, ValueError)):
                metric_speed_mps(
                    a, b, calibration=cal,
                    timebase=Timebase(real_seconds_per_playback_second=bad,
                                      method="t"))


if __name__ == "__main__":
    unittest.main()


class CrossSourceTests(unittest.TestCase):
    """Two observations of "the ball" from two different videos are not two
    observations of the same ball. Measuring between them is meaningless, and
    the target check alone did not catch it."""

    def _obs(self, f, x, sha):
        return PointObservation(target="ball", frame_index=f, x_px=x, y_px=0.0,
                                t_seconds=f * 1001 / 30000, method="t",
                                source_sha256=sha)

    def test_speed_across_two_different_sources_is_refused(self):
        a, b = self._obs(10, 0.0, "a" * 64), self._obs(11, 30.0, "b" * 64)
        with self.assertRaises(ValueError) as e:
            image_speed_px_s(a, b)
        self.assertIn("source", str(e.exception).lower())

    def test_frame_indices_must_increase_not_merely_timestamps(self):
        a = self._obs(11, 0.0, "a" * 64)
        b = PointObservation(target="ball", frame_index=11, x_px=5.0, y_px=0.0,
                             t_seconds=a.t_seconds + 0.01, method="t",
                             source_sha256="a" * 64)
        with self.assertRaises(ValueError) as e:
            image_speed_px_s(a, b)
        self.assertIn("frame", str(e.exception).lower())

    def test_metric_speed_also_refuses_a_cross_source_pair(self):
        from ghostcaddie.upload.observations import Calibration, Timebase
        a, b = self._obs(10, 0.0, "a" * 64), self._obs(11, 100.0, "b" * 64)
        cal = Calibration(meters_per_pixel=0.01, uncertainty=0.001, method="t")
        with self.assertRaises(ValueError):
            metric_speed_mps(a, b, calibration=cal,
                             timebase=Timebase(real_seconds_per_playback_second=1.0,
                                               method="t"))


class TimebaseConventionTests(unittest.TestCase):
    """The documented formula and the arithmetic disagreed.

    Timebase said `real_action_elapsed / playback_elapsed`, which means
    real_dt = playback_dt * factor. The code divided instead, so a declared
    factor of 0.25 produced 2.5 m/s where the documented convention gives 40.

    The field is now named for exactly what it is, and the reciprocal is
    available only through an explicit constructor.
    """

    def _pair(self):
        mk = lambda f, x: PointObservation(
            target="ball", frame_index=f, x_px=x, y_px=0.0, t_seconds=float(f),
            method="t", source_sha256="a" * 64)
        return mk(0, 0.0), mk(1, 10.0)          # 10 px over 1 playback second

    def test_the_documented_factor_multiplies_playback_time(self):
        from ghostcaddie.upload.observations import Calibration, Timebase
        a, b = self._pair()
        cal = Calibration(meters_per_pixel=1.0, uncertainty=0.0, method="t")
        tb = Timebase(real_seconds_per_playback_second=0.25, method="declared")
        m = metric_speed_mps(a, b, calibration=cal, timebase=tb)
        self.assertAlmostEqual(m.meters, 10.0, places=9)
        self.assertAlmostEqual(m.real_dt_seconds, 0.25, places=12)
        self.assertAlmostEqual(m.meters_per_second, 40.0, places=6)

    def test_real_time_is_a_factor_of_one(self):
        from ghostcaddie.upload.observations import Calibration, Timebase
        a, b = self._pair()
        cal = Calibration(meters_per_pixel=1.0, uncertainty=0.0, method="t")
        m = metric_speed_mps(a, b, calibration=cal,
                             timebase=Timebase(real_seconds_per_playback_second=1.0,
                                               method="real time, stated"))
        self.assertAlmostEqual(m.meters_per_second, 10.0, places=9)

    def test_a_slowmo_multiple_converts_explicitly_never_silently(self):
        """4x slow motion means one playback second holds 0.25 real seconds."""
        from ghostcaddie.upload.observations import Timebase
        tb = Timebase.from_slowmo_multiple(4.0, method="declared 4x")
        self.assertAlmostEqual(tb.real_seconds_per_playback_second, 0.25, places=12)

    def test_a_zero_or_negative_slowmo_multiple_is_refused(self):
        from ghostcaddie.upload.observations import Timebase
        for bad in (0.0, -4.0):
            with self.assertRaises(ValueError):
                Timebase.from_slowmo_multiple(bad, method="t")
