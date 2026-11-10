"""Simulation-only perception regression tests; no simulator runtime required."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from rival_geometry import ClosedRaceline, TrackMap
from rival_strategy import CarState
from rival_perception import (SensorConfig, Measurement, OpponentEstimate,
                              SimulatedSensor, OpponentTracker, LidarDetector)


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.line = ClosedRaceline([[0, 0], [20, 0], [20, 20], [0, 20]])

    def state(self, s, speed=2., d=0.):
        x, y = self.line.sample(s, d)
        return CarState(s, d, speed, x, y, self.line.heading(s))

    def measure(self, t, s, speed=None, d=0.):
        state = self.state(s, d=d)
        return Measurement(t, state.x, state.y, speed)

    def test_config_validation(self):
        for key in SensorConfig.__dataclass_fields__:
            for value in (-1., np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    SensorConfig(**{key: value})
        for kwargs in ({'dropout': 1.1}, {'update_period': 0}, {'max_speed': 0}):
            with self.assertRaises(ValueError):
                SensorConfig(**kwargs)
        SensorConfig(noise_std=0, latency=0, dropout=1, max_age=0)

    def test_latency_noise_reproducible_and_acquisition_time(self):
        config = SensorConfig(latency=.2, dropout=0)
        a = SimulatedSensor(self.line, config, 42)
        b = SimulatedSensor(self.line, config, 42)
        outputs = []
        for t in np.arange(10) * .1:
            first = a.update(t, self.state(2 + 2 * t))
            self.assertEqual(first, b.update(t, self.state(2 + 2 * t)))
            if first is not None:
                self.assertLessEqual(first.timestamp + config.latency, t)
                outputs.append(first)
        self.assertGreater(len(outputs), 4)
        self.assertEqual(outputs[0].timestamp, 0)
        self.assertNotEqual(outputs[0].x, 2.)
        self.assertLess(abs(outputs[0].x - 2.), .2)
        self.assertIsNone(a.update(.9, self.state(99)))
        with self.assertRaises(ValueError):
            a.update(.8, self.state(0))

    def test_no_backfilled_or_future_truth(self):
        sensor = SimulatedSensor(self.line, SensorConfig(latency=.2, dropout=0,
                                 noise_std=0, speed_noise_std=0), 1)
        self.assertIsNone(sensor.update(0, self.state(2)))
        old = sensor.update(1., self.state(4))
        self.assertEqual((old.timestamp, old.x), (0., 2.))
        self.assertIsNone(sensor.update(1.1, self.state(5)))
        self.assertEqual(sensor.update(1.2, self.state(6)).timestamp, 1.)

    def test_seeded_dropout_and_total_dropout(self):
        config = SensorConfig(dropout=.5, latency=0)
        sensors = [SimulatedSensor(self.line, config, 7) for _ in range(2)]
        sequences = [[sensor.update(i * .1, self.state(2)) for i in range(40)]
                     for sensor in sensors]
        self.assertEqual(*sequences)
        self.assertTrue(any(m is None for m in sequences[0]))
        self.assertTrue(any(m is not None for m in sequences[0]))
        sensor = SimulatedSensor(self.line, SensorConfig(dropout=1), 0)
        self.assertTrue(all(sensor.update(i, self.state(2)) is None for i in range(5)))

    def test_wrap_and_delayed_extrapolation(self):
        tracker = OpponentTracker(self.line, SensorConfig(noise_std=0))
        tracker.update(.1, self.measure(0, 79.8, 2))
        estimate = tracker.update(.3, self.measure(.2, .2, 2))
        self.assertIsInstance(estimate, OpponentEstimate)
        self.assertAlmostEqual(estimate.state.s, .4)
        self.assertAlmostEqual(estimate.state.speed, 2)
        self.assertAlmostEqual(estimate.age, .1)
        self.assertAlmostEqual(estimate.state.x, .4)

    def test_stale_prediction_and_uncertainty_growth(self):
        tracker = OpponentTracker(self.line, SensorConfig())
        self.assertIsNone(tracker.update(0, None))
        first = tracker.update(0, self.measure(0, 2, 2))
        middle = tracker.update(.3, None)
        stale = tracker.update(1., None)
        self.assertTrue(first.fresh and middle.fresh)
        self.assertFalse(stale.fresh)
        self.assertTrue(stale.valid)
        self.assertAlmostEqual(stale.state.s, 4.)
        self.assertLess(first.position_sigma, middle.position_sigma)
        self.assertLess(middle.position_sigma, stale.position_sigma)
        self.assertLess(middle.velocity_sigma, stale.velocity_sigma)

    def test_invalid_packets_never_refresh(self):
        tracker = OpponentTracker(self.line, SensorConfig())
        tracker.update(0, self.measure(0, 2, 2))
        tracker.update(.2, self.measure(.2, 2.4, 2))
        packets = [self.measure(.1, 2.2), self.measure(.2, 2.4),
                   self.measure(2, 3), Measurement(.3, np.nan, 0),
                   Measurement(.3, 3, 0, np.inf), self.measure(.3, 18),
                   self.measure(.3, 2.6, d=3), self.measure(.3, 2.6, speed=100),
                   self.measure(-1, 2)]
        for packet in packets:
            result = tracker.update(.4, packet)
            self.assertAlmostEqual(result.age, .2)
            self.assertAlmostEqual(result.state.s, 2.8)
        with self.assertRaises(ValueError):
            tracker.update(.3, None)
        with self.assertRaises(ValueError):
            tracker.update(np.nan, None)

    def test_bounded_rates_without_measured_speed(self):
        c = SensorConfig()
        tracker = OpponentTracker(self.line, c)
        tracker.update(0, self.measure(0, 2))
        for i in range(1, 20):
            t = i * .1
            result = tracker.update(t, self.measure(t, 2 + 4 * t, d=.3 * t))
            self.assertLessEqual(abs(result.state.speed), c.max_speed)
            self.assertLessEqual(abs(result.longitudinal_acceleration), c.max_acceleration)
            self.assertLessEqual(abs(result.lateral_speed), c.max_lateral_speed)
        result = tracker.update(30, None)
        self.assertLessEqual(abs(result.state.speed), c.max_speed)
        self.assertTrue(np.isfinite(result.state.x))

    def test_lidar_transform_background_association_and_blind_sector(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'map'
            pixels = np.full((400, 400), 255, dtype=np.uint8)
            Image.fromarray(pixels).save(str(base) + '.png')
            Path(str(base) + '.yaml').write_text(yaml.safe_dump(
                {'resolution': .1, 'origin': [-20., -20., 0.]}))
            track = TrackMap(base)
            detector = LidarDetector(self.line, track, SensorConfig())
            ego = CarState(0, 0, 0, 1, 2, np.pi / 2)
            scan = np.full(471, np.inf)
            scan[233:238] = 2.
            result = detector.detect(0, ego, scan)
            self.assertAlmostEqual(result.x, 1., places=6)
            self.assertAlmostEqual(result.y, 4., places=3)
            self.assertIsNone(result.speed)
            # A second, nearer candidate must not steal an established track.
            scan[100:104] = .8
            associated = detector.detect(.1, ego, scan)
            self.assertGreater(associated.y, 3.9)
            # No visible returns: rear and occluded objects cannot be invented.
            self.assertIsNone(detector.detect(.2, ego, np.full(471, np.inf)))
            # End-of-FOV beams are not rearward pi-radian observations.
            scan[:] = np.inf
            scan[:3] = 2
            edge = LidarDetector(self.line, track, SensorConfig()).detect(0, ego, scan)
            expected = np.mean(np.column_stack((
                np.cos(ego.yaw + np.linspace(-2.35, 2.35, 471)[:3]),
                np.sin(ego.yaw + np.linspace(-2.35, 2.35, 471)[:3]))) * 2,
                axis=0) + [ego.x, ego.y]
            np.testing.assert_allclose([edge.x, edge.y], expected)
            # Static boundary returns are removed by actual TrackMap clearance.
            scan[:] = np.inf
            scan[234:237] = 18.
            self.assertIsNone(detector.detect(.3, ego, scan))


if __name__ == '__main__':
    unittest.main()
