"""Periodic speed-planning regressions; no simulator or bitmap map required.

Run: python -m unittest discover -s examples -p test_rival_speed.py -v
"""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.typing import ArrayLike

from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, TrackEnvelope
from rival_speed import SpeedProfile
from rival_strategy import Settings


def rectangle(width=100., height=40., step=1.):
    vertices = np.array([[0., 0.], [width, 0.], [width, height], [0., height]])
    points = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        count = int(np.ceil(np.linalg.norm(end - start) / step))
        points.extend(start + np.arange(count)[:, None] / count * (end - start))
    return ClosedRaceline(points)


def circle(radius=10., direction=1, count=256):
    theta = direction * np.linspace(0., 2 * np.pi, count, endpoint=False)
    return ClosedRaceline(radius * np.column_stack((np.cos(theta), np.sin(theta))))


def reference(line, offsets: ArrayLike = 0., nodes=None, supplied_slope=None):
    if nodes is None:
        nodes = np.linspace(0., line.length, int(np.ceil(line.length / .5)), endpoint=False)
    nodes = np.asarray(nodes, dtype=float)
    offsets = np.broadcast_to(offsets, nodes.shape).copy()
    slopes = (np.roll(offsets, -1) - offsets) / np.diff(np.r_[nodes, line.length])
    return SimpleNamespace(s=nodes, reference_offsets=offsets,
                           reference_slope=(supplied_slope if supplied_slope is not None else
                               lambda s: float(np.interp(s, nodes, slopes, period=line.length))))


class SpeedProfileTests(unittest.TestCase):
    def setUp(self):
        self.vehicle = Settings()
        self.config = RobustConfig(region='track')
        self.line = rectangle()

    def profile(self, line=None, envelope=None, vehicle=None, config=None, speed_limit=20.):
        return SpeedProfile(self.line if line is None else line, envelope,
                            self.vehicle if vehicle is None else vehicle,
                            self.config if config is None else config, speed_limit)

    def test_flat_straight_has_no_arbitrary_speed_penalty(self):
        line = rectangle(200., 80.)
        profile = self.profile(line=line)
        self.assertEqual(line.curvature(80.), 0.)
        self.assertEqual(profile.limit(80.), 20.)
        self.assertEqual(profile.limit(80., 20.), 20.)
        self.assertEqual(profile.raw_limits[np.searchsorted(profile.s, 80.)], 20.)
        zero = self.profile(line=line, speed_limit=0.)
        self.assertEqual(zero.limit(80., 20.), 0.)

    def test_isolated_future_corner_slows_before_local_curvature_or_lookahead(self):
        profile = self.profile()
        s = 80.  # Twenty metres before an isolated right-angle bend at s=100.
        i = np.searchsorted(profile.s, s)
        self.assertEqual(self.line.curvature(s), 0.)
        self.assertEqual(self.line.curvature(s + self.vehicle.lookahead), 0.)
        self.assertEqual(profile.raw_limits[i], 20.)
        self.assertLess(profile.speeds[i], 20.)
        self.assertLess(profile.limit(s), profile.speeds[i])
        self.assertLess(profile.limit(s, 20.), profile.limit(s, 0.))
        self.assertLess(profile.limit(90.), profile.limit(70.))

    def test_cyclic_wrap_and_negative_positions(self):
        profile = self.profile()
        for s in (-2.3, 0., .1, 60., self.line.length - .1):
            for turns in (-7, -1, 0, 1, 9):
                self.assertAlmostEqual(profile.limit(s, 5.),
                                       profile.limit(s + turns * self.line.length, 5.), places=10)
        self.assertLess(profile.limit(self.line.length - 10.), 20.)
        self.assertAlmostEqual(profile.limit(-1e-8), profile.limit(1e-8), places=6)
        self.assertEqual(profile.limit(40., -10.), profile.limit(40., 0.))

    def test_backward_braking_inequality_including_closing_edge_and_exact_oracle(self):
        line = rectangle(20., 10.)
        profile = self.profile(line=line, speed_limit=8.)
        squared = profile.speeds ** 2
        budget = 2 * self.config.max_braking * profile.segment_distances
        self.assertTrue(np.all(squared <= np.roll(squared, -1) + budget + 1e-12))
        self.assertLessEqual(squared[-1], squared[0] + budget[-1] + 1e-12)
        for i in range(0, len(squared), 7):
            indices = (i + np.arange(len(squared))) % len(squared)
            distances = np.r_[0., np.cumsum(profile.segment_distances[indices[:-1]])]
            oracle = np.min(profile.raw_limits[indices] ** 2
                            + 2 * self.config.max_braking * distances)
            self.assertAlmostEqual(squared[i], oracle, places=10)
        self.assertLessEqual(float(profile.diagnostics['max_braking_residual']), 1e-12)
        self.assertGreater(int(profile.diagnostics['braking_limited_nodes']), 0)

    def test_plan_is_equivariant_to_moving_lap_seam(self):
        # All spacings divide one metre; shifting a seam cannot hide a bend.
        points = self.line.points
        shift = 37
        other_line = ClosedRaceline(np.roll(points, -shift, axis=0))
        first, other = self.profile(), self.profile(line=other_line)
        for s in np.linspace(0., self.line.length, 111):
            self.assertAlmostEqual(first.limit(s + shift, 4.), other.limit(s, 4.), places=8)

    def test_direction_and_signed_offsets_on_circle(self):
        profiles = {}
        for direction in (-1, 1):
            line = circle(direction=direction)
            for inside in (False, True):
                offset = direction * (1. if inside else -1.)
                profile = self.profile(line=line, envelope=reference(line, offset))
                profiles[direction, inside] = profile
                expected = direction / (9. if inside else 11.)
                np.testing.assert_allclose(profile.reference_curvature, expected, atol=1e-11)
                self.assertTrue(np.all(profile.curvature >= np.abs(profile.line_curvature)))
                self.assertAlmostEqual(float(profile.diagnostics['max_reference_slope']), 0.)
                self.assertAlmostEqual(profile.limit(10.), profiles.get((-direction, inside), profile).limit(10.),
                                       places=8)
            self.assertLess(profiles[direction, True].limit(10.), profiles[direction, False].limit(10.))
        # Outside offset must not override the conservative underlying line cap.
        outside = profiles[1, False]
        np.testing.assert_allclose(outside.curvature, .1, atol=1e-11)

    def test_offset_second_derivative_and_slope_use_metres_not_indices(self):
        line = rectangle(40., 20.)
        vehicle = replace(self.vehicle, offset_rate=20.)
        configs = SimpleNamespace(**vars(self.config))
        configs.max_lateral_rate = 20.
        frequency = 2 * np.pi / 10.
        for spacing in (.5, .25):
            nodes = np.arange(0., line.length, spacing)
            d = .5 * np.sin(frequency * nodes)
            profile = self.profile(line=line, envelope=reference(line, d, nodes),
                                   vehicle=vehicle, config=configs)
            s = 2.5  # Crest on a genuinely straight section, not a line corner.
            i = np.argmin(np.abs(profile.s - s))
            expected_second = -.5 * frequency ** 2
            self.assertAlmostEqual(profile.reference_second_derivative[i], expected_second, delta=.002)
            self.assertAlmostEqual(profile.reference_curvature[i], expected_second, delta=.002)
            self.assertGreater(profile.curvature[i], .19)
            self.assertLess(profile.lateral_limits[i], 4.)
            # Off-crest curvature includes (1+d'^2)^(-3/2), not just d''.
            s = 1.5
            i = np.argmin(np.abs(profile.s - s))
            first = .5 * frequency * np.cos(frequency * s)
            second = -.5 * frequency ** 2 * np.sin(frequency * s)
            self.assertAlmostEqual(profile.reference_curvature[i], second / (1 + first ** 2) ** 1.5,
                                   delta=.002)

    def test_nonuniform_reference_derivatives_and_geometry_distance(self):
        line = rectangle(40., 20.)
        nodes = np.sort(np.r_[np.arange(0., 8.), [8., 8.3, 9.1, 10.4, 11.2, 12.],
                              np.arange(13., line.length)])
        offsets = np.where((nodes >= 5) & (nodes <= 15), .02 * (nodes - 10) ** 2, 0.)
        profile = self.profile(line=line, envelope=reference(line, offsets, nodes))
        i = np.argmin(np.abs(profile.s - 10.4))
        self.assertAlmostEqual(profile.reference_second_derivative[i], .04, places=12)
        self.assertAlmostEqual(profile.reference_metric[i], np.sqrt(1 + .016 ** 2), places=12)
        self.assertAlmostEqual(profile.reference_curvature[i], .04 / (1 + .016 ** 2) ** 1.5, places=12)
        ds = np.diff(np.r_[profile.s, line.length])
        self.assertTrue(np.all(profile.segment_distances <= ds + 1e-12))
        inner_line = circle()
        inner = self.profile(line=inner_line, envelope=reference(inner_line, 2.))
        self.assertLess(np.sum(inner.segment_distances), .81 * inner_line.length)
        np.testing.assert_allclose(inner.reference_metric, .8, atol=1e-12)

    def test_reference_slope_bound_includes_supplied_slope_and_closure(self):
        envelope = reference(self.line, supplied_slope=lambda s: -.5)
        profile = self.profile(envelope=envelope)
        expected = .8 * self.vehicle.offset_rate / .5
        np.testing.assert_allclose(profile.slope_limits, expected)
        self.assertAlmostEqual(profile.limit(50.), expected)
        nodes = np.arange(0., self.line.length, .5)
        offsets = np.zeros_like(nodes)
        offsets[-1] = .1
        changing = self.profile(envelope=reference(self.line, offsets, nodes))
        self.assertGreater(changing.reference_slope[0], .19)
        self.assertTrue(np.all(changing.slope_limits * changing.reference_slope
                               <= .8 * self.vehicle.offset_rate + 1e-12))
        self.assertLess(changing.slope_limits[0], 20.)

    def test_full_preview_minimum_does_not_skip_corner_and_matches_brute_force(self):
        profile = self.profile()
        rng = np.random.default_rng(789)
        for _ in range(100):
            s = rng.uniform(0., profile.length)
            measured = rng.uniform(0., 100.)
            local = np.sqrt(profile._at(s))
            distance = (self.vehicle.lookahead + max(local, measured) * profile._response_time
                        + .5 * self.config.max_acceleration * profile._response_time ** 2)
            forward = (profile.s - s) % profile.length
            inside = forward <= distance
            candidates = [profile._at(s), profile._at(s + distance)]
            if np.any(inside):
                candidates.append(np.min(profile.speeds[inside] ** 2))
            expected = np.sqrt(min(candidates))
            self.assertAlmostEqual(profile.limit(s, measured), expected, places=10)
        self.assertAlmostEqual(profile.limit(50., 1e6), np.min(profile.speeds), places=12)
        # Endpoints can both be fast; the internal corner must still be seen.
        start, span = 90., 30.
        self.assertLess(profile._front_minimum(start, span),
                        min(profile._at(start), profile._at(start + span)))

    def test_response_reserve_and_optional_configuration(self):
        config = SimpleNamespace(**vars(self.config), speed_reaction_time=0., speed_jerk_reserve=0.,
                                 speed_lateral_reserve=1., speed_slope_reserve=1.)
        plain = self.profile(config=config)
        reserved = self.profile()
        self.assertLess(reserved.limit(80., 10.), plain.limit(80., 10.))
        self.assertEqual(self.profile(line=rectangle(200., 80.)).limit(80.), 20.)
        slower_jerk = self.profile(config=replace(self.config, max_jerk=1.))
        self.assertLess(slower_jerk.limit(80., 10.), reserved.limit(80., 10.))
        slower_brakes = self.profile(config=replace(self.config, max_braking=1.))
        i = np.searchsorted(plain.s, 70.)
        self.assertLess(slower_brakes.speeds[i], plain.speeds[i])
        alias = dict(vars(self.config))
        alias['max_accel'] = alias.pop('max_acceleration')
        aliased = self.profile(config=SimpleNamespace(**alias))
        np.testing.assert_array_equal(aliased.speeds, reserved.speeds)
        self.assertEqual(aliased.limit(80.), reserved.limit(80.))

    def test_real_track_envelope_interface_without_map_queries(self):
        class BroadTrack:
            def corridor(self, *args, **kwargs):
                return -2., 2.

        line = circle()
        envelope = TrackEnvelope(line, BroadTrack(), self.vehicle, self.config)
        profile = self.profile(line=line, envelope=envelope)
        self.assertEqual(profile.diagnostics['reference_source'], 'envelope')
        self.assertTrue(np.isfinite(profile.limit(0.)))
        # Querying the shared profile never revisits geometry or reference APIs.
        with patch.object(line, 'sample', side_effect=AssertionError('online geometry')), \
                patch.object(line, 'curvature', side_effect=AssertionError('online curvature')), \
                patch.object(envelope, 'reference_slope', side_effect=AssertionError('online slope')):
            for s in np.linspace(-100., 100., 200):
                self.assertTrue(np.isfinite(profile.limit(s, 5.)))
        with self.assertRaises(ValueError):
            profile.speeds[0] = 0.
        self.assertEqual(profile.diagnostics['reference_nodes'], len(envelope.s))

    def test_rejects_nonfinite_or_invalid_settings_and_queries(self):
        for name in ('max_lateral_accel', 'max_braking', 'max_acceleration', 'max_jerk',
                     'control_period', 'speed_profile_spacing', 'speed_lateral_reserve',
                     'speed_slope_reserve', 'speed_reaction_time', 'speed_jerk_reserve'):
            for value in (np.nan, np.inf, -1.):
                config = SimpleNamespace(**vars(self.config))
                setattr(config, name, value)
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.profile(config=config)
        for name in ('lookahead', 'offset_rate'):
            for value in (np.nan, np.inf, -1.):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.profile(vehicle=replace(self.vehicle, **{name: value}))
        for speed in (-1., np.nan, np.inf, 1e200):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                self.profile(speed_limit=speed)
        for name in ('speed_lateral_reserve', 'speed_slope_reserve'):
            config = SimpleNamespace(**vars(self.config))
            setattr(config, name, 1.01)
            with self.assertRaises(ValueError):
                self.profile(config=config)
        profile = self.profile()
        for s, v in [(np.nan, 0.), (np.inf, 0.), (0., np.nan), (0., np.inf)]:
            with self.assertRaises(ValueError):
                profile.limit(s, v)

    def test_rejects_invalid_reference_and_nonfinite_geometry(self):
        cases = [([0., 1.], [0., 0.]), ([.1, 1., 2.], [0.] * 3),
                 ([0., 1., 1.], [0.] * 3), ([0., 1., self.line.length], [0.] * 3),
                 ([0., 1., 2.], [0., np.nan, 0.]), ([0., np.inf, 2.], [0.] * 3),
                 ([0., 1., 2.], [0., 0.]), ([0., 1., 2.], [0., 1e300, 0.])]
        for nodes, offsets in cases:
            envelope = SimpleNamespace(s=nodes, reference_offsets=offsets, reference_slope=lambda s: 0.)
            with self.subTest(nodes=nodes, offsets=offsets), self.assertRaises(ValueError):
                self.profile(envelope=envelope)
        for invalid in (np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, 'reference_slope'):
                self.profile(envelope=reference(self.line, supplied_slope=lambda s, invalid=invalid: invalid))
            with patch.object(self.line, 'curvature', return_value=invalid), self.assertRaises(ValueError):
                self.profile()
            with patch.object(self.line, 'sample', return_value=np.array([invalid, 0.])), \
                    self.assertRaises(ValueError):
                self.profile()
        line = circle()
        with self.assertRaisesRegex(ValueError, 'folds|singular'):
            self.profile(line=line, envelope=reference(line, 11.))


if __name__ == '__main__':
    unittest.main()
