"""Focused offline corner diagnostics; no simulator, controller or display needed.

.venv-py315/bin/python -m unittest discover -s examples -p test_rival_corner_metrics.py -v
"""

import copy
import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from rival_corner_metrics import corner_metrics, main
from rival_geometry import ClosedRaceline
from rival_robust import RobustConfig, TrackEnvelope
from rival_strategy import Settings


EXAMPLES = Path(__file__).resolve().parent


class ConstantLine:
    def __init__(self, curvature=0., heading=0.):
        self.k, self.angle = curvature, heading

    def curvature(self, s):
        return self.k

    def heading(self, s):
        return self.angle


class Reference:
    def __init__(self, offset=.4, slope=0.):
        self.offset, self.slope = offset, slope

    def reference(self, s):
        return self.offset

    def reference_slope(self, s):
        return self.slope


def row(t=2., dt=.1, *, d=.4, yaw=0., speed=1., steering=.1, s=0.) -> dict[str, Any]:
    result = dict(time=t, action_elapsed=dt, steering=steering,
                  attacker_steering=steering,
                  # These are intentionally wrong: metrics must not use commands/targets.
                  speed=99., attacker_speed_command=99., target=99., attacker_target=99.)
    for prefix in ('ego_truth_eval', 'opponent_truth_eval'):
        result.update({f'{prefix}_{key}': value for key, value in
                       dict(s=s, d=d, yaw=yaw, speed=speed).items()})
    return result


class CornerMetricsTests(unittest.TestCase):
    def test_constant_straight_reference_relative_not_target_relative(self):
        rows = [row(2. + i * .1, d=.7, yaw=.2) for i in range(4)]
        original = copy.deepcopy(rows)
        report = corner_metrics(iter(rows), ConstantLine(), Reference())
        self.assertEqual(rows, original)
        self.assertEqual(report['samples'], 4)
        self.assertEqual(report['parameters']['start_time'], 2.)
        self.assertEqual(report['parameters']['steering_deadband'], .01)
        for car in ('def', 'att'):
            result = report[car]
            self.assertEqual(result['samples'], 4)
            self.assertAlmostEqual(result['duration'], .4)
            for stat in ('rms', 'p95', 'max'):
                self.assertAlmostEqual(result['derror'][stat], .3)
                self.assertAlmostEqual(result['heading_error'][stat], .2)
                self.assertIsNone(result['corner']['derror'][stat])
                self.assertIsNone(result['corner']['heading_error'][stat])
            self.assertEqual(result['corner']['samples'], 0)
            self.assertEqual(result['corner']['duration'], 0.)
            self.assertEqual(result['corner']['slow_stopped_duration'], 0.)
            self.assertIsNone(result['corner']['steering']['total_variation'])
            self.assertEqual(result['steering']['total_variation'], 0.)
        json.dumps(report, allow_nan=False)

    def test_constant_curve_both_cars_have_independent_truth_and_commands(self):
        rows = [row(2. + i * .1, d=.6, yaw=.3, steering=.1) for i in range(3)]
        for index, sample in enumerate(rows):
            sample.update(opponent_truth_eval_d=.1, opponent_truth_eval_yaw=-.4,
                          opponent_truth_eval_speed=.2, attacker_steering=.1 * index)
        report = corner_metrics(rows, ConstantLine(curvature=.1), Reference())
        self.assertAlmostEqual(report['def']['corner']['derror']['rms'], .2)
        self.assertAlmostEqual(report['att']['corner']['derror']['rms'], .3)
        self.assertAlmostEqual(report['def']['corner']['heading_error']['rms'], .3)
        self.assertAlmostEqual(report['att']['corner']['heading_error']['rms'], .4)
        self.assertEqual(report['def']['corner']['slow_stopped_duration'], 0.)
        self.assertAlmostEqual(report['att']['corner']['slow_stopped_duration'], .3)
        self.assertEqual(report['def']['steering']['total_variation'], 0.)
        self.assertAlmostEqual(report['att']['steering']['total_variation'], .2)

    def test_heading_wrap_across_pi_and_multiple_turns(self):
        for turns in (-3, 0, 4):
            with self.subTest(turns=turns):
                report = corner_metrics([row(yaw=-math.pi + .02 + turns * 2 * math.pi)],
                    ConstantLine(.1, math.pi - .03), Reference())
                self.assertAlmostEqual(report['def']['heading_error']['rms'], .05)
                self.assertAlmostEqual(report['att']['heading_error']['max'], .05)

    def test_heading_uses_reference_slope_and_reference_not_measured_offset(self):
        for k, offset, slope in ((.2, 2., .3), (1., 2., .3), (.2, 2., -.3)):
            with self.subTest(k=k, offset=offset, slope=slope):
                heading = math.pi - .01
                expected_heading = heading + math.atan2(slope, 1 - k * offset)
                sample = row(d=offset + 3., yaw=expected_heading + .04 - 2 * math.pi)
                report = corner_metrics([sample], ConstantLine(k, heading), Reference(offset, slope))
                self.assertAlmostEqual(report['def']['corner']['heading_error']['rms'], .04)
                self.assertAlmostEqual(report['def']['derror']['rms'], 3.)

    def test_corner_mask_inclusive_absolute_or_and_per_car_station(self):
        for k, slope, expected in ((.049, .049, 0), (.05, 0., 1),
                                   (-.05, 0., 1), (0., .05, 1), (0., -.05, 1)):
            with self.subTest(k=k, slope=slope):
                report = corner_metrics([row()], ConstantLine(k), Reference(slope=slope))
                self.assertEqual(report['def']['corner']['samples'], expected)
        line = ConstantLine()
        line.curvature = lambda s: .1 if s > 1. else 0.
        sample = row()
        sample['opponent_truth_eval_s'] = 2.
        report = corner_metrics([sample], line, Reference())
        self.assertEqual(report['def']['corner']['samples'], 0)
        self.assertEqual(report['att']['corner']['samples'], 1)

    def test_actual_duration_partial_cycles_and_per_car_finish_clipping(self):
        rows = [row(1.99, .01, d=100., speed=0.),
                row(2., .1, d=1., speed=.2), row(2.1, .01, d=3., speed=-.2),
                row(2.11, .04, d=0., speed=-.3)]
        report = corner_metrics(rows, ConstantLine(.1), Reference(0.),
                                end_times={'def': 2.105, 'att': 2.13})
        defender, attacker = report['def']['corner'], report['att']['corner']
        self.assertEqual(defender['samples'], 2)
        self.assertEqual(attacker['samples'], 3)
        self.assertAlmostEqual(defender['duration'], .105)
        self.assertAlmostEqual(attacker['duration'], .13)
        self.assertAlmostEqual(defender['slow_stopped_duration'], .105)
        self.assertAlmostEqual(attacker['slow_stopped_duration'], .11)
        self.assertAlmostEqual(defender['derror']['rms'], math.sqrt((.1 + 9 * .005) / .105))
        self.assertAlmostEqual(attacker['derror']['rms'], math.sqrt((.1 + 9 * .01) / .13))
        self.assertEqual(defender['derror']['p95'], 1.)
        self.assertEqual(attacker['derror']['p95'], 3.)
        self.assertEqual(defender['derror']['max'], 3.)
        self.assertEqual(report['def']['excluded_samples'],
                         dict(before_start=1, at_or_after_finish=1))
        self.assertEqual(report['duration_sources'], dict(action_elapsed=4, next_time_gap=0))

    def test_finish_boundary_exclusive_start_override_and_no_carry_in(self):
        rows = [row(0., 1.), row(1., 1.), row(2., .2)]
        report = corner_metrics(rows, ConstantLine(.1), Reference(),
                                start_time=.5, end_times=[2., 1.])
        self.assertEqual(report['def']['samples'], 1)
        self.assertEqual(report['def']['duration'], 1.)
        self.assertEqual(report['att']['samples'], 0)
        self.assertEqual(report['def']['excluded_samples']['at_or_after_finish'], 1)
        self.assertEqual(corner_metrics(rows, ConstantLine(), Reference(),
                                       start_time=0.)['def']['samples'], 3)

    def test_next_time_gap_fallback_preserves_partial_final_elapsed(self):
        rows = [row(2., .1), row(2.1, .025), row(2.125, .005)]
        for sample in rows[:-1]:
            del sample['action_elapsed']
        report = corner_metrics(rows, ConstantLine(.1), Reference())
        self.assertAlmostEqual(report['def']['duration'], .13)
        self.assertEqual(report['duration_sources'], dict(action_elapsed=1, next_time_gap=2))
        self.assertEqual(report['def']['steering']['adjacent_pairs'], 2)

    def test_actual_elapsed_beats_next_gap_and_steering_does_not_bridge_gaps(self):
        rows = [row(2., .01, steering=.1), row(2.1, .01, steering=-.1)]
        report = corner_metrics(rows, ConstantLine(.1), Reference())
        self.assertAlmostEqual(report['def']['corner']['duration'], .02)
        self.assertEqual(report['def']['steering']['total_variation'], 0.)
        self.assertEqual(report['def']['steering']['significant_sign_reversals'], 0)
        self.assertEqual(report['def']['steering']['adjacent_pairs'], 0)

    def test_statistics_invariant_to_splitting_constant_observation_intervals(self):
        original = [row(2., .99, d=1.), row(2.99, .01, d=3.)]
        split = [row(2., .33, d=1.), row(2.33, .66, d=1.), row(2.99, .01, d=3.)]
        results = [corner_metrics(rows, ConstantLine(.1), Reference(0.)) for rows in (original, split)]
        for car in ('def', 'att'):
            for stat in ('rms', 'p95', 'max'):
                self.assertAlmostEqual(results[0][car]['derror'][stat], results[1][car]['derror'][stat])
            self.assertEqual(results[0][car]['derror']['p95'], 1.)
            self.assertAlmostEqual(results[0][car]['derror']['rms'], math.sqrt(1.08))

    def test_steering_tv_and_significant_reversals_through_deadband(self):
        commands = [.1, .005, -.004, -.1, -.005, .1]
        rows = [row(2. + i * .1, steering=command) for i, command in enumerate(commands)]
        report = corner_metrics(rows, ConstantLine(.1), Reference())
        for car in ('def', 'att'):
            steering = report[car]['corner']['steering']
            self.assertAlmostEqual(steering['total_variation'],
                                   sum(abs(b - a) for a, b in zip(commands, commands[1:])))
            self.assertEqual(steering['significant_sign_reversals'], 2)
            self.assertEqual(steering['adjacent_pairs'], 5)
        jitter = [row(2. + i * .1, steering=command)
                  for i, command in enumerate([.01, -.01, .001, -.001])]
        result = corner_metrics(jitter, ConstantLine(.1), Reference())['def']['steering']
        self.assertEqual(result['significant_sign_reversals'], 0)
        self.assertGreater(result['total_variation'], 0.)

    def test_corner_steering_does_not_bridge_noncorner_rows(self):
        line = ConstantLine()
        line.curvature = lambda s: 0. if s == 1. else .1
        rows = [row(2., s=0., steering=.1), row(2.1, s=1., steering=.1),
                row(2.2, s=2., steering=-.1)]
        report = corner_metrics(rows, line, Reference())
        self.assertEqual(report['def']['steering']['significant_sign_reversals'], 1)
        self.assertEqual(report['def']['corner']['steering']['significant_sign_reversals'], 0)
        self.assertEqual(report['def']['corner']['steering']['adjacent_pairs'], 0)

    def test_empty_and_fully_excluded_are_strict_json_with_null_statistics(self):
        for rows in ([], [row(t=0.)]):
            report = corner_metrics(rows, ConstantLine(.1), Reference())
            for car in ('def', 'att'):
                self.assertEqual(report[car]['samples'], 0)
                self.assertIsNone(report[car]['derror']['rms'])
                self.assertIsNone(report[car]['heading_error']['p95'])
                self.assertIsNone(report[car]['corner']['derror']['max'])
            json.dumps(report, allow_nan=False)

    def test_missing_and_nonfinite_fields_rejected_even_outside_window(self):
        required = ['time', 'action_elapsed', 'steering', 'attacker_steering']
        required += [f'{prefix}_{field}' for prefix in ('ego_truth_eval', 'opponent_truth_eval')
                     for field in ('s', 'd', 'yaw', 'speed')]
        for field in required:
            for value in (None, '', 'nan', math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    sample = row(t=0.)
                    sample[field] = value
                    with self.assertRaisesRegex(ValueError, f'row 0 {field}'):
                        corner_metrics([sample], ConstantLine(), Reference())
            sample = row()
            del sample[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, field):
                corner_metrics([sample], ConstantLine(), Reference())

    def test_invalid_time_order_overlap_unknown_tail_and_duration_rejected(self):
        cases = [([row(2.), row(2.)], 'strictly increasing'),
                 ([row(2.1), row(2.)], 'strictly increasing'),
                 ([row(-1.)], 'nonnegative'),
                 ([row(2., 0.)], 'positive'), ([row(2., -.1)], 'positive'),
                 ([row(2., .2), row(2.1)], 'overlaps')]
        for rows, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                corner_metrics(rows, ConstantLine(), Reference())
        sample = row()
        del sample['action_elapsed']
        with self.assertRaisesRegex(ValueError, 'final interval duration is unknown'):
            corner_metrics([sample], ConstantLine(), Reference())

    def test_invalid_parameters_and_nonfinite_geometry_rejected(self):
        cases: list[dict[str, Any]] = [
            {'start_time': -1.}, {'start_time': math.nan},
            {'end_times': {'def': math.inf}}, {'end_times': {'att': -1.}},
            {'end_times': {'typo': 3.}}, {'end_times': [1.]}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                corner_metrics([], ConstantLine(), Reference(), **options)
        for obj, name in ((ConstantLine(), 'curvature'), (ConstantLine(), 'heading'),
                          (Reference(), 'reference'), (Reference(), 'reference_slope')):
            setattr(obj, name, lambda s: math.nan)
            line = obj if isinstance(obj, ConstantLine) else ConstantLine()
            reference = obj if isinstance(obj, Reference) else Reference()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'row 0 def geometry'):
                corner_metrics([row()], line, reference)

    def test_real_closed_raceline_and_track_envelope_constant_curve(self):
        theta = np.linspace(0., 2 * np.pi, 64, endpoint=False)
        line = ClosedRaceline(5 * np.column_stack((np.cos(theta), np.sin(theta))))
        track = SimpleNamespace(corridor=lambda *args, **kwargs: (-2., 2.))
        envelope = TrackEnvelope(line, track, Settings(legal_min=.4), RobustConfig(region='track'))
        rows = [row(2. + .1 * i, s=i * .2, d=envelope.reference(i * .2) + .1,
                    yaw=line.heading(i * .2) + .03) for i in range(6)]
        report = corner_metrics(rows, line, envelope)
        for car in ('def', 'att'):
            self.assertEqual(report[car]['corner']['samples'], 6)
            self.assertAlmostEqual(report[car]['corner']['derror']['rms'], .1)
            self.assertAlmostEqual(report[car]['corner']['heading_error']['rms'], .03)
        json.dumps(report, allow_nan=False)

    def test_cli_default_fast_config_csv_strings_and_optional_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'actions.csv'
            output = Path(directory) / 'report.json'
            samples = [row(), row(2.1)]
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            # Isolate CLI I/O; real geometry is exercised separately below/above.
            with patch('rival_baseline.load', return_value=(Settings(), ConstantLine(.1), None, None, None)) as load:
                with patch('rival_robust.TrackEnvelope', return_value=Reference()):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        report = main(['--csv', str(source), '--def-end-time', '2.05'])
                    self.assertEqual(json.loads(stdout.getvalue()), report)
                    self.assertEqual(load.call_args.args[0], EXAMPLES / 'rival_advanced_fast.yaml')
                    self.assertAlmostEqual(report['def']['duration'], .05)
                    self.assertTrue(report['source']['configuration']['robust']['track_speed_planning'])
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        report = main(['--csv', str(source), '--output', str(output)])
                    self.assertEqual(stdout.getvalue(), '')
                    self.assertEqual(json.loads(output.read_text()), report)
                    samples[0]['ego_truth_eval_d'] = 'nan'
                    with source.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                        writer.writeheader()
                        writer.writerows(samples)
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        main(['--csv', str(source)])
                    self.assertEqual(error.exception.code, 2)

    def test_actual_saved_run_with_actual_map_reference_is_finite(self):
        source = EXAMPLES / 'rival_results/fast_controlled_trial/00_adaptive_robust_defense.csv'
        if not source.is_file():
            self.skipTest('local saved run is not distributed with the repository')
        from rival_baseline import load
        import yaml
        config = EXAMPLES / 'rival_advanced_fast.yaml'
        raw = yaml.safe_load(config.read_text())
        cfg, line, track, _, _ = load(config)
        envelope = TrackEnvelope(line, track, cfg, RobustConfig.from_dict(raw['robust']))
        with source.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        report = corner_metrics(rows, line, envelope)
        self.assertGreater(report['def']['corner']['samples'], 0)
        self.assertGreater(report['att']['corner']['samples'], 0)
        json.dumps(report, allow_nan=False)
        sample = next(sample for sample in rows if float(sample['time']) >= 2.)
        single = corner_metrics([sample], line, envelope)
        for car, prefix in (('def', 'ego_truth_eval'), ('att', 'opponent_truth_eval')):
            s = float(sample[f'{prefix}_s'])
            expected_d = float(sample[f'{prefix}_d']) - envelope.reference(s)
            tangent_angle = math.atan2(envelope.reference_slope(s),
                                      1 - line.curvature(s) * envelope.reference(s))
            angle = float(sample[f'{prefix}_yaw']) - line.heading(s) - tangent_angle
            expected_heading = math.atan2(math.sin(angle), math.cos(angle))
            self.assertAlmostEqual(single[car]['derror']['rms'], abs(expected_d))
            self.assertAlmostEqual(single[car]['heading_error']['rms'], abs(expected_heading))


if __name__ == '__main__':
    unittest.main()
