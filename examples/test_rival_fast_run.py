"""Opt-in real-Gym regression for the repaired fast profile.

RIVAL_FAST_TESTS=1 python -m unittest discover -s examples -p test_rival_fast_run.py -v
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from rival_robust import RobustConfig
from rival_strategy import Settings

EXAMPLES = Path(__file__).resolve().parent


class FastProfileConfigTests(unittest.TestCase):
    def test_profile_opt_in_does_not_change_default_controller(self):
        default = yaml.safe_load((EXAMPLES / 'rival_advanced.yaml').read_text())
        fast = yaml.safe_load((EXAMPLES / 'rival_advanced_fast.yaml').read_text())
        self.assertFalse(RobustConfig.from_dict(default['robust']).track_speed_planning)
        self.assertTrue(RobustConfig.from_dict(fast['robust']).track_speed_planning)
        self.assertTrue(RobustConfig.from_dict(fast['robust']).corner_tracking)
        settings = Settings.from_dict(fast['settings'])
        self.assertLess(settings.defender_speed, settings.attacker_speed)
        self.assertLessEqual(settings.attacker_speed, fast['robust']['max_opponent_speed'])
        self.assertGreater(settings.initial_gap, settings.car_length + settings.minimum_gap)
        self.assertEqual(fast['robust']['region'], default['robust']['region'])
        self.assertEqual(settings.boundary_margin, Settings().boundary_margin)

    def test_speed_planning_flag_requires_a_boolean(self):
        for name in ('track_speed_planning', 'compiled_attack_preview', 'corner_tracking'):
            for value in (0, 1, 'false', None):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                    RobustConfig(**{name: value})


class FastRunTests(unittest.TestCase):
    def check_result(self, result, rows):
        self.assertTrue(result['speed_planning']['enabled'])
        self.assertEqual(result['stop_reason'], 'laps_completed')
        self.assertTrue(result['completed'])
        self.assertFalse(result['incomplete'])
        self.assertEqual(result['robust']['region'], 'inside')
        self.assertAlmostEqual(result['integration_dt'], .01)
        self.assertLess(result['elapsed'], 120.01)
        self.assertTrue(rows)
        self.assertAlmostEqual(float(rows[-1]['step_end_time']), result['elapsed'], delta=1e-6)
        for label in ('defender', 'attacker'):
            car = result['cars'][label]
            with self.subTest(car=label):
                self.assertTrue(car['completed'])
                self.assertGreaterEqual(car['laps'], 1.)
                for flag in ('collision', 'offtrack', 'numerical_failure', 'locally_stalled'):
                    self.assertFalse(car[flag], car)
                self.assertEqual(car['max_stall_time'], 0.)
        for row in rows:
            for label in ('ego', 'attacker'):
                for flag in ('collision', 'offtrack', 'numerical_failure', 'locally_stalled'):
                    self.assertEqual(row[f'{label}_{flag}'], 'False', f'{label}: {flag} at {row["time"]}')
        for value in result['command_metrics']['max_abs_jerk']:
            self.assertLessEqual(value, result['robust']['max_jerk'] + 1e-7)
        for value in result['command_metrics']['max_abs_target_rate']:
            self.assertLessEqual(value, result['robust']['max_lateral_rate'] + 1e-7)
        # Completion must not erase unresolved decisions or credit a deferred attack.
        self.assertFalse(result['defended_attack_credit'])
        if not result['executed_sampled_feasible_attack_steps']:
            self.assertFalse(result['attack_evaluation']['attack_evaluation_eligible'])
        self.assertTrue(result['robust']['corner_tracking'])
        for index, label in enumerate(('defender', 'attacker')):
            self.assertLessEqual(result['command_metrics']['max_speed'][index],
                                 result['settings'][label + '_speed'] + 1e-7)
        target_rates = [0.] + [float(row['ego_target_rate']) for row in rows]
        for previous, current in zip(target_rates, target_rates[1:]):
            self.assertLessEqual(abs(current - previous) / result['action_period'],
                                 result['robust']['max_lateral_acceleration'] + 1e-7)
        from rival_baseline import load
        from rival_corner_metrics import corner_metrics
        from rival_robust import TrackEnvelope
        _, line, track, _, _ = load(EXAMPLES / 'rival_advanced_fast.yaml')
        vehicle = Settings.from_dict(result['settings'])
        robust = RobustConfig.from_dict(result['robust'])
        envelope = TrackEnvelope(line, track, vehicle, robust)
        metrics = corner_metrics(rows, line, envelope, end_times={
            'def': result['cars']['defender']['finish_time'],
            'att': result['cars']['attacker']['finish_time']})
        for label, p95, peak in (('def', .15, .25), ('att', .06, .10)):
            corner = metrics[label]['corner']
            self.assertGreater(corner['duration'], 10.)
            self.assertEqual(corner['slow_stopped_duration'], 0.)
            self.assertLess(corner['derror']['p95'], p95)
            self.assertLess(corner['derror']['max'], peak)

    @unittest.skipUnless(os.environ.get('RIVAL_FAST_TESTS') == '1',
                         'set RIVAL_FAST_TESTS=1 for the real-Gym fast-profile lap regression')
    def test_adaptive_fast_profile_completes_both_laps(self):
        with tempfile.TemporaryDirectory(prefix='rival-fast-run-') as directory:
            output = Path(directory) / 'results'
            command = [sys.executable, str(EXAMPLES / 'rival_advanced.py'), '--config',
                       str(EXAMPLES / 'rival_advanced_fast.yaml'), '--scenario', 'adaptive',
                       '--duration', '120', '--stall-timeout', '10', '--output', str(output)]
            process = subprocess.run(command, cwd=EXAMPLES.parent, text=True,
                                     capture_output=True, timeout=300)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            base = output / '00_adaptive_robust_defense'
            result = json.loads(base.with_suffix('.json').read_text())
            with base.with_suffix('.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.check_result(result, rows)


class PassingRunTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('RIVAL_PASS_TESTS') == '1',
                         'set RIVAL_PASS_TESTS=1 for the real-Gym controlled passing comparison')
    def test_controlled_speed_advantage_passes_in_matched_runs(self):
        with tempfile.TemporaryDirectory(prefix='rival-pass-run-') as directory:
            output = Path(directory) / 'results'
            command = [sys.executable, str(EXAMPLES / 'rival_advanced.py'), '--config',
                       str(EXAMPLES / 'rival_passing_demo.yaml'), '--scenario', 'left-pass',
                       '--duration', '20', '--compare', '--output', str(output)]
            process = subprocess.run(command, cwd=EXAMPLES.parent, text=True,
                                     capture_output=True, timeout=180)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            times = []
            for label in ('defense', 'no_defense'):
                result = json.loads((output / f'00_left-pass_robust_{label}.json').read_text())
                self.assertEqual(result['robust']['region'], 'inside')
                self.assertEqual(result['settings']['defender_speed'], .5)
                self.assertEqual(result['settings']['attacker_speed'], 2.)
                self.assertEqual(result['stop_reason'], 'duration_limit')
                self.assertEqual(result['completed_passes'], 1)
                self.assertIsNotNone(result['pass_time'])
                self.assertGreater(result['executed_sampled_feasible_attack_steps'], 0)
                self.assertTrue(result['attack_evaluation']['attack_evaluation_eligible'])
                self.assertEqual(result['attacker_unresolved_steps'], 0)
                self.assertFalse(result['defended_attack_credit'])
                for flag in ('collisions', 'offtrack', 'numerical_failures', 'stalled'):
                    self.assertFalse(any(result[flag]), (flag, result[flag]))
                times.append(result['pass_time'])
            # This is a controlled pass probe, not evidence of better defense.
            self.assertAlmostEqual(times[0], times[1], delta=.02)


if __name__ == '__main__':
    unittest.main()
