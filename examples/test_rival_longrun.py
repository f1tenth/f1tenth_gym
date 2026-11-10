"""Opt-in real-Gym adaptive regressions (no mocks or simulator resets).

RIVAL_LONG_TESTS=1 enables 60 simulated seconds; RIVAL_SOAK_TESTS=1 enables
300 seconds. Run with the project's simulator-capable Python interpreter.
"""
import bisect
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXAMPLES = Path(__file__).resolve().parent


class RivalLongRunTests(unittest.TestCase):
    def run_adaptive(self, duration):
        # Use the CLI's real default YAML/config loading, robust controller,
        # inside region, seed 12345 and .01-second physics substeps unchanged.
        with tempfile.TemporaryDirectory(prefix='rival-longrun-') as directory:
            output = Path(directory) / 'results'
            command = [sys.executable, str(EXAMPLES / 'rival_advanced.py'),
                       '--scenario', 'adaptive', '--duration', str(duration),
                       '--output', str(output)]
            try:
                process = subprocess.run(command, cwd=EXAMPLES.parent,
                                         capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired as exc:
                self.fail(f'Real Gym {duration}s run exceeded 300s wall timeout: {exc}')
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((output / '00_adaptive_robust_defense.json').read_text())
            with (output / '00_adaptive_robust_defense.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.check_result(result, rows, duration)

    def check_result(self, result, rows, duration):
        self.assertEqual(result['settings']['seed'], 12345)
        self.assertEqual(result['robust']['region'], 'inside')
        self.assertAlmostEqual(result['integration_dt'], .01)
        self.assertEqual(result['stop_reason'] in ('duration_limit', 'laps_completed'),
                         True, json.dumps(result, indent=2))
        if result['stop_reason'] == 'duration_limit':
            self.assertAlmostEqual(result['elapsed'], duration, delta=.02)
        else:
            self.assertTrue(result['completed'])
            self.assertTrue(all(car['completed'] for car in result['cars'].values()))
        self.assertTrue(rows, 'No action-level evidence')
        times = [0.] + [float(row['step_end_time']) for row in rows]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertAlmostEqual(times[-1], result['elapsed'], delta=1e-6)
        for label, prefix, minimum in (('defender', 'ego', 60.),
                                       ('attacker', 'attacker', 10.)):
            with self.subTest(car=label):
                car = result['cars'][label]
                for flag in ('numerical_failure', 'collision', 'offtrack', 'locally_stalled'):
                    self.assertFalse(car[flag], f'{label}: {flag}; {car}')
                self.assertEqual(car['numerical_failure_count'], 0)
                self.assertGreater(car['last_rolling_progress'], .25)
                self.assertLess(car['max_stall_time'], 5.)
                if duration == 60:
                    self.assertGreater(car['progress'], minimum)
                progress = [0.] + [float(row[f'{prefix}_progress']) for row in rows]
                self.assertTrue(all(math.isfinite(value) for value in progress))
                self.assertAlmostEqual(progress[-1], car['progress'], delta=1e-6)
                for index, row in enumerate(rows, 1):
                    for flag in ('numerical_failure', 'collision', 'offtrack'):
                        self.assertEqual(row[f'{prefix}_{flag}'], 'False',
                                         f'First counterexample {label}: {row}')
                    self.assertLess(float(row[f'{prefix}_max_stall_time']), 5.,
                                    f'First stall counterexample {label}: {row}')
                    if times[index] < 5.:
                        continue
                    # Independently reconstruct signed rolling progress from CSV,
                    # rather than trusting whole-run averages or JSON summaries.
                    start = times[index] - 5.
                    left = bisect.bisect_right(times, start) - 1
                    fraction = (start - times[left]) / (times[left + 1] - times[left])
                    initial = progress[left] + fraction * (progress[left + 1] - progress[left])
                    rolling = progress[index] - initial
                    self.assertGreater(rolling, .25,
                                       f'First low-progress window {label}: '
                                       f'[{start}, {times[index]}], progress={rolling}; {row}')
                    self.assertAlmostEqual(rolling, float(row[f'{prefix}_rolling_progress_5s']),
                                           delta=.02)

    @unittest.skipUnless(os.environ.get('RIVAL_LONG_TESTS') == '1',
                         'set RIVAL_LONG_TESTS=1 for the real-Gym 60s regression')
    def test_adaptive_60_seconds(self):
        self.run_adaptive(60)

    @unittest.skipUnless(os.environ.get('RIVAL_SOAK_TESTS') == '1',
                         'set RIVAL_SOAK_TESTS=1 for the real-Gym 300s soak')
    def test_adaptive_300_seconds(self):
        self.run_adaptive(300)


if __name__ == '__main__':
    unittest.main()
