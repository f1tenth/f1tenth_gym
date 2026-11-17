"""Runner/viewer integration: opt in with RIVAL_VIEWER_GL_TEST=1.

Real Gym and a hidden OpenGL window verify that presentation changes neither
commands nor simulated outcomes. No display is needed for ordinary discovery.
"""
import csv
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rival_advanced import run
from rival_baseline import load
from rival_perception import SensorConfig
from rival_robust import RobustConfig


class PresentationOptionsTests(unittest.TestCase):
    def test_invalid_viewer_is_rejected_before_gym_setup(self):
        with self.assertRaisesRegex(ValueError, 'viewer_kind'):
            run(None, True, None, None, None, None, None, None, None, None,
                viewer_kind='unsupported')

    def test_invalid_pacing_is_rejected_before_gym_setup(self):
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'render_speed'):
                run(None, True, None, None, None, None, None, None, None, None,
                    render_speed=value)


@unittest.skipUnless(os.environ.get('RIVAL_VIEWER_GL_TEST') == '1',
                     'set RIVAL_VIEWER_GL_TEST=1 with an OpenGL display')
class PresentationIntegrationTests(unittest.TestCase):
    def setUp(self):
        import yaml
        source = Path(__file__).with_name('rival_advanced.yaml')
        raw = yaml.safe_load(source.read_text())
        cfg, self.line, self.track, self.map_base, self.map_ext = load(source)
        self.cfg = replace(cfg, duration=.2)
        self.rc = RobustConfig.from_dict(raw['robust'])
        self.sc = SensorConfig(**raw['sensor'])

    def simulate(self, output, render):
        return run('adaptive', True, self.cfg, self.rc, self.sc, self.line,
                   self.track, self.map_base, self.map_ext, output,
                   render=render, render_speed=0.)

    def test_hidden_render_matches_headless_commands_and_outcomes(self):
        self.assert_render_matches_headless()

    def test_fast_profile_render_matches_headless(self):
        self.cfg = replace(self.cfg, defender_speed=3., attacker_speed=3.4,
                           offset_rate=.45, initial_gap=8., minimum_gap=2.5)
        self.rc = replace(self.rc, track_speed_planning=True,
                           compiled_attack_preview=True, corner_tracking=True)
        self.assert_render_matches_headless()

    def assert_render_matches_headless(self):
        from rival_viewer import RaceViewer
        instances = []

        def hidden_viewer(*args, **kwargs):
            viewer = RaceViewer(*args, **kwargs, visible=False)
            instances.append(viewer)
            return viewer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            headless = self.simulate(root / 'headless', False)
            with patch('rival_viewer.RaceViewer', side_effect=hidden_viewer):
                rendered = self.simulate(root / 'rendered', True)
            self.assertEqual(len(instances), 1)
            self.assertIsNone(instances[0].window.context)
            for key in ('progress', 'elapsed', 'stop_reason', 'collisions', 'offtrack',
                        'numerical_failures', 'racecraft', 'physics_steps'):
                self.assertEqual(headless[key], rendered[key], key)
            records = []
            for name in ('headless', 'rendered'):
                with (root / name).with_suffix('.csv').open(newline='') as stream:
                    records.append(list(csv.DictReader(stream)))
            self.assertEqual(len(records[0]), len(records[1]))
            for a, b in zip(*records):
                for key in ('time', 'target', 'steering', 'speed', 'attacker_target',
                            'attacker_steering', 'attacker_speed_command', 'racecraft_phase'):
                    self.assertEqual(a[key], b[key], key)

    def test_user_close_is_explicit_not_success_or_interruption(self):
        from rival_viewer import RaceViewer
        instances = []

        def closed_viewer(*args, **kwargs):
            viewer = RaceViewer(*args, **kwargs, visible=False)
            viewer.close()
            instances.append(viewer)
            return viewer

        with tempfile.TemporaryDirectory() as directory:
            with patch('rival_viewer.RaceViewer', side_effect=closed_viewer):
                result = self.simulate(Path(directory) / 'closed', True)
        self.assertEqual(result['stop_reason'], 'viewer_closed')
        self.assertFalse(result['completed'])
        self.assertFalse(result['incomplete'])
        self.assertEqual(result['action_cycles'], 1)
        self.assertIsNone(instances[0].window.context)


if __name__ == '__main__':
    unittest.main()
