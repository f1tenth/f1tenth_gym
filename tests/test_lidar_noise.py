"""Richer LiDAR observation noise: dropout + per-beam systematic bias."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
from f1tenth_gym.envs.lidar import LiDARConfig

_MAX = 30.0


def _scan_at_fixed_pose(env, seed=1):
    cl = env.unwrapped.track.centerline
    pose = np.array([[float(cl.xs[100]), float(cl.ys[100]), 0.0]], dtype=np.float32)
    env.reset(seed=seed, options={"poses": pose})
    obs, *_ = env.step(np.array([[0.0, 0.0]], dtype=np.float32))
    return obs["agent_0"]["scan"].copy()


def _env(**lidar):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            lidar_config=LiDARConfig(noise_std=0.0, **lidar),
            render_enabled=False,
        ),
    )


class TestLidarNoise(unittest.TestCase):
    def test_ranges_below_min_are_zero(self):
        base_env = _env()
        limited_env = _env(range_min=29.9)
        try:
            base = _scan_at_fixed_pose(base_env)
            limited = _scan_at_fixed_pose(limited_env)
        finally:
            base_env.close()
            limited_env.close()

        expected = np.where(base < 29.9, 0.0, base)
        np.testing.assert_array_equal(limited, expected)
        self.assertTrue(np.any(limited == 0.0))

    def test_dropout_sets_beams_to_max_range(self):
        base = _scan_at_fixed_pose(_env())
        base_frac = float((base >= _MAX - 1e-3).mean())
        env = _env(dropout_prob=0.5)
        d = _scan_at_fixed_pose(env)
        self.assertGreater(float((d >= _MAX - 1e-3).mean()), base_frac + 0.3)
        env.close()

    def test_bias_reproducible_and_seed_dependent(self):
        env = _env(range_bias_std=1.0)
        a = _scan_at_fixed_pose(env, seed=5)
        b = _scan_at_fixed_pose(env, seed=5)
        np.testing.assert_array_equal(a, b)  # reproducible per reset seed
        c = _scan_at_fixed_pose(env, seed=9)
        self.assertFalse(np.allclose(a, c))  # different seed -> different bias
        env.close()

    def test_defaults_backward_compatible(self):
        base = _scan_at_fixed_pose(_env())
        same = _scan_at_fixed_pose(_env())  # dropout=0, bias=0
        np.testing.assert_array_equal(base, same)

    def test_validation(self):
        with self.assertRaises(ValueError):
            LiDARConfig(dropout_prob=1.5)
        with self.assertRaises(ValueError):
            LiDARConfig(range_bias_std=-0.1)


if __name__ == "__main__":
    unittest.main()
