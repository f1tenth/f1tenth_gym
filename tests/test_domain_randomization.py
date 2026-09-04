"""DomainRandomizationConfig: per-episode vehicle-param randomization.

The range is two ordinary ``VehicleParameters`` (low/high); a field with
``low == high`` is not randomized.
"""
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import (
    EnvConfig, SimulationConfig, DomainRandomizationConfig,
)
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as BASE


def _mk(dr):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            domain_randomization_config=dr,
            render_enabled=False,
        ),
    )


def _dr(**ranges):
    """DomainRandomizationConfig from {field: (low, high)} against the F1TENTH base."""
    low = BASE.with_updates(**{k: v[0] for k, v in ranges.items()})
    high = BASE.with_updates(**{k: v[1] for k, v in ranges.items()})
    return DomainRandomizationConfig(enabled=True, low=low, high=high)


class TestDomainRandomization(unittest.TestCase):
    def test_disabled_keeps_params_constant(self):
        env = _mk(DomainRandomizationConfig())  # disabled
        vals = []
        for s in range(4):
            env.reset(seed=s)
            vals.append(float(env.unwrapped.sim.vehicle_params.m))
        self.assertEqual(len(set(vals)), 1)
        env.close()

    def test_samples_within_range_and_varies(self):
        env = _mk(_dr(m=(3.0, 4.0), mu=(0.9, 1.1)))
        ms, mus, arrays = [], [], []
        for s in range(8):
            env.reset(seed=s)
            p = env.unwrapped.sim.vehicle_params
            ms.append(float(p.m))
            mus.append(float(p.mu))
            arrays.append(env.unwrapped.sim.params_array.copy())
        self.assertTrue(all(3.0 <= m <= 4.0 for m in ms))
        self.assertTrue(all(0.9 <= mu <= 1.1 for mu in mus))
        self.assertGreater(len(set(ms)), 1, "mass did not vary")
        # the params_array (what the dynamics kernels index) must actually change
        self.assertFalse(np.array_equal(arrays[0], arrays[1]))
        env.close()

    def test_reproducible_with_reset_seed(self):
        env = _mk(_dr(m=(3.0, 4.0)))
        env.reset(seed=7)
        a = env.unwrapped.sim.params_array.copy()
        env.reset(seed=7)
        b = env.unwrapped.sim.params_array.copy()
        np.testing.assert_array_equal(a, b)
        env.close()

    def test_only_the_varying_fields_are_randomized(self):
        dr = _dr(m=(3.0, 4.0), mu=(0.9, 1.1))
        self.assertEqual(dr.randomized_fields(), ("mu", "m"))

    def test_untouched_fields_come_back_bit_identical(self):
        # low == high must round-trip exactly, not merely closely: uniform(x, x)
        # is x + 0 * r. Otherwise every reset would jitter the whole vehicle.
        env = _mk(_dr(m=(3.0, 4.0)))
        env.reset(seed=3)
        p = env.unwrapped.sim.vehicle_params
        for name in ("lf", "lr", "I", "h", "v_max", "s_max", "width", "length"):
            self.assertEqual(getattr(p, name), getattr(BASE, name), name)
        env.close()

    def test_nan_multibody_block_survives_sampling(self):
        # uniform() rejects NaN bounds outright, so the non-finite fields must be
        # passed through rather than drawn.
        env = _mk(_dr(m=(3.0, 4.0)))
        env.reset(seed=5)
        p = env.unwrapped.sim.vehicle_params
        self.assertTrue(np.isnan(p.K_zt))
        self.assertEqual(len(p.missing_mb_parameters()), 68)
        env.close()

    def test_nominal_params_are_left_alone(self):
        env = _mk(_dr(m=(3.0, 4.0)))
        env.reset(seed=2)
        self.assertEqual(env.unwrapped.vehicle_params.m, BASE.m)
        env.close()

    def test_randomized_lr_is_forwarded_to_the_renderer(self):
        env = _mk(_dr(lr=(0.1, 0.3)))
        renderer = mock.Mock()
        env.unwrapped.renderer = renderer
        try:
            env.reset(seed=2)
            renderer.update_params.assert_called_once_with(
                env.unwrapped.sim.vehicle_params
            )
        finally:
            env.unwrapped.renderer = None
            env.close()

    def test_config_stays_hashable(self):
        # VehicleParameters is frozen and hashable, so an EnvConfig carrying a DR
        # the config is hashable too.
        self.assertIsInstance(hash(EnvConfig()), int)
        self.assertIsInstance(
            hash(EnvConfig(domain_randomization_config=_dr(m=(3.0, 4.0)))), int
        )

    def test_validation(self):
        with self.assertRaises(ValueError):  # low > high
            DomainRandomizationConfig(
                enabled=True,
                low=BASE.with_updates(m=4.0),
                high=BASE.with_updates(m=3.0),
            )
        with self.assertRaises(ValueError):  # enabled without bounds
            DomainRandomizationConfig(enabled=True)
        with self.assertRaises(TypeError):  # not VehicleParameters
            DomainRandomizationConfig(enabled=True, low={"m": 3.0}, high=BASE)
        with self.assertRaises(ValueError):  # finiteness mismatch
            DomainRandomizationConfig(
                enabled=True, low=BASE, high=BASE.with_updates(K_zt=1.0)
            )

    def test_unknown_field_is_a_typo_error_at_the_bound(self):
        # A bound is a VehicleParameters, so a misspelled field cannot be passed.
        with self.assertRaises(TypeError):
            BASE.with_updates(mass=3.0)

    def test_mb_only_randomization_is_rejected_by_the_environment(self):
        dr = DomainRandomizationConfig(
            enabled=True,
            low=BASE.with_updates(K_zt=1000.0),
            high=BASE.with_updates(K_zt=1200.0),
        )
        with self.assertRaisesRegex(ValueError, "unsupported MB-only fields: K_zt"):
            EnvConfig(domain_randomization_config=dr)


class TestWidestParamSpaces(unittest.TestCase):
    """#6: spaces are a fixed superset of every DR episode."""

    def test_randomized_limits_stay_inside_the_spaces(self):
        cfg = EnvConfig(
            domain_randomization_config=_dr(s_max=(1.2, 1.4), s_min=(-1.4, -1.2)),
            render_enabled=False,
        )
        env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
        obs, _ = env.reset(seed=1)
        for t in range(60):
            action = np.array([[1.4 if t % 2 else -1.4, 3.0]], dtype=np.float32)
            obs, *_ = env.step(action)
            self.assertTrue(env.observation_space.contains(obs), f"step {t} left the space")
        env.close()

    def test_disabled_dr_leaves_spaces_untouched(self):
        plain = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
        off = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                domain_randomization_config=DomainRandomizationConfig(enabled=False),
                render_enabled=False,
            ),
        )
        self.assertEqual(plain.observation_space, off.observation_space)
        self.assertEqual(plain.action_space, off.action_space)
        plain.close()
        off.close()

    def test_update_params_rebuilds_the_observation_space(self):
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
        before = float(env.observation_space["agent_0"]["std_state"].high[3])
        env.unwrapped.update_params(
            env.unwrapped.vehicle_params.with_updates(v_max=40.0)
        )
        after = float(env.observation_space["agent_0"]["std_state"].high[3])
        self.assertGreater(after, before + 15.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
