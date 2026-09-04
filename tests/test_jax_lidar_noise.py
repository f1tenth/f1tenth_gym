"""Key-driven observed LiDAR noise for the functional JAX sensor."""

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.lidar.functional import (
    ScanConfig,
    ScanParams,
    ScanState,
    observed_scan,
    reset_scan_state,
)


CONFIG = ScanConfig(num_agents=2, num_beams=5, angle_min=-1.0, angle_max=1.0)


def scan_params(**updates):
    params = ScanParams.from_lidar_config(
        LiDARConfig(
            num_beams=CONFIG.num_beams,
            range_min=0.2,
            range_max=10.0,
            noise_std=0.0,
        )
    )
    return replace(params, **updates)


class TestObservedScanContract(unittest.TestCase):
    def test_bias_then_range_limits_then_dropout_order(self):
        clean = jnp.array(
            [[0.1, 1.0, 9.8, 10.0, 4.0], [0.2, 2.0, 9.5, 5.0, 7.0]],
            dtype=jnp.float32,
        )
        bias = jnp.array(
            [[-1.0, 0.5, 1.0, -0.5, 0.0], [0.0, -3.0, 0.6, 0.0, 4.0]],
            dtype=jnp.float32,
        )
        state = ScanState(range_bias=bias)
        actual = observed_scan(
            jax.random.key(1), clean, state, CONFIG, scan_params()
        )
        raw = np.asarray(clean + bias)
        expected = np.where(raw < 0.2, 0.0, np.minimum(raw, 10.0))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(float(actual[0, 0]), 0.0)
        self.assertAlmostEqual(float(actual[1, 0]), 0.2, places=6)

        params = scan_params(dropout_prob=0.5)
        key = jax.random.key(9)
        mixed = observed_scan(key, clean, state, CONFIG, params)
        _noise_key, dropout_key = jax.random.split(key)
        mask = jax.random.uniform(dropout_key, clean.shape) < 0.5
        self.assertTrue(bool(jnp.any(mask)))
        self.assertFalse(bool(jnp.all(mask)))
        expected = np.where(mask, 10.0, expected)
        np.testing.assert_array_equal(mixed, expected)

        dropped = observed_scan(
            jax.random.key(1),
            clean,
            state,
            CONFIG,
            scan_params(dropout_prob=1.0),
        )
        np.testing.assert_array_equal(dropped, np.full((2, 5), 10.0))

    def test_gaussian_draw_matches_the_named_key_split(self):
        key = jax.random.key(17)
        clean = jnp.full((2, 5), 4.0, dtype=jnp.float32)
        params = scan_params(noise_std=0.3)
        actual = observed_scan(
            key,
            clean,
            ScanState(jnp.zeros_like(clean)),
            CONFIG,
            params,
        )
        noise_key, _dropout_key = jax.random.split(key)
        expected = clean + 0.3 * jax.random.normal(noise_key, clean.shape)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

        replay = observed_scan(
            key, clean, ScanState(jnp.zeros_like(clean)), CONFIG, params
        )
        other = observed_scan(
            jax.random.key(18),
            clean,
            ScanState(jnp.zeros_like(clean)),
            CONFIG,
            params,
        )
        np.testing.assert_array_equal(actual, replay)
        self.assertFalse(np.array_equal(actual, other))

    def test_clean_input_and_scan_state_are_not_mutated(self):
        clean = jnp.full((2, 5), 4.0, dtype=jnp.float32)
        state = ScanState(jnp.full((2, 5), 0.25, dtype=jnp.float32))
        clean_before = np.asarray(clean).copy()
        bias_before = np.asarray(state.range_bias).copy()
        observed_scan(jax.random.key(3), clean, state, CONFIG, scan_params())
        np.testing.assert_array_equal(clean, clean_before)
        np.testing.assert_array_equal(state.range_bias, bias_before)

    def test_shape_errors_name_the_bad_leaf(self):
        with self.assertRaisesRegex(ValueError, "clean_ranges"):
            observed_scan(
                jax.random.key(0),
                jnp.zeros((1, 5)),
                ScanState(jnp.zeros((2, 5))),
                CONFIG,
                scan_params(),
            )
        with self.assertRaisesRegex(ValueError, "state.range_bias"):
            observed_scan(
                jax.random.key(0),
                jnp.zeros((2, 5)),
                ScanState(jnp.zeros((2, 4))),
                CONFIG,
                scan_params(),
            )


class TestEpisodeBias(unittest.TestCase):
    def test_reset_is_reproducible_seed_dependent_and_float32(self):
        params = scan_params(range_bias_std=0.4)
        first = reset_scan_state(jax.random.key(5), CONFIG, params)
        replay = reset_scan_state(jax.random.key(5), CONFIG, params)
        other = reset_scan_state(jax.random.key(6), CONFIG, params)
        self.assertEqual(first.range_bias.shape, (2, 5))
        self.assertEqual(first.range_bias.dtype, jnp.float32)
        np.testing.assert_array_equal(first.range_bias, replay.range_bias)
        self.assertFalse(np.array_equal(first.range_bias, other.range_bias))

    def test_zero_bias_is_exact_and_state_stays_fixed_across_steps(self):
        zero = reset_scan_state(jax.random.key(2), CONFIG, scan_params())
        np.testing.assert_array_equal(zero.range_bias, 0.0)

        state = reset_scan_state(
            jax.random.key(2), CONFIG, scan_params(range_bias_std=0.4)
        )
        clean = jnp.full((2, 5), 3.0, dtype=jnp.float32)
        first = observed_scan(
            jax.random.key(10), clean, state, CONFIG, scan_params()
        )
        second = observed_scan(
            jax.random.key(11), clean, state, CONFIG, scan_params()
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, clean + state.range_bias)


class TestTransformability(unittest.TestCase):
    def test_jit_vmap_and_lax_scan_keep_static_shapes(self):
        reset = jax.jit(reset_scan_state, static_argnums=1)
        params = scan_params(range_bias_std=0.1, noise_std=0.05)
        state = reset(jax.random.key(4), CONFIG, params)
        clean = jnp.full((2, 5), 4.0, dtype=jnp.float32)
        run = jax.jit(observed_scan, static_argnums=3)
        self.assertEqual(run(jax.random.key(5), clean, state, CONFIG, params).shape,
                         (2, 5))

        params_batch = jax.tree.map(
            lambda value: jnp.asarray([value, value]), params
        )
        params_batch = replace(
            params_batch,
            noise_std=jnp.asarray([0.0, 0.3], dtype=jnp.float32),
        )
        keys = jax.random.split(jax.random.key(6), 2)
        states = jax.tree.map(lambda value: jnp.stack((value, value)), state)
        batched = jax.jit(
            jax.vmap(
                lambda key, one_state, one_params: observed_scan(
                    key, clean, one_state, CONFIG, one_params
                )
            )
        )(keys, states, params_batch)
        self.assertEqual(batched.shape, (2, 2, 5))
        self.assertFalse(np.array_equal(batched[0], batched[1]))

        step_keys = jax.random.split(jax.random.key(7), 4)

        def body(carry, key):
            observation = observed_scan(key, clean, state, CONFIG, params)
            return carry + 1, observation

        final, rollout = jax.jit(lambda: jax.lax.scan(body, 0, step_keys))()
        self.assertEqual(int(final), 4)
        self.assertEqual(rollout.shape, (4, 2, 5))
        self.assertTrue(bool(jnp.all(jnp.isfinite(rollout))))

    def test_different_traced_bias_scales_vmap_without_static_rebuilds(self):
        params = scan_params()
        params_batch = jax.tree.map(
            lambda value: jnp.asarray([value, value]), params
        )
        params_batch = replace(
            params_batch,
            range_bias_std=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        )
        keys = jax.random.split(jax.random.key(8), 2)
        states = jax.jit(
            jax.vmap(lambda key, one_params: reset_scan_state(
                key, CONFIG, one_params
            ))
        )(keys, params_batch)
        np.testing.assert_array_equal(states.range_bias[0], 0.0)
        self.assertFalse(np.allclose(states.range_bias[1], 0.0))


if __name__ == "__main__":
    unittest.main()
