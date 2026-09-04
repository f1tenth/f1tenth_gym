"""Contracts for correlated, device-native JAX vehicle randomization."""

from dataclasses import replace
import ast
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.contact.solver import ContactParams
from f1tenth_gym.envs.dynamic_models import (
    F1TENTH_VEHICLE_PARAMETERS,
    PARAMETER_ORDER,
)
from f1tenth_gym.envs.dynamic_models.jax_core import DynamicsRuntimeParams
from f1tenth_gym.envs.jax_core import CoreParams
from f1tenth_gym.envs.episode import EpisodeParams
from f1tenth_gym.envs.lidar.functional import ScanParams, lidar_poses
from f1tenth_gym.envs.dynamic_models.randomization import (
    ACTIVE_VEHICLE_FIELDS,
    ActiveVehicleParams,
    VehicleRandomizationParams,
    domain_randomization_key,
    replace_core_vehicle_params,
    sample_core_params,
    sample_vehicle_params,
)


VEHICLE = F1TENTH_VEHICLE_PARAMETERS
ACTIVE_COUNT = len(ACTIVE_VEHICLE_FIELDS)
NOMINAL_ARRAY = np.asarray(
    [getattr(VEHICLE, name) for name in ACTIVE_VEHICLE_FIELDS],
    dtype=np.float32,
)


def active(values=NOMINAL_ARRAY):
    return ActiveVehicleParams.from_array(jnp.asarray(values, dtype=jnp.float32))


def randomization_spec(enabled=True):
    delta = np.linspace(0.001, 0.020, ACTIVE_COUNT, dtype=np.float32)
    return VehicleRandomizationParams(
        nominal=active(),
        low=active(NOMINAL_ARRAY - delta),
        high=active(NOMINAL_ARRAY + delta),
        enabled=jnp.asarray(enabled, dtype=jnp.bool_),
    )


def core_params():
    vehicle = active()
    return CoreParams(
        dynamics=DynamicsRuntimeParams(
            vehicle=vehicle.to_dynamics(),
            timestep=jnp.asarray(0.01, dtype=jnp.float32),
            steer_kp=jnp.asarray(2.5, dtype=jnp.float32),
            steer_noise_std=jnp.asarray(0.03, dtype=jnp.float32),
            accel_noise_std=jnp.asarray(0.04, dtype=jnp.float32),
        ),
        body=vehicle.to_body(),
        contact=ContactParams(
            *(
                jnp.asarray(value, dtype=jnp.float32)
                for value in (0.1, 0.2, 0.3, 0.4, 0.005)
            )
        ),
        scan=ScanParams(
            range_max=jnp.asarray(30.0, dtype=jnp.float32),
            offset_x=jnp.asarray(0.1, dtype=jnp.float32),
            offset_y=jnp.asarray(0.0, dtype=jnp.float32),
            offset_yaw=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        episode=EpisodeParams(
            max_laps=jnp.asarray(2, dtype=jnp.int32),
            progress_weight=jnp.asarray(3.0, dtype=jnp.float32),
        ),
    )


class TestActiveVehicleABI(unittest.TestCase):
    def test_active_fields_are_the_exact_supported_host_prefix(self):
        self.assertEqual(ACTIVE_VEHICLE_FIELDS, PARAMETER_ORDER[:20])
        self.assertEqual(
            tuple(ActiveVehicleParams.__dataclass_fields__),
            ACTIVE_VEHICLE_FIELDS,
        )

    def test_array_round_trip_preserves_order_shape_and_dtype(self):
        values = jnp.arange(ACTIVE_COUNT, dtype=jnp.float32)
        params = ActiveVehicleParams.from_array(values)
        np.testing.assert_array_equal(params.as_array(), values)
        self.assertEqual(params.as_array().shape, (ACTIVE_COUNT,))
        self.assertEqual(params.as_array().dtype, jnp.float32)

        batched = ActiveVehicleParams.from_array(jnp.stack((values, values + 1)))
        self.assertEqual(batched.as_array().shape, (2, ACTIVE_COUNT))
        np.testing.assert_array_equal(batched.as_array()[1], values + 1)

    def test_from_array_rejects_a_wrong_final_axis(self):
        for bad in (jnp.zeros(()), jnp.zeros((19,)), jnp.zeros((2, 21))):
            with self.assertRaisesRegex(ValueError, "final axis"):
                ActiveVehicleParams.from_array(bad)

    def test_views_share_wheelbase_and_body_offset_values(self):
        values = NOMINAL_ARRAY.copy()
        values[3] = 0.25
        values[4] = 0.40
        values[16] = 0.31
        values[17] = 0.59
        values[18] = 0.15
        values[19] = -0.02
        params = active(values)
        dynamics = params.to_dynamics()
        body = params.to_body()

        self.assertEqual(float(dynamics.lf), 0.25)
        self.assertAlmostEqual(float(dynamics.lr), 0.40, places=6)
        self.assertAlmostEqual(float(body.width), 0.31, places=6)
        self.assertAlmostEqual(float(body.length), 0.59, places=6)
        self.assertAlmostEqual(float(body.centre_x), -0.25, places=6)
        self.assertAlmostEqual(float(body.centre_y), -0.02, places=6)


class TestVehicleSampling(unittest.TestCase):
    def test_named_key_is_stable_and_does_not_consume_reset_splits(self):
        reset_key = jax.random.key(91)
        baseline = jax.random.split(reset_key, 3)
        derived = domain_randomization_key(reset_key)

        np.testing.assert_array_equal(
            jax.random.key_data(derived),
            jax.random.key_data(jax.random.fold_in(reset_key, 0x46314452)),
        )
        np.testing.assert_array_equal(
            jax.random.key_data(jax.random.split(reset_key, 3)),
            jax.random.key_data(baseline),
        )

    def test_replay_and_bounds(self):
        key = domain_randomization_key(jax.random.key(3))
        spec = randomization_spec()
        first = sample_vehicle_params(key, spec).as_array()
        second = sample_vehicle_params(key, spec).as_array()

        np.testing.assert_array_equal(first, second)
        self.assertTrue(bool(jnp.all(first >= spec.low.as_array())))
        self.assertTrue(bool(jnp.all(first <= spec.high.as_array())))

    def test_equal_endpoints_are_exact_and_disabled_is_nominal(self):
        endpoint = jnp.linspace(-2.0, 2.0, ACTIVE_COUNT, dtype=jnp.float32)
        exact = VehicleRandomizationParams(
            nominal=active(),
            low=ActiveVehicleParams.from_array(endpoint),
            high=ActiveVehicleParams.from_array(endpoint),
            enabled=jnp.asarray(True),
        )
        for seed in (0, 1, 200):
            np.testing.assert_array_equal(
                sample_vehicle_params(jax.random.key(seed), exact).as_array(),
                endpoint,
            )

        disabled = replace(randomization_spec(), enabled=jnp.asarray(False))
        for seed in (0, 99):
            np.testing.assert_array_equal(
                sample_vehicle_params(jax.random.key(seed), disabled).as_array(),
                disabled.nominal.as_array(),
            )

    def test_each_field_uses_its_own_folded_abi_stream(self):
        zeros = jnp.zeros((ACTIVE_COUNT,), dtype=jnp.float32)
        ones = jnp.ones((ACTIVE_COUNT,), dtype=jnp.float32)
        spec = VehicleRandomizationParams(
            nominal=ActiveVehicleParams.from_array(0.5 * ones),
            low=ActiveVehicleParams.from_array(zeros),
            high=ActiveVehicleParams.from_array(ones),
            enabled=jnp.asarray(True),
        )
        key = jax.random.key(41)
        actual = sample_vehicle_params(key, spec).as_array()
        expected = jnp.stack(
            tuple(
                jax.random.uniform(
                    jax.random.fold_in(key, index),
                    (),
                    dtype=jnp.float32,
                )
                for index in range(ACTIVE_COUNT)
            )
        )
        np.testing.assert_array_equal(actual, expected)

    def test_changing_one_bound_does_not_shift_other_field_draws(self):
        base = randomization_spec()
        index = ACTIVE_VEHICLE_FIELDS.index("I")
        changed_low = base.low.as_array().at[index].set(10.0)
        changed_high = base.high.as_array().at[index].set(20.0)
        changed = replace(
            base,
            low=ActiveVehicleParams.from_array(changed_low),
            high=ActiveVehicleParams.from_array(changed_high),
        )
        key = jax.random.key(18)
        before = sample_vehicle_params(key, base).as_array()
        after = sample_vehicle_params(key, changed).as_array()

        mask = np.arange(ACTIVE_COUNT) != index
        np.testing.assert_array_equal(np.asarray(before)[mask], np.asarray(after)[mask])
        self.assertGreaterEqual(float(after[index]), 10.0)
        self.assertLessEqual(float(after[index]), 20.0)

    def test_enabled_is_a_traced_runtime_boolean(self):
        spec = randomization_spec()
        key = jax.random.key(52)

        @jax.jit
        def run(enabled):
            return sample_vehicle_params(
                key, replace(spec, enabled=enabled)
            ).as_array()

        np.testing.assert_array_equal(run(jnp.asarray(False)), spec.nominal.as_array())
        enabled = run(jnp.asarray(True))
        self.assertTrue(bool(jnp.all(enabled >= spec.low.as_array())))
        self.assertTrue(bool(jnp.all(enabled <= spec.high.as_array())))


class TestCoreReplacement(unittest.TestCase):
    def test_replacement_is_immutable_and_selective(self):
        base = core_params()
        values = NOMINAL_ARRAY.copy()
        values[0] = 0.75
        values[4] = 0.42
        values[16] = 0.29
        values[17] = 0.63
        values[18] = 0.12
        vehicle = active(values)
        updated = replace_core_vehicle_params(base, vehicle)

        self.assertIs(updated.contact, base.contact)
        self.assertIs(updated.scan, base.scan)
        self.assertIs(updated.episode, base.episode)
        self.assertIsNot(updated.dynamics, base.dynamics)
        self.assertIs(updated.dynamics.timestep, base.dynamics.timestep)
        self.assertIs(updated.dynamics.steer_kp, base.dynamics.steer_kp)
        self.assertEqual(float(updated.dynamics.vehicle.mu), 0.75)
        self.assertAlmostEqual(
            float(updated.dynamics.vehicle.lr), 0.42, places=6
        )
        self.assertAlmostEqual(float(updated.body.centre_x), -0.30, places=6)
        self.assertAlmostEqual(
            float(base.dynamics.vehicle.mu), float(VEHICLE.mu), places=6
        )
        self.assertAlmostEqual(
            float(base.body.centre_x),
            -float(VEHICLE.lr) + float(VEHICLE.collision_body_center_x),
            places=6,
        )
        state = jnp.zeros((1, 7), dtype=jnp.float32)
        base_lidar = lidar_poses(
            state, base.scan, base.dynamics.vehicle.lr
        )
        updated_lidar = lidar_poses(
            state, updated.scan, updated.dynamics.vehicle.lr
        )
        self.assertAlmostEqual(
            float(base_lidar[0, 0]),
            float(base.scan.offset_x) - float(VEHICLE.lr),
            places=6,
        )
        self.assertAlmostEqual(
            float(updated_lidar[0, 0]),
            float(base.scan.offset_x) - 0.42,
            places=6,
        )

    def test_sample_core_params_returns_the_installed_vehicle(self):
        base = core_params()
        endpoint = NOMINAL_ARRAY.copy()
        endpoint[3] = 0.21
        endpoint[4] = 0.38
        endpoint[18] = 0.14
        fixed = active(endpoint)
        spec = VehicleRandomizationParams(
            nominal=active(),
            low=fixed,
            high=fixed,
            enabled=jnp.asarray(True),
        )
        updated, sampled = jax.jit(sample_core_params)(
            jax.random.key(33), base, spec
        )

        np.testing.assert_array_equal(sampled.as_array(), fixed.as_array())
        self.assertEqual(float(updated.dynamics.vehicle.lr), float(sampled.lr))
        self.assertAlmostEqual(
            float(updated.body.centre_x),
            -float(sampled.lr) + float(sampled.collision_body_center_x),
            places=6,
        )

    def test_sampling_jits_and_vmaps_as_a_pytree(self):
        base = core_params()
        spec = randomization_spec()
        keys = jax.random.split(jax.random.key(70), 5)
        run = jax.jit(
            jax.vmap(sample_core_params, in_axes=(0, None, None))
        )
        batched_core, batched_vehicle = run(keys, base, spec)

        self.assertEqual(batched_vehicle.as_array().shape, (5, ACTIVE_COUNT))
        self.assertEqual(batched_core.dynamics.vehicle.mu.shape, (5,))
        self.assertEqual(batched_core.body.centre_x.shape, (5,))
        expected_centres = (
            -batched_vehicle.lr + batched_vehicle.collision_body_center_x
        )
        np.testing.assert_allclose(batched_core.body.centre_x, expected_centres)

    def test_production_float32_dtypes_survive_sampling_and_replacement(self):
        updated, sampled = jax.jit(sample_core_params)(
            jax.random.key(17), core_params(), randomization_spec()
        )
        for leaf in jax.tree_util.tree_leaves(sampled):
            self.assertEqual(leaf.shape, ())
            self.assertEqual(leaf.dtype, jnp.float32)
        for tree in (updated.dynamics.vehicle, updated.body):
            for leaf in jax.tree_util.tree_leaves(tree):
                self.assertEqual(leaf.dtype, jnp.float32)


class TestModulePurity(unittest.TestCase):
    def test_module_imports_only_stdlib_jax_and_pure_siblings(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "f1tenth_gym"
            / "envs"
            / "dynamic_models"
            / "randomization.py"
        )
        tree = ast.parse(path.read_text())
        absolute = set()
        relative = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                absolute.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative.add(node.module)
                elif node.module:
                    absolute.add(node.module.split(".")[0])
        self.assertLessEqual(absolute, {"__future__", "dataclasses", "typing", "jax"})
        self.assertEqual(relative, {"jax", "jax_core", "contact.geometry"})


if __name__ == "__main__":
    unittest.main()
