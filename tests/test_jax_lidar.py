"""Differential gates for the fixed-shape functional exact LiDAR path."""

import ast
from dataclasses import replace
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.lidar.config import LiDARConfig
from f1tenth_gym.envs.lidar import ray_cast
from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D
from f1tenth_gym.envs.simulator import F110Simulator
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.contact.geometry import BodyParams, body_vertices
from f1tenth_gym.envs.lidar.functional import (
    ScanConfig,
    ScanParams,
    beam_angles,
    clean_scan,
    lidar_poses,
    opponent_ranges,
)
from f1tenth_gym.envs.track.preprocessing import preprocess_track
from f1tenth_gym.envs.track.functional import TileTable, WallTable


HALF_FOV = 2.3561945
MAX_RANGE = 30.0


def model_states(poses, state_dim=7):
    """Place world poses into supported native-state columns."""
    poses = np.asarray(poses, dtype=np.float32)
    states = np.zeros((len(poses), state_dim), dtype=np.float32)
    states[:, 0] = poses[:, 0]
    states[:, 1] = poses[:, 1]
    states[:, 4] = poses[:, 2]
    return jnp.asarray(states)


def scan_config(num_agents, num_beams=120, angle_min=-HALF_FOV,
                angle_max=HALF_FOV):
    return ScanConfig(num_agents, num_beams, angle_min, angle_max)


class TestRigidTransforms(unittest.TestCase):
    def test_lidar_mount_is_resolved_from_rear_axle_base_link(self):
        vehicle = F1TENTH_VEHICLE_PARAMETERS.with_updates(lr=0.42)
        poses = np.array(
            [[1.2, -3.4, 0.8], [-0.2, 4.1, -2.7]], dtype=np.float32
        )
        for transform in ((0.0, 0.0, 0.0), (0.31, -0.07, 0.23)):
            with self.subTest(transform=transform):
                lidar = LiDARConfig(
                    num_beams=4,
                    base_link_to_lidar_tf=transform,
                    noise_std=0.0,
                )
                params = ScanParams.from_lidar_config(lidar)
                dx = transform[0] - vehicle.lr
                expected = poses.copy()
                cosine = np.cos(poses[:, 2])
                sine = np.sin(poses[:, 2])
                expected[:, 0] += dx * cosine - transform[1] * sine
                expected[:, 1] += dx * sine + transform[1] * cosine
                expected[:, 2] += transform[2]

                functional = np.asarray(
                    lidar_poses(model_states(poses), params, vehicle.lr)
                )
                fake = type("SimulatorConfig", (), {})()
                fake.config = type("EnvConfig", (), {"lidar_config": lidar})()
                fake.vehicle_params = vehicle
                mutable = np.stack(
                    [
                        F110Simulator._lidar_pose_from_cog(fake, pose)
                        for pose in poses
                    ]
                )
                np.testing.assert_allclose(functional, expected, atol=1.0e-6)
                np.testing.assert_allclose(mutable, expected, atol=1.0e-6)

    def test_lidar_mount_accepts_traced_lr_without_recompilation(self):
        params = ScanParams.from_lidar_config(
            LiDARConfig(num_beams=4, base_link_to_lidar_tf=(0.3, 0.0, 0.0))
        )
        state = model_states([[1.0, 2.0, 0.0]])
        run = jax.jit(lambda lr: lidar_poses(state, params, lr))

        first = np.asarray(run(jnp.float32(0.1)))
        second = np.asarray(run(jnp.float32(0.25)))
        self.assertAlmostEqual(float(first[0, 0]), 1.2, places=6)
        self.assertAlmostEqual(float(second[0, 0]), 1.05, places=6)

    def test_body_vertices_match_the_host_with_the_cog_offset(self):
        vehicle = F1TENTH_VEHICLE_PARAMETERS.with_updates(
            collision_body_center_y=0.04
        )
        body = BodyParams.from_vehicle_parameters(vehicle)
        pose = np.array([2.3, -1.7, 1.1], dtype=np.float32)
        got = np.asarray(body_vertices(jnp.asarray(pose), body))
        offset_pose = pose.copy()
        dx = -vehicle.lr + vehicle.collision_body_center_x
        dy = vehicle.collision_body_center_y
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        offset_pose[0] += dx * cosine - dy * sine
        offset_pose[1] += dx * sine + dy * cosine
        expected = get_vertices(offset_pose.astype(np.float64), vehicle.length,
                                vehicle.width)
        np.testing.assert_allclose(got, expected, atol=2.0e-7)


class TestFunctionalWallScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.vehicle = F1TENTH_VEHICLE_PARAMETERS
        cls.body = BodyParams.from_vehicle_parameters(cls.vehicle)
        cls.lidar = LiDARConfig(num_beams=120, noise_std=0.0)
        cls.config = scan_config(1, cls.lidar.num_beams,
                                 cls.lidar.angle_min, cls.lidar.angle_max)
        cls.table = preprocess_track(
            cls.track, cls.vehicle, ray_max_range=cls.lidar.range_max
        )
        cls.params = ScanParams.from_lidar_config(cls.lidar)
        cls.host = SegmentScanSimulator2D(
            cls.lidar.num_beams,
            cls.lidar.field_of_view,
            angle_min=cls.lidar.angle_min,
            angle_max=cls.lidar.angle_max,
            std_dev=0.0,
            min_range=cls.lidar.range_min,
            max_range=cls.lidar.range_max,
        )
        cls.host.set_map(cls.track)

    def test_wall_ranges_match_the_live_segment_scanner(self):
        indexes = (25, 130, 310, 505)
        poses = np.stack(
            [
                self.track.raceline.xs[list(indexes)],
                self.track.raceline.ys[list(indexes)],
                self.track.raceline.yaws[list(indexes)],
            ],
            axis=1,
        ).astype(np.float32)
        fake = type("SimulatorConfig", (), {})()
        fake.config = type("EnvConfig", (), {"lidar_config": self.lidar})()
        fake.vehicle_params = self.vehicle
        run = jax.jit(
            lambda state: clean_scan(
                state, self.table, self.body, self.config, self.params,
                self.vehicle.lr,
            )
        )
        for pose in poses:
            got = np.asarray(run(model_states([pose]))[0])
            sensor_pose = F110Simulator._lidar_pose_from_cog(fake, pose)
            expected = self.host.scan(sensor_pose, rng=None)
            np.testing.assert_allclose(got, expected, atol=2.0e-3)

    def test_an_empty_map_returns_max_range(self):
        blank = Track.from_track_name("Spielberg_blank", 1.0)
        table = preprocess_track(blank, self.vehicle, ray_max_range=MAX_RANGE)
        params = ScanParams.from_lidar_config(self.lidar)
        got = clean_scan(
            model_states([[0.0, 0.0, 0.0]]),
            table,
            self.body,
            self.config,
            params,
            self.vehicle.lr,
        )
        np.testing.assert_array_equal(
            np.asarray(got), np.full(got.shape, MAX_RANGE, dtype=np.float32)
        )

    def test_a_masked_candidate_cannot_alias_real_wall_zero(self):
        wall = WallTable(
            a=jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            b=jnp.array([[1.0, 1.0]], dtype=jnp.float32),
            normals=jnp.array([[-1.0, 0.0]], dtype=jnp.float32),
            adjacency=jnp.zeros((1, 2), dtype=jnp.int32),
            adjacency_mask=jnp.zeros((1, 2), dtype=jnp.bool_),
            lengths=jnp.array([2.0], dtype=jnp.float32),
            mask=jnp.ones((1,), dtype=jnp.bool_),
        )
        tiles = TileTable(
            indices=jnp.zeros((1, 1, 1), dtype=jnp.int32),
            mask=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
            origin=jnp.array([-10.0, -10.0], dtype=jnp.float32),
            tile_size=jnp.asarray(20.0, dtype=jnp.float32),
            reach=jnp.asarray(MAX_RANGE, dtype=jnp.float32),
        )
        table = replace(self.table, walls=wall, ray_tiles=tiles)
        config = scan_config(1, 1, 0.0, 1.0)
        params = replace(self.params, offset_x=0.0, offset_y=0.0,
                         offset_yaw=0.0)
        got = clean_scan(
            model_states([[0.0, 0.0, 0.0]]), table, self.body, config, params,
            self.vehicle.lr,
        )
        self.assertEqual(float(got[0, 0]), MAX_RANGE)

    def test_one_beam_uses_angle_min(self):
        config = scan_config(1, 1, -0.73, 1.8)
        np.testing.assert_array_equal(
            np.asarray(beam_angles(config)), np.asarray([-0.73], dtype=np.float32)
        )


class TestOpponentOcclusion(unittest.TestCase):
    def test_base_link_mount_avoids_a_false_min_range_arc_at_an_opponent(self):
        blank = Track.from_track_name("Spielberg_blank", 1.0)
        vehicle = F1TENTH_VEHICLE_PARAMETERS
        lidar = LiDARConfig(
            num_beams=3,
            angle_min=-0.1,
            angle_max=0.1,
            range_min=0.1,
            range_max=10.0,
            noise_std=0.0,
        )
        config = EnvConfig(
            map_name=blank,
            num_agents=2,
            params=vehicle,
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.ST,
                compute_frenet_frame=False,
                max_laps=None,
            ),
            lidar_config=lidar,
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        poses = np.array(
            [[0.0, 0.0, 0.0], [0.59, 0.0, 0.0]], dtype=np.float32
        )

        env = F110Env(config)
        try:
            observation, _ = env.reset(seed=3, options={"poses": poses})
            mutable_range = float(observation["agent_0"]["scan"][1])
        finally:
            env.close()

        table = preprocess_track(blank, vehicle, ray_max_range=lidar.range_max)
        functional = clean_scan(
            model_states(poses),
            table,
            BodyParams.from_vehicle_parameters(vehicle),
            scan_config(2, 3, lidar.angle_min, lidar.angle_max),
            ScanParams.from_lidar_config(lidar),
            vehicle.lr,
        )
        functional_range = float(functional[0, 1])

        # The cars have a 1 cm bumper gap. The correct origin is 0.10355 m
        # ahead of ego's CoG, putting the opponent face 0.1901 m away. Applying
        # the 0.275 m mount directly to the CoG instead yields 0.01865 m, which
        # the observed scan turns into a misleading range_min arc.
        self.assertGreater(mutable_range, lidar.range_min)
        self.assertAlmostEqual(mutable_range, 0.1901, places=5)
        self.assertAlmostEqual(functional_range, 0.1901, places=5)

    def test_random_opponents_match_the_current_brute_force_result(self):
        rng = np.random.default_rng(9)
        pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self_vertices = get_vertices(pose.astype(np.float64), 0.58, 0.31)
        for beams, field_of_view in ((181, np.pi), (241, 2.0 * np.pi)):
            config = scan_config(2, beams, -field_of_view / 2, field_of_view / 2)
            angles = np.asarray(beam_angles(config))
            run = jax.jit(
                lambda vertices: opponent_ranges(
                    jnp.asarray(pose), vertices, jnp.int32(0),
                    beam_angles(config), jnp.float32(MAX_RANGE)
                )
            )
            for _ in range(24):
                while True:
                    opponent_pose = np.array(
                        [rng.uniform(-5, 5), rng.uniform(-5, 5),
                         rng.uniform(-np.pi, np.pi)]
                    )
                    if np.linalg.norm(opponent_pose[:2]) > 0.4:
                        break
                other = get_vertices(opponent_pose, 0.58, 0.31)
                vertices = jnp.asarray(np.stack((self_vertices, other)), jnp.float32)
                got = np.asarray(run(vertices))
                expected = ray_cast(
                    pose.astype(np.float64),
                    np.full(beams, MAX_RANGE),
                    angles.astype(np.float64),
                    other,
                )
                np.testing.assert_allclose(got, expected, atol=2.0e-5)

    def test_multiple_opponents_are_pair_order_invariant(self):
        config = scan_config(3, 256)
        angles = beam_angles(config)
        pose = jnp.zeros(3, dtype=jnp.float32)
        bodies = np.stack(
            (
                get_vertices(np.array([0.0, 0.0, 0.0]), 0.58, 0.31),
                get_vertices(np.array([2.0, 0.5, 0.3]), 0.58, 0.31),
                get_vertices(np.array([1.2, -1.4, -0.6]), 0.58, 0.31),
            )
        )
        first = opponent_ranges(
            pose, jnp.asarray(bodies), jnp.int32(0), angles, MAX_RANGE
        )
        second = opponent_ranges(
            pose, jnp.asarray(bodies[[0, 2, 1]]), jnp.int32(0), angles, MAX_RANGE
        )
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_a_body_behind_a_partial_sweep_does_not_occlude(self):
        config = scan_config(2, 181, -np.pi / 2, np.pi / 2)
        bodies = jnp.asarray(
            np.stack(
                (
                    get_vertices(np.array([0.0, 0.0, 0.0]), 0.58, 0.31),
                    get_vertices(np.array([-2.0, 0.0, 0.0]), 0.58, 0.31),
                )
            ),
            jnp.float32,
        )
        got = opponent_ranges(
            jnp.zeros(3), bodies, jnp.int32(0), beam_angles(config), MAX_RANGE
        )
        np.testing.assert_array_equal(
            np.asarray(got), np.full(got.shape, MAX_RANGE, dtype=np.float32)
        )


class TestTransformability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.vehicle = F1TENTH_VEHICLE_PARAMETERS
        cls.table = preprocess_track(
            cls.track, cls.vehicle, ray_max_range=MAX_RANGE
        )
        cls.body = BodyParams.from_vehicle_parameters(cls.vehicle)
        cls.config = scan_config(1, 48)
        cls.params = ScanParams.from_lidar_config(
            LiDARConfig(num_beams=48, noise_std=0.0)
        )
        cls.pose = np.array(
            [cls.track.raceline.xs[200], cls.track.raceline.ys[200],
             cls.track.raceline.yaws[200]],
            dtype=np.float32,
        )

    def test_clean_scan_jits_evaluates_shapes_and_has_a_finite_pose_gradient(self):
        run = jax.jit(
            lambda state: clean_scan(
                state, self.table, self.body, self.config, self.params,
                self.vehicle.lr,
            )
        )
        state = model_states([self.pose])
        result = run(state)
        shaped = jax.eval_shape(run, state)
        self.assertEqual(result.shape, (1, 48))
        self.assertEqual(shaped.shape, result.shape)
        gradient = jax.grad(lambda value: jnp.sum(run(value)))(state)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_environment_vmap_accepts_different_traced_sensor_and_body_values(self):
        state = model_states([self.pose])
        states = jnp.stack((state, state))
        bodies = jax.tree.map(
            lambda value: jnp.asarray([value, value * 1.02]), self.body
        )
        params = jax.tree.map(
            lambda value: jnp.asarray([value, value]), self.params
        )
        params = replace(
            params,
            offset_x=jnp.asarray([self.params.offset_x,
                                  self.params.offset_x + 0.08]),
        )
        run = jax.jit(
            jax.vmap(
                lambda one_state, one_body, one_params: clean_scan(
                    one_state, self.table, one_body, self.config, one_params,
                    self.vehicle.lr,
                )
            )
        )
        got = run(states, bodies, params)
        self.assertEqual(got.shape, (2, 1, 48))
        self.assertFalse(np.allclose(np.asarray(got[0]), np.asarray(got[1])))

    def test_functional_scan_modules_do_not_import_numpy_gym_or_callbacks(self):
        envs = pathlib.Path(__file__).resolve().parents[1] / "f1tenth_gym" / "envs"
        for path in (
            envs / "contact" / "geometry.py",
            envs / "lidar" / "functional.py",
            envs / "lidar" / "kernels.py",
        ):
            source = path.read_text()
            tree = ast.parse(source)
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(("." * node.level) + (node.module or ""))
            self.assertFalse(any(value.startswith("numpy") for value in imported))
            self.assertNotIn("gymnasium", source)
            self.assertNotIn("pure_callback", source)


class TestValidation(unittest.TestCase):
    def test_static_shape_errors_are_named(self):
        for args in ((0, 4, -1.0, 1.0), (1, 0, -1.0, 1.0),
                     (1, 4, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                ScanConfig(*args)
        config = scan_config(2, 4)
        vehicle = F1TENTH_VEHICLE_PARAMETERS
        track = preprocess_track(
            Track.from_track_name("Spielberg", 1.0), vehicle,
            ray_max_range=MAX_RANGE,
        )
        body = BodyParams.from_vehicle_parameters(vehicle)
        params = ScanParams(MAX_RANGE, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "model_state"):
            clean_scan(
                jnp.zeros((1, 7)), track, body, config, params, vehicle.lr
            )


if __name__ == "__main__":
    unittest.main()
