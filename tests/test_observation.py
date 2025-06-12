import unittest

import gymnasium as gym
import numpy as np
from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.observation import observation_factory
from f1tenth_gym.envs.utils import deep_update
from gymnasium.spaces import Box


class TestObservationInterface(unittest.TestCase):
    @staticmethod
    def _make_env(config={}) -> F110Env:
        conf = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "params": {"mu": 1.0},
        }
        conf = deep_update(conf, config)

        env = gym.make("f1tenth_gym:f1tenth-v0", config=conf)
        return env

    def test_original_obs_space(self):
        """
        Check backward compatibility with the original observation space.
        """
        env = self._make_env(config={"observation_config": {"type": "original"}})

        obs, _ = env.reset()

        obs_keys = [
            "ego_idx",
            "scans",
            "poses_x",
            "poses_y",
            "poses_theta",
            "linear_vels_x",
            "linear_vels_y",
            "ang_vels_z",
            "collisions",
            "lap_times",
            "lap_counts",
        ]

        # check that the observation space has the correct types
        self.assertTrue(
            all(
                [
                    isinstance(env.observation_space.spaces[k], Box)
                    for k in obs_keys
                    if k != "ego_idx"
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    env.observation_space.spaces[k].dtype == np.float32
                    for k in obs_keys
                    if k != "ego_idx"
                ]
            )
        )

        # check the observation space is a dict
        self.assertTrue(isinstance(obs, dict))

        # check that the observation has the correct keys
        self.assertTrue(all([k in obs for k in obs_keys]))
        self.assertTrue(all([k in obs_keys for k in obs]))
        self.assertTrue(env.observation_space.contains(obs))

    def test_features_observation(self):
        """
        Check the FeatureObservation allows to select an arbitrary subset of features.
        """
        features = ["pose_x", "pose_y", "pose_theta"]

        env = self._make_env(
            config={"observation_config": {"type": "features", "features": features}}
        )

        # check the observation space is a dict
        self.assertTrue(isinstance(env.observation_space, gym.spaces.Dict))

        # check that the observation space has the correct keys
        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all([k in space for k in features]))
            self.assertTrue(all([k in features for k in space]))

        # check that the observation space has the correct types
        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all([isinstance(space[k], Box) for k in features]))
            self.assertTrue(all([space[k].dtype == np.float32 for k in features]))

        # check the actual observation
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            pose_x, pose_y, pose_theta = env.unwrapped.sim.agent_poses[i]
            obs_x, obs_y, obs_theta = (
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
            )

            for ground_truth, observation in zip(
                [pose_x, pose_y, pose_theta], [obs_x, obs_y, obs_theta]
            ):
                self.assertTrue(np.allclose(ground_truth, observation))

    def test_unexisting_obs_space(self):
        """
        Check that an error is raised when an unexisting observation type is requested.
        """
        env = self._make_env()
        with self.assertRaises(ValueError):
            observation_factory(env, vehicle_id=0, type="unexisting_obs_type")

    def test_kinematic_obs_space(self):
        """
        Check the kinematic state observation space contains the correct features [x, y, theta, v].
        """
        env = self._make_env(config={"observation_config": {"type": "kinematic_state"}})

        kinematic_features = ["pose_x", "pose_y", "pose_theta", "linear_vel_x", "delta"]

        # check kinematic features are in the observation space
        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all([k in space for k in kinematic_features]))
            self.assertTrue(all([k in kinematic_features for k in space]))

        # check the actual observation
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            pose_x, pose_y, _, velx, pose_theta, _, _ = env.unwrapped.sim.agents[i].state
            obs_x, obs_y, obs_theta = (
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
            )
            obs_velx = obs[agent_id]["linear_vel_x"]

            for ground_truth, observed in zip(
                [pose_x, pose_y, pose_theta, velx], [obs_x, obs_y, obs_theta, obs_velx]
            ):
                self.assertTrue(np.allclose(ground_truth, observed))

    def test_dynamic_obs_space(self):
        """
        Check the dynamic state observation space contains the correct features.
        """
        env = self._make_env(config={"observation_config": {"type": "original"}})

        kinematic_features = [
            "pose_x",
            "pose_y",
            "pose_theta",
            "linear_vel_x",
            "ang_vel_z",
            "delta",
            "beta",
        ]

        # check kinematic features are in the observation space
        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all([k in space for k in kinematic_features]))
            self.assertTrue(all([k in kinematic_features for k in space]))

        # check the actual observation
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            pose_x, pose_y, delta, velx, pose_theta, _, beta = env.unwrapped.sim.agents[i].state

            agent_obs = obs[agent_id]
            obs_x, obs_y, obs_theta = (
                agent_obs["pose_x"],
                agent_obs["pose_y"],
                agent_obs["pose_theta"],
            )
            obs_velx, obs_delta, obs_beta = (
                agent_obs["linear_vel_x"],
                agent_obs["delta"],
                agent_obs["beta"],
            )

            for ground_truth, observed in zip(
                [pose_x, pose_y, pose_theta, velx, delta, beta],
                [obs_x, obs_y, obs_theta, obs_velx, obs_delta, obs_beta],
            ):
                self.assertTrue(np.allclose(ground_truth, observed))

    def test_consistency_observe_space(self):
        obs_type_ids = ["direct", "original"]

        env = self._make_env()
        env.reset()

        for obs_type_id in obs_type_ids:
            obs_type = observation_factory(env, type=obs_type_id)
            space = obs_type.space()
            observation = obs_type.observe()

            self.assertTrue(
                space.contains(observation),
                f"Observation {obs_type_id} is not contained in its space",
            )

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        obs_type_ids = ["direct", "original"]

        for obs_type_id in obs_type_ids:
            env = self._make_env(config={"observation_config": {"type": obs_type_id}})
            check_env(
                env.unwrapped,
                skip_render_check=True,
            )
