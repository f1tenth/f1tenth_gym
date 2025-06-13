import unittest

import gymnasium as gym
import numpy as np
from f1tenth_gym.envs.utils import deep_update


class TestEnvInterface(unittest.TestCase):
    def _make_env(self, config={}):
        conf = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "params": {"mu": 1.0},
        }
        conf = deep_update(conf, config)

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=conf,
        )
        return env

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        env = self._make_env()
        check_env(env.unwrapped, skip_render_check=True)

    def test_configure_method(self):
        """
        Test that the configure method works as expected, and that the parameters are
        correctly updated in the simulator and agents.
        """

        # create a base environment and use configure() to change the width
        config_ext = {"params": {"width": 15.0}}
        base_env = self._make_env()
        base_env.unwrapped.configure(config=config_ext)

        # create an extended environment, with the width set on initialization
        extended_env = self._make_env(config=config_ext)

        # check consistency parameters in config
        for par in base_env.unwrapped.config["params"]:
            base_val = base_env.unwrapped.config["params"][par]
            extended_val = extended_env.unwrapped.config["params"][par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in simulator parameters
        for par in base_env.unwrapped.sim.params:
            base_val = base_env.unwrapped.sim.params[par]
            extended_val = extended_env.unwrapped.sim.params[par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in agent parameters
        for agent, ext_agent in zip(base_env.unwrapped.sim.agents, extended_env.unwrapped.sim.agents):
            for par in agent.params:
                base_val = agent.params[par]
                extended_val = ext_agent.params[par]

                self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # finally, run a simulation and check that the results are the same
        obs0, _ = base_env.reset(options={"poses": np.array([[0.0, 0.0, np.pi / 2]])})
        obs1, _ = extended_env.reset(
            options={"poses": np.array([[0.0, 0.0, np.pi / 2]])}
        )
        done0 = done1 = False
        t = 0

        while not done0 and not done1:
            action = base_env.action_space.sample()
            obs0, _, done0, _, _ = base_env.step(action)
            obs1, _, done1, _, _ = extended_env.step(action)
            base_env.render()
            for k in obs0:
                self.assertTrue(
                    np.allclose(obs0[k], obs1[k]),
                    f"Observations {k} should be the same",
                )
            self.assertTrue(done0 == done1, "Done should be the same")
            t += 1

        print(f"Done after {t} steps")

        base_env.close()
        extended_env.close()

    def test_configure_action_space(self):
        """
        Try to change the upper bound of the action space, and check that the
        action space is correctly updated.
        """
        base_env = self._make_env()
        action_space_low = base_env.action_space.low
        action_space_high = base_env.action_space.high

        params = base_env.unwrapped.sim.params.copy()
        new_v_max = 5.0
        params["v_max"] = new_v_max

        base_env.unwrapped.configure(config={"params": params})
        new_action_space_low = base_env.action_space.low
        new_action_space_high = base_env.action_space.high

        self.assertTrue(
            (action_space_low == new_action_space_low).all(),
            "Steering action space should be the same",
        )
        self.assertTrue(
            action_space_high[0][0] == new_action_space_high[0][0],
            "Steering action space should be the same",
        )
        self.assertTrue(
            new_action_space_high[0][1] == new_v_max,
            f"Speed action high should be {new_v_max}",
        )

    def test_acceleration_action_space(self):
        """
        Test that the acceleration action space is correctly configured.
        """
        base_env = self._make_env(config={"control_input": ["accl", "steering_speed"]})
        params = base_env.unwrapped.sim.params
        action_space_low = base_env.action_space.low
        action_space_high = base_env.action_space.high

        self.assertTrue(
            (action_space_low[0][0] - params["sv_min"]) < 1e-6,
            "lower sv does not match min steering velocity",
        )
        self.assertTrue(
            (action_space_high[0][0] - params["sv_max"]) < 1e-6,
            "upper sv does not match max steering velocity",
        )
        self.assertTrue(
            (action_space_low[0][1] + params["a_max"]) < 1e-6,
            "lower acceleration bound does not match a_min",
        )
        self.assertTrue(
            (action_space_high[0][1] - params["a_max"]) < 1e-6,
            "upper acceleration bound does not match a_max",
        )

    def test_manual_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
        }
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", asynchronous=False, config=config, num_envs=num_envs
        )

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )

    def test_manual_reset_options_in_asynch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
        }
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", vectorization_mode="async", config=config, num_envs=num_envs
        )

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )

    def test_auto_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment without explicit poses.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
            "reset_config": {"type": "rl_random_random"},
        }
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", vectorization_mode="sync", config=config, num_envs=num_envs,
        )

        obss, infos = vec_env.reset()

        for i, agent_id in enumerate(obss):
            agent_pose0 = np.array(
                [
                    obss[agent_id]["pose_x"][0],
                    obss[agent_id]["pose_y"][0],
                    obss[agent_id]["pose_theta"][0],
                ]
            )
            for ie in range(1, num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertFalse(
                    np.allclose(agent_pose, agent_pose0),
                    f"pose of agent {agent_id} in env {ie} should be random, got same {agent_pose} == {agent_pose0}",
                )

        # test auto reset
        all_dones_once = [False] * num_envs
        all_dones_twice = [False] * num_envs

        max_steps = 1000
        while not all(all_dones_twice) and max_steps > 0:
            actions = vec_env.action_space.sample()
            obss, rewards, dones, truncations, infos = vec_env.step(actions)

            all_dones_once = [all_dones_once[i] or dones[i] for i in range(num_envs)]
            all_dones_twice = [
                all_dones_twice[i] or all_dones_once[i] for i in range(num_envs)
            ]
            max_steps -= 1

        vec_env.close()
        self.assertTrue(
            all(all_dones_twice),
            f"All envs should be done twice, got {all_dones_twice}",
        )
