import unittest

import gymnasium as gym
import numpy as np
from f110_gym.envs.utils import deep_update


class TestEnvInterface(unittest.TestCase):
    def _make_env(self, config={}):

        conf = {
            "map": "Example",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": "speed",
            "params": {"mu": 1.0},
        }
        conf = deep_update(conf, config)

        env = gym.make(
            "f110_gym:f110-v0",
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
        base_env.configure(config=config_ext)

        # create an extended environment, with the width set on initialization
        extended_env = self._make_env(config=config_ext)

        # check consistency parameters in config
        for par in base_env.config["params"]:
            base_val = base_env.config["params"][par]
            extended_val = extended_env.config["params"][par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in simulator parameters
        for par in base_env.sim.params:
            base_val = base_env.sim.params[par]
            extended_val = extended_env.sim.params[par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in agent parameters
        for agent, ext_agent in zip(base_env.sim.agents, extended_env.sim.agents):
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

        params = base_env.sim.params.copy()
        new_v_max = 5.0
        params["v_max"] = new_v_max

        base_env.configure(config={"params": params})
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
        base_env = self._make_env(config={"control_input": "accl"})
        params = base_env.sim.params
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
