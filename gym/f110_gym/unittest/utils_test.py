import pathlib
import unittest
from argparse import Namespace

import numpy as np
import yaml


class TestUtilities(unittest.TestCase):
    def test_deep_update(self):
        import f110_gym
        import gymnasium as gym

        default_env = gym.make("f110_gym:f110-v0")
        custom_env = gym.make("f110_gym:f110-v0", config={"params": {"mu": 1.0}})

        # check all parameters are the same except for mu
        for par in default_env.sim.params:
            default_val = default_env.sim.params[par]
            custom_val = custom_env.sim.params[par]

            if par == "mu":
                self.assertNotEqual(default_val, custom_val, "mu should be different")
            else:
                self.assertEqual(default_val, custom_val, f"{par} should be the same")

        default_env.close()
        custom_env.close()

    def test_configure_method(self):
        import f110_gym
        import gymnasium as gym

        example_dir = pathlib.Path(__file__).parent.parent.parent.parent / "examples"

        with open(example_dir / "config_example_map.yaml") as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        conf.map_path = str(example_dir / conf.map_path)

        config = {
            "map": conf.map_path,
            "map_ext": conf.map_ext,
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": "speed",
        }

        extended_config = config.copy()
        extended_config["params"] = {"width": 15.0}

        base_env = gym.make(
            "f110_gym:f110-v0",
            config=config,
            render_mode="human",
        )
        base_env.configure(config={"params": {"width": 15.0}})

        extended_env = gym.make(
            "f110_gym:f110-v0",
            config=extended_config,
            render_mode="human",
        )

        # check all parameters in config dictionary are the same
        for par in base_env.sim.params:
            base_val = base_env.config["params"][par]
            extended_val = extended_env.config["params"][par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check all params in simulator are the same
        for par in base_env.sim.params:
            base_val = base_env.sim.params[par]
            extended_val = extended_env.sim.params[par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check all params in agents are the same
        for agent, ext_agent in zip(base_env.sim.agents, extended_env.sim.agents):
            for par in agent.params:
                base_val = agent.params[par]
                extended_val = ext_agent.params[par]

                self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # run a simulation and check that the observations are the same
        obs0, _ = base_env.reset()
        obs1, _ = extended_env.reset()
        done0 = done1 = False
        t = 0

        while not done0 and not done1:
            action = base_env.action_space.sample()
            obs0, _, done0, _, _ = base_env.step(action)
            obs1, _, done1, _, _ = extended_env.step(action)
            base_env.render()
            for k in obs0:
                self.assertTrue(np.allclose(obs0[k], obs1[k]), f"Observations {k} should be the same")
            self.assertTrue(done0 == done1, "Done should be the same")
            t += 1

        print(f"Done after {t} steps")

        base_env.close()
        extended_env.close()
