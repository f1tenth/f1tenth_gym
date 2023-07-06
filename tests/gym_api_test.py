import pathlib
import unittest
from argparse import Namespace

import yaml

from f110_gym.envs import Integrator


class TestEnvInterface(unittest.TestCase):
    def test_gymnasium_api(self):
        import f110_gym
        from gymnasium.utils.env_checker import check_env
        import gymnasium as gym

        example_dir = pathlib.Path("examples")

        with open(example_dir / "config_example_map.yaml") as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        conf.map_path = str(example_dir / conf.map_path)

        env = gym.make(
            "f110_gym:f110-v0",
            map=conf.map_path,
            map_ext=conf.map_ext,
            num_agents=1,
            timestep=0.01,
            integrator=Integrator.Euler,
        )

        check_env(env.unwrapped)


if __name__ == "__main__":
    unittest.main()
