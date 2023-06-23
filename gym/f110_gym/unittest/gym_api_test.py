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

        env = gym.make('f110_gym:f110-v0', num_agents=1)

        check_env(env.unwrapped)