import unittest


class TestEnvInterface(unittest.TestCase):
    def test_gymnasium_api(self):
        import gymnasium as gymnasium
        import f110_gym
        from gymnasium.utils.env_checker import check_env

        env = gymnasium.make('f110_gym:f110-v0', num_agents=1)

        check_env(env.unwrapped)