import unittest


class TestUtilities(unittest.TestCase):
    def test_deep_update(self):
        """
        Test that the deep_update function works as expected with nested dictionaries,
        by comparing two environments with different mu values.
        """
        import gymnasium as gym

        default_env = gym.make("f1tenth_gym:f1tenth-v0")
        custom_env = gym.make("f1tenth_gym:f1tenth-v0", config={"params": {"mu": 1.0}})

        # check all parameters are the same except for mu
        for par in default_env.unwrapped.sim.params:
            default_val = default_env.unwrapped.sim.params[par]
            custom_val = custom_env.unwrapped.sim.params[par]

            if par == "mu":
                self.assertNotEqual(default_val, custom_val, "mu should be different")
            else:
                self.assertEqual(default_val, custom_val, f"{par} should be the same")

        default_env.close()
        custom_env.close()
