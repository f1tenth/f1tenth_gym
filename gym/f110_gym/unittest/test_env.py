import unittest
from f110_gym.envs import F110Env
import numpy as np
import pickle

class EnvPickleTests(unittest.TestCase):
    def test_renderer_deletion(self):
        env = F110Env(map='skirk')
        env.reset(poses=np.zeros((2, 3)))
        env.render()
        self.assertTrue(env.renderer is not None)
        pkl = pickle.dumps(env)
        env = pickle.loads(pkl)
        self.assertTrue(env.renderer is None)


if __name__ == '__main__':
    unittest.main()
