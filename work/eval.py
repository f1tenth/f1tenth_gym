import gym
from stable_baselines3 import PPO
import warnings
from f110_gym.envs.base_classes import Integrator
from utils import *
import logging
import time

import cProfile
import pstats
from io import StringIO

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

env = create_env()

model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_2400000.zip", env=env)

obs = env.reset()

done = False
# loops when env not done
def trying():
    
    env = create_env()

    model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_500000.zip", env=env)

    obs = env.reset()

    done = False

    while not done:
        # action, _state = model.predict(obs, deterministic=True)
        action, _state = model.predict(obs)
        # stepping through the environment
        obs, reward, done, info = env.step(action)
        env.render(mode='human_fast')

trying()

# pr = cProfile.Profile()
# pr.enable()
# trying()  # Replace this with your function
# pr.disable()

# s = StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()

# print(s.getvalue())