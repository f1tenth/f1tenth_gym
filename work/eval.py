import gym
from stable_baselines3 import PPO
import warnings
from f110_gym.envs.base_classes import Integrator
from utils import *
import logging

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# instantiating the environment
def create_env():
    env = gym.make('f110_gym:f110-v0', num_agents=1, map='/Users/meraj/workspace/f1tenth_gym/examples/example_map', integrator=Integrator.RK4)
    env = FrenetObsWrapper(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = NewReward(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = ScaledObservationEnv(env)
    env = FilterObservationSpace(env)
    return env

env = create_env()

model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_150000.zip", env=env)

obs = env.reset()

done = False
# loops when env not done
while not done:
    # get action based on the observation
    # action, _state = model.predict(obs, deterministic=True)
    action, _state = model.predict(obs)
    # stepping through the environment
    obs, reward, done, info = env.step(action)
    env.render()
