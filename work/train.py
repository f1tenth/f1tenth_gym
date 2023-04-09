import gym
from stable_baselines3 import PPO
import warnings
from f110_gym.envs.base_classes import Integrator
from utils import *
import logging

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

log_dir = "./metrics/"

def create_env():
    env = gym.make('f110_gym:f110-v0', num_agents=1, map='/Users/meraj/workspace/f1tenth_gym/examples/example_map', integrator=Integrator.RK4)
    env = FrenetObsWrapper(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = NewReward(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = FilterObservationSpace(env)
    env = ScaledObservationEnv(env)
    # env = FilterObservationSpace(env)
    return env

env = create_env()
# model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, max_grad_norm=0.5, clip_range=0.2, clip_range_vf=1.0, tensorboard_log="./metrics/")

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cuda')

save_interval = 50_000
save_path = "./models/ppo_model"

callback = SaveModelCallback(save_interval, save_path)

while True:
    model.learn(total_timesteps=10000_000, callback=callback, progress_bar=True)
    # model.learn(total_timesteps=10, callback=callback, progress_bar=True)
model.save("ppo_model")

env.env.close()
env.close()