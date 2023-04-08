import gym
from stable_baselines3 import PPO
import warnings
from f110_gym.envs.base_classes import Integrator
# from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import *

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")

log_dir = "./metrics/"

def create_env():
    env = gym.make('f110_gym:f110-v0', num_agents=1, map='/Users/meraj/workspace/f1tenth_gym/examples/example_map', integrator=Integrator.RK4)
    env = FrenetObsWrapper(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = NewReward(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = ScaledObservationEnv(env)
    env = FilterObservationSpace(env)
    return env

# num_envs = 1

env = create_env()
# env = SubprocVecEnv([lambda: create_env() for _ in range(num_envs)])
# model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, max_grad_norm=0.5, clip_range=0.2, clip_range_vf=1.0, tensorboard_log="./metrics/")

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

save_interval = 30_000
save_path = "ppo_model"

callback = SaveModelCallback(save_interval, save_path)

model.learn(total_timesteps=1000_000, callback=callback, progress_bar=True)
model.save("ppo_model")

env.env.close()
env.close()