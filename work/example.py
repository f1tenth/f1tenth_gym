import gym
import numpy as np
from stable_baselines3 import PPO
import stable_baselines3.common.env_checker as check_env
import warnings

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")

# instantiating the environment
env = gym.make('f110_gym:f110-v0', num_agents=1, map = '/Users/meraj/workspace/f1tenth_gym/gym/f110_gym/envs/maps/vegas')

obs = env.reset()

check_env.check_env(env)
exit()


# obs, step_reward, done, info = env.reset(poses=np.array([[0, 0, 0.1]]))
obs = env.reset()

model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

vec_env = model.get_env()
# obs, step_reward, done, info = env.reset(poses=np.array([[0, 0, 0.1]]))
obs = env.reset()

# simulation loop
lap_time = 0.

done = False
# loops when env not done
while not done:
    # get action based on the observation
    action = np.array([[0,0.1]])
    action, _state = model.predict(obs, deterministic=True)
    print('HI')
    print(action)
    print(type(action))
    print(action.shape)
    exit()
    # stepping through the environment
    obs, step_reward, done, info = env.step(action)
    env.render()
    # print('obs scans', obs['scans'])
    # print('scan shape', obs['scans'][0].shape)
    # print('reward', step_reward)
    # print('done', done)
    # print('info', info, '\n')
    lap_time += step_reward
