import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from f110_gym.envs.base_classes import Integrator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from gym import spaces
from reward import *
import os


from sklearn.neighbors import KDTree

NUM_BEAMS = 300

def create_env(maps=[0]):
    env = gym.make('f110_gym:f110-v0', num_agents=1, maps=maps, num_beams = NUM_BEAMS, integrator=Integrator.RK4)
    
    env = FrenetObsWrapper(env=env)
    env = NewReward(env)
    env = ReducedObs(env)
    # env = NormalizeActionWrapper(env)    
    # env = VecNormalize(env)
    # env = DummyVecEnv([lambda: env])
    env = Monitor(env)
    return env

class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Normalize action space
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high

    def step(self, action):
        print('before action', action)
        
        action[0] = (self.high[0] - self.low[0]) * action[0] / 2
        action[1] = (self.high[1] - self.low[1]) * action[1] / 2

        # action = low + action * (high - low)
        print(self.low)
        print(self.high)
        print('after action ', action)
        
        # Call the original environment's step function
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = self.map_csv
        self.kdtree = KDTree(self.map_data[:, 1:3])
                
        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Box(low=0, high=self.num_agents - 1, shape=(1,), dtype=np.int32),
            'scans': spaces.Box(low=0, high=100, shape=(NUM_BEAMS, ), dtype=np.float32),
            'poses_x': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),      
            'poses_y': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),       
            'poses_theta': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(self.num_agents,), dtype=np.float32),       
            'linear_vels_x': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),     
            'linear_vels_y': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'ang_vels_z': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'collisions': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),   
            'lap_times': spaces.Box(low=0, high=1e6, shape=(self.num_agents,), dtype=np.float32), 
            'lap_counts': spaces.Box(low=0, high=9999, shape=(self.num_agents,), dtype=np.int32),
            'poses_s': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),      
            'poses_d': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),       
            'linear_vels_s': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),     
            'linear_vels_d': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        })

    def observation(self, obs):
        poses_x = obs['poses_x'][0]
        poses_y = obs['poses_y'][0]
        vel_magnitude = obs['linear_vels_x']
        poses_theta = obs['poses_theta'][0]
        
        frenet_coords = convert_to_frenet(poses_x, poses_y, vel_magnitude, poses_theta, self.map_data, self.kdtree)
        
        obs['poses_s'] = np.array(frenet_coords[0]).reshape((1, -1))
        obs['poses_d'] = np.array(frenet_coords[1])
        obs['linear_vels_s'] = np.array(frenet_coords[2]).reshape((1, -1))
        obs['linear_vels_d'] = np.array(frenet_coords[3])

        return obs
    
    
class ReducedObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ReducedObs, self).__init__(env)

        self.observation_space = spaces.Dict({
            'scans': spaces.Box(low=0, high=100, shape=(NUM_BEAMS, ), dtype=np.float32),
        })

    def observation(self, obs):        
        del obs['poses_x']
        del obs['poses_y']
        del obs['linear_vels_x']
        del obs['linear_vels_y']
        
        del obs['ego_idx']
        del obs['collisions']
        del obs['lap_times']
        del obs['lap_counts']
        del obs['ang_vels_z']
        del obs['poses_theta']
        
        del obs['poses_s']
        del obs['poses_d']
        del obs['linear_vels_s']
        del obs['linear_vels_d']

        return obs
    
    
# class TensorboardCallback(BaseCallback):
#     def __init__(self, save_interval, save_path, verbose=1):
#         super().__init__(verbose)
        
#         self.save_interval = save_interval
#         self.save_path = save_path
        
#         self.lap_counts = np.zeros(100, dtype=int)
#         self.first_lap_times = np.zeros(100, dtype=float)
#         self.episode_index = 0

#     def _on_step(self) -> bool:
#         if self.num_timesteps % self.save_interval == 0:
#             self.model.save(f"{self.save_path}_{self.num_timesteps}")
        
#         infos = self.locals.get("infos", [{}])[0]
#         checkpoint_done = infos.get("checkpoint_done", False)
        

#         if checkpoint_done:
#             self.lap_counts[self.episode_index] = infos.get("lap_count", 0)
#             self.episode_index = (self.episode_index + 1) % 10
            
#             success_rate = np.mean(self.lap_counts > 1)
#             self.logger.record("rollout/success_rate", success_rate)
            
#         # self.logger.record("rollout/poses_s", self.poses_s)
#         return True

class TensorboardCallback(BaseCallback):
    def __init__(self, save_interval, save_path, verbose=1):
        super().__init__(verbose)
        
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{self.num_timesteps}")

        return True
    