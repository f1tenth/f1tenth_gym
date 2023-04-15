import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from f110_gym.envs.base_classes import Integrator
from gym import spaces
from reward import *
import os


from sklearn.neighbors import KDTree

NUM_BEAMS = 600

def create_env(maps=[0]):
    cwd = os.getcwd()
    # map_name = cwd + '/tracks/maps/{}'.format(map)
    
    env = gym.make('f110_gym:f110-v0', num_agents=1, maps=maps, num_beams = 600, integrator=Integrator.RK4)
    env = FrenetObsWrapper(env=env)
    env = NewReward(env)
    env = ReducedObs(env)
    # env.reset()
    return env

class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = self.map_csv
        self.kdtree = KDTree(self.map_data[:, 1:3])
                
        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Box(low=0, high=self.num_agents - 1, shape=(1,), dtype=np.int32),
            'scans': spaces.Box(low=0, high=100, shape=(self.num_beams, ), dtype=np.float32),
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
    
    
class TensorboardCallback(BaseCallback):
    def __init__(self, save_interval, save_path, verbose=1):
        super().__init__(verbose)
        
        self.save_interval = save_interval
        self.save_path = save_path
        self.poses_s = 0
        self.map_data = read_csv(self.training_env.get_attr('map_csv'))
        self.kdtree = KDTree(self.map_data[:, 1:3])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{self.num_timesteps}")
        
        infos = self.locals.get("infos", [{}])[0]['checkpoint_done']
        if infos == True:
            obs = self.training_env.get_attr("curr_obs")[0]
            poses_x = obs["poses_x"][0]
            poses_y = obs["poses_y"][0]

            self.poses_s = convert_to_frenet(x= poses_x, y = poses_y, vel_magnitude = 0, pose_theta = 0, map_data=self.map_data, kdtree=self.kdtree)[0]
            
        if self.poses_s > 150:
            self.poses_s -= 156.3585883 

        self.logger.record("rollout/poses_s", self.poses_s)
        return True
    