import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from f110_gym.envs.base_classes import Integrator
from gym import spaces
from reward import *


NUM_BEAMS = 600

def create_env():
    env = gym.make('f110_gym:f110-v0', num_agents=1, map='/Users/meraj/workspace/f1tenth_gym/examples/example_map', integrator=Integrator.RK4)
    env = FrenetObsWrapper(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = NewReward(env, '/Users/meraj/workspace/f1tenth_gym/examples/example_waypoints.csv')
    env = FilterObservationSpace(env)
    # env = ScaledObservationEnv(env)
    return env



# class SaveModelCallback(BaseCallback):
#     def __init__(self, save_interval, save_path):
#         super().__init__()
#         self.save_interval = save_interval
#         self.save_path = save_path

#     def _on_step(self) -> bool:
#         if self.num_timesteps % self.save_interval == 0:
#             self.model.save(f"{self.save_path}_{self.num_timesteps}")
#         return True



class TensorboardCallback(BaseCallback):
    def __init__(self, save_interval, save_path, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{self.num_timesteps}")
        return True
    
    def _on_rollout_end(self) -> None:
        obs = self.training_env.get_attr("curr_obs")
        poses_s = obs[0]["poses_s"]
        self.logger.record("rollout/poses_s", poses_s)
        return True


# class LogFinalPosesSCallback(SaveModelCallback):
#     def __init__(self, save_interval, save_path, log_dir, verbose=0):
#         super(LogFinalPosesSCallback, self).__init__(save_interval, save_path, verbose)
#         self.log_dir = log_dir
#         self.tensorboard = TensorBoardOutputFormat(log_dir)
#         self.n_calls = 0
#         self.last_obs = None  # Add this line

#     def _on_step(self) -> bool:
#         super(LogFinalPosesSCallback, self)._on_step()
#         self.last_obs = self.locals.get("obs", None)  # Add this line
#         return True

#     def _on_rollout_end(self) -> None:
#         if self.last_obs is not None:  # Add this line
#             obs = self.last_obs
#             poses_s = obs['poses_s']
#             episode = self.locals["episode_num"]

#             # Log the final value of poses_s
#             self.tensorboard.write_scalar("final_poses_s", poses_s, episode)


class FilterObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            'scans': spaces.Box(low=0, high=100, shape=(NUM_BEAMS, ), dtype=np.float32),
            'poses_s': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),      
            'poses_d': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),       
            'poses_theta': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(1,), dtype=np.float32),       
            'linear_vels_s': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),     
            'linear_vels_d': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),    
            'ang_vels_z': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),    
        })

    def observation(self, observation):
        # Remove unnecessary keys from the observation
        keys_to_remove = ['ego_idx', 'collisions', 'lap_times', 'lap_counts']
        filtered_observation = {k: v for k, v in observation.items() if k not in keys_to_remove}
        return filtered_observation
                       

class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, csv_file_path):
        super(FrenetObsWrapper, self).__init__(env)
        self.map_data = read_csv(csv_file_path)
        
        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Box(low=0, high=self.num_agents - 1, shape=(1,), dtype=np.int32),
            'scans': spaces.Box(low=0, high=100, shape=(NUM_BEAMS, ), dtype=np.float32),
            'poses_s': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),      
            'poses_d': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),       
            'poses_theta': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(self.num_agents,), dtype=np.float32),       
            'linear_vels_s': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),     
            'linear_vels_d': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'ang_vels_z': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'collisions': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),   
            'lap_times': spaces.Box(low=0, high=1e6, shape=(self.num_agents,), dtype=np.float32), 
            'lap_counts': spaces.Box(low=0, high=9999, shape=(self.num_agents,), dtype=np.int32)    
        })

    def observation(self, obs):


        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        frenet_coords = np.array([convert_to_frenet(poses_x[i], poses_y[i], self.map_data) for i in range(len(poses_x))])

        obs['poses_s'] = frenet_coords[:, 0]
        obs['poses_d'] = frenet_coords[:, 1]
        
        # Convert 'linear_vels_x' and 'linear_vels_y' to s and d velocities
        s_velocities = obs['linear_vels_x'] * np.cos(obs['poses_theta']) + obs['linear_vels_y'] * np.sin(obs['poses_theta'])
        d_velocities = -obs['linear_vels_x'] * np.sin(obs['poses_theta']) + obs['linear_vels_y'] * np.cos(obs['poses_theta'])

        obs['linear_vels_s'] = s_velocities
        obs['linear_vels_d'] = d_velocities

        # Remove original 'poses_x' and 'poses_y'
        del obs['poses_x']
        del obs['poses_y']
        del obs['linear_vels_x']
        del obs['linear_vels_y']

        return obs
    

class ScaledObservationEnv(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.scalers = {}

        for key, space in self.env.observation_space.spaces.items():
            if isinstance(space, spaces.Box) and space.dtype in [np.float32, np.float64]:
                scaler = StandardScaler()
                scaler.fit(np.array([space.low, space.high]))
                self.scalers[key] = scaler

    def _handle_invalid_values(self, obs):
        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                obs[key] = np.zeros_like(value)
        return obs

    def observation(self, observation):
        observation = self._handle_invalid_values(observation)
        
        scaled_observation = observation.copy()
        for key, scaler in self.scalers.items():
            scaled_observation[key] = scaler.transform(observation[key].reshape(1, -1)).reshape(-1)
        return scaled_observation