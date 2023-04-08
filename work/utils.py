import gym
import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import BaseCallback
from gym import spaces
import torch
import math

class SaveModelCallback(BaseCallback):
    def __init__(self, save_interval, save_path):
        super().__init__()
        self.save_interval = save_interval
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            self.model.save(f"{self.save_path}_{self.num_timesteps}")
        return True

class FilterObservationSpace(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            'scans': spaces.Box(low=0, high=100, shape=(1080, ), dtype=np.float32),
            'poses_s': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),      
            'poses_d': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),       
            'poses_theta': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(1,), dtype=np.float32),       
            'linear_vels_x': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),     
            'linear_vels_y': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),    
            'ang_vels_z': spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),    
        })

    def observation(self, observation):
        # Remove unnecessary keys from the observation
        keys_to_remove = ['ego_idx', 'collisions', 'lap_times', 'lap_counts']
        filtered_observation = {k: v for k, v in observation.items() if k not in keys_to_remove}
        return filtered_observation

class TensorboardCallback(BaseCallback):
    def __init__(self, env):
        super().__init__(verbose=0)
        self.env = env
        self.episode_rewards = []
    def _on_step(self) -> bool:
        # Accumulate rewards for each environment
        if self.locals.get("rewards") is not None:
            rewards = self.locals["rewards"]
            self.episode_rewards.append(rewards)

        # Log episode reward to Tensorboard when an episode ends
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            for idx, done in enumerate(dones):
                if done:
                    episode_reward = sum(self.episode_rewards)
                    self.logger.record("episode_reward", episode_reward)
                    self.episode_rewards = []  # Reset the accumulated rewards
                    self.logger.dump(step=self.num_timesteps)  # Write logs to the Tensorboard file

        return True


class NewReward(gym.Wrapper):
    def __init__(self, env, csv_file_path):
        super().__init__(env)
        self.map_data = read_csv(csv_file_path)

    def reward(self, obs):
        ego_s = obs["poses_s"]
        ego_d = obs["poses_d"]

        target_s = 5.0
        target_d = 0.0

        d_s = target_s - ego_s
        d_d = target_d - ego_d
        distance = np.sqrt(d_s**2 + d_d**2)

        # Encourage the agent to move closer to the target
        reward = -0.1 * distance

        # Reward the agent for reaching the target area
        if distance < 1.0:
            reward += 1.0

        # Penalize the agent for being stationary
        stationary_threshold = 0.05
        ego_linear_speed = np.sqrt(obs['linear_vels_x'][self.ego_idx]**2 + obs['linear_vels_y'][self.ego_idx]**2)
        if ego_linear_speed < stationary_threshold:
            reward -= 0.1

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 10
            
        if obs['ang_vels_z'] > 5:
            reward -= 50

        return reward

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        new_reward = self.reward(obs)
        self.step_reward = new_reward
        return obs, new_reward.item(), done, info
                       

class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, csv_file_path):
        super(FrenetObsWrapper, self).__init__(env)
        self.map_data = read_csv(csv_file_path)
        
        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Box(low=0, high=self.num_agents - 1, shape=(1,), dtype=np.int32),
            'scans': spaces.Box(low=0, high=100, shape=(1080, ), dtype=np.float32),
            'poses_s': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),      
            'poses_d': spaces.Box(low=-1000, high=1000, shape=(self.num_agents,), dtype=np.float32),       
            'poses_theta': spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(self.num_agents,), dtype=np.float32),       
            'linear_vels_x': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),     
            'linear_vels_y': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'ang_vels_z': spaces.Box(low=-10, high=10, shape=(self.num_agents,), dtype=np.float32),    
            'collisions': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),   
            'lap_times': spaces.Box(low=0, high=1e6, shape=(self.num_agents,), dtype=np.float32), 
            'lap_counts': spaces.Box(low=0, high=9999, shape=(self.num_agents,), dtype=np.int32)    
        })

    def observation(self, obs):


        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        # frenet_coords = np.array([convert_to_frenet(x, y, self.map_data) for x, y in zip(poses_x, poses_y)])
        frenet_coords = np.array([convert_to_frenet(poses_x[i], poses_y[i], self.map_data) for i in range(len(poses_x))])


        # Replace 'poses_x' and 'poses_y' with Frenet coordinates
        obs['poses_s'] = frenet_coords[:, 0]
        obs['poses_d'] = frenet_coords[:, 1]

        # Remove original 'poses_x' and 'poses_y'
        del obs['poses_x']
        del obs['poses_y']

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
    
class DebugNaNsCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Check for NaNs in the observation
        if self.locals.get("clipped_actions") is not None:
            obs = self.locals["new_obs"]
            # Change starts: Check NaNs for each item in obs dictionary
            if obs is not None:
                for key, value in obs.items():
                    value_tensor = torch.tensor(value)
                    if torch.isnan(value_tensor).any():
            # Change ends
                        print(f"NaN detected in observation: {key}")
                        return False  # Stop training

        # Check for NaNs in the action
        if self.locals.get("actions") is not None:
            actions = self.locals["actions"]
            if torch.isnan(torch.tensor(actions)).any():
                print("NaN detected in actions")
                return False  # Stop training

        # Check for NaNs in the rewards
        if self.locals.get("rewards") is not None:
            rewards = self.locals["rewards"]
            if torch.isnan(torch.tensor(rewards)).any():
                print("NaN detected in rewards")
                return False  # Stop training

        return True  # Continue trainingclass DebugNaNsCallback(BaseCallback):

def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data

def get_closest_point_index(x, y, map_data):
    map_data_np = np.array(map_data)
    map_x, map_y = map_data_np[:, 1], map_data_np[:, 2]
    distances = np.sqrt((x - map_x)**2 + (y - map_y)**2)
    closest_point_index = np.argmin(distances)
    return closest_point_index

# @lru_cache(maxsize=1000)
def convert_to_frenet(x, y, map_data):
    closest_point_index = get_closest_point_index(x, y, map_data)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m = closest_point[0], closest_point[1], closest_point[2]

    if closest_point_index == 0:
        prev_point = map_data[-1]
    else:
        prev_point = map_data[closest_point_index - 1]
    x_prev, y_prev = prev_point[1], prev_point[2]

    heading = np.arctan2(y_m - y_prev, x_m - x_prev)
    dx = x - x_m
    dy = y - y_m
    d = math.cos(heading) * dy - math.sin(heading) * dx
    s = s_m + math.cos(heading) * dx + math.sin(heading) * dy

    return s, d