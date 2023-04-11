import gym
import numpy as np
import math

class NewReward(gym.Wrapper):
    def __init__(self, env, csv_file_path):
        super().__init__(env)
        self.map_data = read_csv(csv_file_path)


    def reward(self, obs):
        ego_d = obs["poses_d"]

        reward = 0

        # Penalize the agent for being stationary
        stationary_threshold = 0.25
        ego_linear_speed = np.sqrt(obs['linear_vels_s'][self.ego_idx]**2 + obs['linear_vels_d'][self.ego_idx]**2)
        if ego_linear_speed < stationary_threshold:
            reward -= 10.0


        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 1.0
        else:
            reward += 1.0

        # Encourage the agent to maintain a safe distance from the walls
        wall_distance_threshold = 0.5
        if abs(ego_d) < wall_distance_threshold:
            reward -= 3.0 * (wall_distance_threshold - abs(ego_d)) * abs(wall_distance_threshold - abs(ego_d))

        # Encourage the agent to move in the desired direction (along the s-axis)
        direction_reward_weight = 1.0
        reward += direction_reward_weight * obs['linear_vels_s'][self.ego_idx]

        # Penalize the agent for high lateral velocity (to discourage erratic behavior)
        lateral_vel_penalty_weight = 1.0
        reward -= lateral_vel_penalty_weight * abs(obs['linear_vels_d'][self.ego_idx])


        pose_theta_penalty_weight = 0.3
        desired_orientation = np.pi / 2

        # Normalize the pose_theta
        normalized_pose_theta = (obs['poses_theta'][self.ego_idx] + np.pi) % (2 * np.pi) - np.pi

        # Calculate the difference between the current orientation and desired orientation
        angle_diff = normalized_pose_theta - desired_orientation

        # Normalize the angle difference
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        reward += 1 - pose_theta_penalty_weight * abs(angle_diff) ** 2
        
        return reward

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        new_reward = self.reward(obs)
        return obs, new_reward.item(), done, info
    
    
def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=3)
    return data

def get_closest_point_index(x, y, map_data):
    map_data_np = np.array(map_data)
    map_x, map_y = map_data_np[:, 1], map_data_np[:, 2]
    distances = np.sqrt((x - map_x)**2 + (y - map_y)**2)
    closest_point_index = np.argmin(distances)
    return closest_point_index

def convert_to_frenet(x, y,vel_magnitude, pose_theta, map_data):
    closest_point_index = get_closest_point_index(x, y, map_data)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m, psi_rad = closest_point[0], closest_point[1], closest_point[2], closest_point[3]
    
    dx = x - x_m
    dy = y - y_m
    
    vx = vel_magnitude * np.cos(pose_theta)
    vy = vel_magnitude * np.sin(pose_theta)
    
    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d =  dx * np.cos(psi_rad) + dy * np.sin(psi_rad)
    
    vs = -vx * np.sin(psi_rad) + vy * np.cos(psi_rad)
    vd =  vx * np.cos(psi_rad) + vy * np.sin(psi_rad)
    
    return s, d, vs, vd
