import gym
import numpy as np

class NewReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.first = 0.0
        self.second = 0.0
        self.third = 0.0
        self.known_lap_count = 0
        self.current_action = None


    def reward(self, obs):
        vs = obs['linear_vels_s'][self.ego_idx]
        vd = obs['linear_vels_d'][self.ego_idx]

        reward = 0

        # Penalize the agent for being stationary
        stationary_threshold = 0.25
        ego_linear_speed = np.sqrt(vs ** 2 + vd ** 2)
        if ego_linear_speed < stationary_threshold:
            reward -= 1.0

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 1000.0
        else:
            reward += 1.0

        # Encourage the agent to move in the desired direction (along the s-axis)
        reward += 1.0 * vs 
        # reward -= 0.5 * abs(vd)
        
        # lap_count = obs['lap_counts'][self.ego_idx]
        # lap_time  = obs['lap_times'][self.ego_idx]
        
        # if lap_count != self.known_lap_count and lap_time >= 50.0:
        #     self.known_lap_count = lap_count
        #     reward += max(500 - 2.5 * self.first, 500)
        #     print(lap_time)
        # print('reward ', reward)
        
        return reward



    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        self.current_action = action
        new_reward = self.reward(obs)
        return obs, new_reward.item(), done, info
    

