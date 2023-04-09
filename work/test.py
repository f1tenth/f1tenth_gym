import os
import gym
import time
import glob
import numpy as np
import stable_baselines3

from gym import spaces
from datetime import datetime

class F110_Wrapped(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)

    # here we define a normalised range [-1, 1], to allow for faster training
    self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)
    
    # observation space, just taking the lidar scans for now
    self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(1080,), dtype=np.float)
    
    self.s_min = self.env.params['s_min']
    self.s_max = self.env.params['s_max']
    self.v_min = self.env.params['v_min']
    self.v_max = self.env.params['v_max']
    self.range_s = self.s_max - self.s_min
    self.range_v = self.v_max - self.v_min

  def step(self, action):

    # convert normalised actions (from RL algorithm) back to actual action range
    action_convert = self.convert_actions(action)
    # step forward on frame in the simulator with the chosen actions
    observation, _, done, info = self.env.step(np.array([action_convert]))

    # currently setting the magnitude of the car's velocity to be a positive reward
    vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
    reward = vel_magnitude
 
    return observation['scans'][0], reward, bool(done), info

  def reset(self):
    rand_x = np.random.uniform(-1.0, 1.0)   # random x coordinate on map (within a certain range)
    rand_y = np.random.uniform(-1.0, 1.0)   # random y coordinate on map (within a certain range)
    rand_t = np.random.uniform(65.0, 125.0) # rotational position of car (in degrees)
    starting_pose = [rand_x, rand_y, np.radians(rand_t)] # convert degrees to radians for angle
    observation, _, _, _ = self.env.reset(np.array([starting_pose]))
    return observation['scans'][0]  # reward, done, info can't be included in the Gym format

  def convert_actions(self, actions):
    # convert actions values from normalised range [-1, 1] to the normal steering/speed range
    steer = (((actions[0] + 1) * self.range_s) / 2) + self.s_min
    speed = (((actions[1] + 1) * self.range_v) / 2) + self.v_min
    return np.array([steer, speed], dtype=np.float)