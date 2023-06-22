"""
Author: Luigi Berducci
"""
from abc import abstractmethod
from typing import List

import gymnasium as gym
import numpy as np


class Observation:
    """
    Abstract class for observations. Each observation must implement the space and observe methods.

    :param env: The environment.
    :param vehicle_id: The id of the observer vehicle.
    :param kwargs: Additional arguments.
    """

    def __init__(self, env: 'F110Env'):
        self.env = env

    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()


class OriginalObservation(Observation):
    def __init__(self, env: 'F110Env'):
        super().__init__(env)

    def space(self):
        num_agents = self.env.num_agents
        scan_size = self.env.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        obs_space = gym.spaces.Dict({
            'ego_idx': gym.spaces.Discrete(num_agents),
            'scans': gym.spaces.Box(low=0.0, high=scan_range, shape=(num_agents, scan_size), dtype=np.float32),
            'poses_x': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'poses_y': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'poses_theta': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'linear_vels_x': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'linear_vels_y': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'ang_vels_z': gym.spaces.Box(low=-large_num, high=large_num, shape=(num_agents,), dtype=np.float32),
            'collisions': gym.spaces.Box(low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32),
            'lap_times': gym.spaces.Box(low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32),
            'lap_counts': gym.spaces.Box(low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32),
        })

        return obs_space

    def observe(self):
        # state indices
        agent = self.env.sim.agents[0]
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(len(agent.state))

        observations = {'ego_idx': self.env.sim.ego_idx,
                        'scans': [],
                        'poses_x': [],
                        'poses_y': [],
                        'poses_theta': [],
                        'linear_vels_x': [],
                        'linear_vels_y': [],
                        'ang_vels_z': [],
                        'collisions': [],
                        'lap_times': [],
                        'lap_counts': []}

        for i, agent in enumerate(self.env.sim.agents):
            agent_scan = self.env.sim.agent_scans[i]
            lap_time = self.env.lap_times[i]
            lap_count = self.env.lap_counts[i]
            collision = self.env.sim.collisions[i]

            observations['scans'].append(agent_scan)
            observations['poses_x'].append(agent.state[xi])
            observations['poses_y'].append(agent.state[yi])
            observations['poses_theta'].append(agent.state[yawi])
            observations['linear_vels_x'].append(agent.state[vxi])
            observations['linear_vels_y'].append(0.0)
            observations['ang_vels_z'].append(agent.state[yaw_ratei])
            observations['collisions'].append(collision)
            observations['lap_times'].append(lap_time)
            observations['lap_counts'].append(lap_count)

        # cast to match observation space
        for key in observations.keys():
            if isinstance(observations[key], np.ndarray) or isinstance(observations[key], list):
                observations[key] = np.array(observations[key], dtype=np.float32)

        return observations


class FeaturesObservation(Observation):
    def __init__(self, env: 'F110Env', features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        scan_size = self.env.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        complete_space = {}
        for agent_id in self.env.agent_ids:
            agent_dict = {
                "scan": gym.spaces.Box(low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32),
                "pose_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "pose_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "pose_theta": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "linear_vel_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "linear_vel_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "ang_vel_z": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "delta": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "beta": gym.spaces.Box(low=-large_num, high=large_num, shape=(1,), dtype=np.float32),
                "collision": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "lap_time": gym.spaces.Box(low=0.0, high=large_num, shape=(1,), dtype=np.float32),
                "lap_count": gym.spaces.Box(low=0.0, high=large_num, shape=(1,), dtype=np.float32),
            }
            complete_space[agent_id] = gym.spaces.Dict({k: agent_dict[k] for k in self.features})

        obs_space = gym.spaces.Dict(complete_space)
        return obs_space

    def observe(self):
        # state indices
        agent = self.env.sim.agents[0]
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(len(agent.state))

        obs = {}  # dictionary agent_id -> observation dict

        for i, agent_id in enumerate(self.env.agent_ids):
            scan = self.env.sim.agent_scans[i]
            agent = self.env.sim.agents[i]

            # create agent's observation dict
            agent_obs = {
                "scan": scan,
                "pose_x": agent.state[xi],
                "pose_y": agent.state[yi],
                "pose_theta": agent.state[yawi],
                "linear_vel_x": agent.state[vxi],
                "linear_vel_y": 0.0,  # note: always set to 0.0? deprecated or not?
                "ang_vel_z": agent.state[yaw_ratei],
                "delta": agent.state[deltai],
                "beta": agent.state[slipi],
                "collision": int(agent.in_collision),
                "lap_time": self.env.lap_times[i],
                "lap_count": self.env.lap_counts[i],
            }

            # add agent's observation to multi-agent observation
            obs[agent_id] = {k: agent_obs[k] for k in self.features}

        return obs


def observation_factory(env: 'F110Env', type: str, **kwargs) -> Observation:
    if type == "original":
        return OriginalObservation(env)
    elif type == "features":
        return FeaturesObservation(env, **kwargs)
    elif type == "kinematic_state":
        features = ["pose_x", "pose_y", "pose_theta", "linear_vel_x"]
        return FeaturesObservation(env, features=features)
    elif type == "dynamic_state":
        features = ["pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta", "ang_vel_z", "beta"]
        return FeaturesObservation(env, features=features)
    else:
        raise ValueError(f"Invalid observation type {type}.")
