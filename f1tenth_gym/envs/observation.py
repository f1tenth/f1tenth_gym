from __future__ import annotations
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

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()

class DirectObservation(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)
        
        complete_space = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict = {
                "scan": gym.spaces.Box(
                    low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32
                ),
                "std_state": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(7,), dtype=np.float32
                ),
                # "dyna_state": gym.spaces.Box(
                #     low=-large_num, high=large_num, shape=(int(self.env.model.state_dim),), dtype=np.float32
                # ),
                # "frenet_state": gym.spaces.Box(
                #     low=-large_num, high=large_num, shape=(7,), dtype=np.float32
                # ),
                "collision": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
                "lap_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "lap_count": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
            complete_space[agent_id] = gym.spaces.Dict(
                agent_dict
            )

        obs_space = gym.spaces.Dict(complete_space)
        return obs_space

    def observe(self):
        obs = {}  # dictionary agent_id -> observation dict

        for i, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = self.env.unwrapped.sim.agent_scans[i]
            agent = self.env.unwrapped.sim.agents[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]

            # create agent's observation dict
            agent_obs = {
                "scan": scan,
                "std_state": agent.standard_state,
                "collision": agent.in_collision,
                "lap_time": lap_time,
                "lap_count": lap_count,
            }

            # add agent's observation to multi-agent observation
            obs[agent_id] = agent_obs

            # cast to match observation space
            for key in obs[agent_id].keys():
                obs[agent_id][key] = np.array(obs[agent_id][key], dtype=np.float32)

        return obs

class OriginalObservation(Observation):
    def __init__(self, env):
        super().__init__(env)

    def space(self):
        num_agents = self.env.unwrapped.num_agents
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = (
            self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        )  # add 1.0 to avoid small errors
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_range,
                    shape=(num_agents, scan_size),
                    dtype=np.float32,
                ),
                "poses_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_theta": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "ang_vels_z": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "collisions": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
                ),
                "lap_times": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "lap_counts": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
            }
        )

        return obs_space

    def observe(self):
        # state indices
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(
            7
        )  # 7 largest state size (ST Model)

        observations = {
            "ego_idx": self.env.unwrapped.sim.ego_idx,
            "scans": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "linear_vels_x": [],
            "linear_vels_y": [],
            "ang_vels_z": [],
            "collisions": [],
            "lap_times": [],
            "lap_counts": [],
        }

        for i, agent in enumerate(self.env.unwrapped.sim.agents):
            agent_scan = self.env.unwrapped.sim.agent_scans[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]
            collision = self.env.unwrapped.sim.collisions[i]

            std_state = agent.standard_state
            x, y, theta = std_state[0], std_state[1], std_state[4]
            vx = std_state[3] * np.cos(std_state[6])
            vy = std_state[3] * np.sin(std_state[6])
            angvel = std_state[5]

            observations["scans"].append(agent_scan)
            observations["poses_x"].append(x)
            observations["poses_y"].append(y)
            observations["poses_theta"].append(theta)
            observations["linear_vels_x"].append(vx)
            observations["linear_vels_y"].append(vy)
            observations["ang_vels_z"].append(angvel)
            observations["collisions"].append(collision)
            observations["lap_times"].append(lap_time)
            observations["lap_counts"].append(lap_count)

        # cast to match observation space
        for key in observations.keys():
            if isinstance(observations[key], np.ndarray) or isinstance(
                observations[key], list
            ):
                observations[key] = np.array(observations[key], dtype=np.float32)

        return observations


def observation_factory(env, type: str | None) -> Observation:
    type = type or "original"

    if type == "original":
        return OriginalObservation(env)
    elif type == "direct":
        return DirectObservation(env)
    else:
        raise ValueError(f"Invalid observation type {type}.")
