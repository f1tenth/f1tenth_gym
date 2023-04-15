# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
from gym import spaces
import numpy as np
import os
import time
import random

from f110_gym.envs.base_classes import Simulator, Integrator

# pyglet.options['debug_gl'] = False

# Constants
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
DTYPE = np.float32

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # kwargs extraction     
        self.seed = kwargs.get('seed', 12345)
        
        self.maps = kwargs.get('maps')
        
        default_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 
                          'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 
                          'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 
                          'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        self.params = kwargs.get('params', default_params)
        
        # simulation parameters
        self.num_agents = kwargs.get('num_agents', 2)
        self.timestep = kwargs.get('timestep', 0.01)
        self.ego_idx = kwargs.get('ego_idx', 0)
        
        self.integrator = kwargs.get('integrator', Integrator.RK4)
           
        # number of lidar beams
        self.num_beams = kwargs.get('num_beams', 600)

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses = np.zeros((self.num_agents, 3))
        self.collisions = np.zeros(self.num_agents)

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, seed=self.seed, num_beams= self.num_beams, time_step=self.timestep, integrator=self.integrator)        
        self._set_random_map()
        
        # stateful observations for rendering
        self.render_obs = None
        
        self.action_space = spaces.Box(low=np.array([self.params['s_min'], 0]), high=np.array([self.params['s_max'], self.params['sv_max']]), dtype=np.float32)
        
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
            'lap_counts': spaces.Box(low=0, high=9999, shape=(self.num_agents,), dtype=np.int32)    
        })

    def _set_random_map(self):
        random.seed(time.time())
        map_idx = random.randint(0, len(self.maps) - 1)
        self.map_dir = '/Users/meraj/workspace/f1tenth_gym/work/tracks'
        self.map_name = 'map{}'.format(self.maps[map_idx])
        self.map_path = f"{self.map_dir}/maps/{self.map_name}.yaml"
        self.map_csv = self.read_csv(f"{self.map_dir}/centerline/{self.map_name}.csv")

        self.update_map(self.map_path, '.png')
        
    def read_csv(self, file_path):
        data = np.genfromtxt(file_path, delimiter=';', skip_header=1)
        return data

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by the simulation.

        Args:
            map_path (str): Absolute path to the map YAML file.
            map_ext (str): Extension of the map image file.
        """
        # print(map_path)
        self.sim.set_map(map_path, map_ext)

        # if F110Env.renderer is not None:
        #     F110Env.renderer.update_map(map_path[:-5], map_ext)
        
    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        self.max_episode_time = 500
        
        if self.current_time >= self.max_episode_time or self.lap_counts == 3:
            done = True

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        for key in ['poses_x', 'poses_y', 'poses_theta', 'collisions']:
            setattr(self, key, obs_dict[key])
    
    def get_obs(self):
        return self.curr_obs
    
    def _format_obs(self, obs):
        formatted_obs = {
            key: np.array(value, dtype=DTYPE)
            for key, value in obs.items()
        }
        formatted_obs['ego_idx'] = np.array([obs['ego_idx']], dtype=np.int32)
        formatted_obs['lap_counts'] = np.array(obs['lap_counts'], dtype=np.int32)
        return formatted_obs

    def _update_render_obs(self, obs):
        self.render_obs = {
            key: obs[key] for key in ['ego_idx', 'poses_x', 'poses_y', 'poses_theta', 'lap_times', 'lap_counts']
        }
        
    def _convert_obs_to_arrays(self, obs):
        return {key: np.array(value) for key, value in obs.items()}
    

    def step(self, action):
        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        self._update_render_obs(obs)

        self.current_time += self.timestep

        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': done, 'lap_count' : self.lap_counts, 'lap_times' : self.lap_times}
        
        obs['scans'] = obs['scans'][0]
        obs = self._format_obs(obs)
        self.curr_obs = obs
        return obs, 0, done, info


    def reset(self, poses=None):
        """
        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        random.seed(time.time())
        self._set_random_map()
        
        if poses is None:
            # Generate random poses for the agents
            random.seed(time.time())
            init_x = np.random.uniform(-0.3, 0.3)
            init_y = np.random.uniform(-0.3, 0.3)
            init_angle = np.pi/2 + self.map_csv[1, 3] + np.random.uniform(-np.pi/12, np.pi/12)
            poses = np.array([[init_x, init_y, init_angle]])
            
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)

        self._update_render_obs(obs)
        obs = self._convert_obs_to_arrays(obs)
        obs = self._format_obs(obs)

        return obs

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by the simulation for vehicles.

        Args:
            params (dict): Dictionary of parameters.
            index (int, default=-1): If >= 0 then only update a specific agent's params.
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add an extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): Custom function to be called during render().
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with Pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current FPS (bottom-left corner), and the race information as text.

        Args:
            mode (str, default='human'): Rendering mode, currently supports:
                'human': Slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed.
                'human_fast': Render as fast as possible.
        """
        assert mode in ['human', 'human_fast']

        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            # print(self.map_name)
            # exit()
            yaml_path = self.map_dir + '/maps/' + self.map_name
            F110Env.renderer.update_map(yaml_path, '.png')

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
