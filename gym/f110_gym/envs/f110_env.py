# TODO: - check all todos
#       - convert env to `gymnasium`
#         - test reset method
#       - add raceline to the env and draw it on the renderer (test pp on it)
#         - improve reward considering distance to the raceline


# gym imports
import gymnasium as gym

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.raceline import Raceline
from f110_gym import ThrottledPrinter

# others
import numpy as np
import os
import time
import yaml
from PIL import Image

# gl
import pyglet
pyglet.options['debug_gl'] = False

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

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
    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 300}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        self.throttled_printer = ThrottledPrinter(min_interval=0.5)

        self.render_mode = kwargs['render_mode']
        print('Render mode:', self.render_mode)

        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            self.map_path = self.map_name + '.yaml'
        except:
            raise ValueError('Map name not provided. Please provide a map name.')

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        if not os.path.exists(self.map_name + '.yaml') or not os.path.exists(self.map_name + self.map_ext):
            raise FileNotFoundError(f"Map file {self.map_name + '.yaml'} or image file {self.map_name + self.map_ext} "
                                    f"not found.")
        with open(self.map_name + '.yaml', 'r') as file:
            try:
                map_yaml = yaml.safe_load(file)
                self.resolution = map_yaml['resolution']
                self.origin_x = map_yaml['origin'][0]
                self.origin_y = map_yaml['origin'][1]
            except yaml.YAMLError as exc:
                print(exc)
                raise ValueError(f"Error loading map file {self.map_name + '.yaml'}")

        self.map_img = np.array(Image.open(self.map_name + self.map_ext).transpose(Image.FLIP_TOP_BOTTOM))
        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        try:
            self.raceline_path = kwargs['raceline_path']
        except:
            raise ValueError("Raceline path not provided. Please provide a raceline path.")

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489,
                           'C_Sf': 4.718,
                           'C_Sr': 5.4562,
                           'lf': 0.15875,
                           'lr': 0.17145,
                           'h': 0.074,
                           'm': 3.74,
                           'I': 0.04712,
                           's_min': -0.46,
                           's_max': 0.46,
                           'sv_min': -3.2,
                           'sv_max': 3.2,
                           'v_switch': 7.319,
                           'a_max': 9.51,
                           'v_min':-5.0,
                           'v_max': 20.0,
                           'width': 0.31,
                           'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 1

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        # try:
        #     self.ego_idx = kwargs['ego_idx']
        # except:
        self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4

        try:
            self.points_in_foreground = kwargs['points_in_foreground']
        except:
            self.points_in_foreground = False

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # Create a single row vector for one agent
        single_agent_low = np.array(
            [-np.inf, -np.inf, self.params['s_min'], self.params['v_min'], 0.0, self.params['sv_min'], -np.inf], dtype=np.float32)
        single_agent_high = np.array(
            [+np.inf, +np.inf, self.params['s_max'], self.params['v_max'], 2 * np.pi, self.params['sv_max'], +np.inf], dtype=np.float32)

        # Duplicate for all agents to match the shape (num_agents, 7)
        obs_low = np.tile(single_agent_low, (self.num_agents, 1))
        obs_high = np.tile(single_agent_high, (self.num_agents, 1))

        # Now create the observation space with the proper dimensions
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.num_agents, 7),
            dtype=np.float32
        )

        single_agent_low = np.array([-np.inf, self.params['s_min']], dtype=np.float32)
        single_agent_high = np.array([+np.inf, self.params['s_max']], dtype=np.float32)

        action_low = np.tile(single_agent_low, (self.num_agents, 1))
        action_high = np.tile(single_agent_high, (self.num_agents, 1))

        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.num_agents, 2),
            dtype=np.float32
        )

        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        self.off_track = np.zeros((self.num_agents, ))

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator)
        self.sim.set_map(self.map_path, self.map_ext)

        self.raceline = Raceline(self.raceline_path)
        self.previous_s = None
        self.ego_lap_count = 0

        # stateful observations for rendering
        self.render_obs = None

    def _get_obs(self):
        """
        Get the current observation of the environment

        Args:
            None

        Returns:
            obs (np.ndarray): current observation of the environment
        """
        obs = np.zeros((self.num_agents, 7), dtype=np.float32)
        for i in range(self.num_agents):
            obs[i, :] = self.sim.agents[i].state

        return obs

    def _get_info(self):
        """
        Get the current information of the environment

        Args:
            None

        Returns:
            info (dict): current information of the environment
        """
        info = {
            'collisions': self.collisions
        }
        return info

    def close(self):
        """
        Finalizer, does cleanup
        """
        pass

    def is_inside_track(self, x, y):
        """
        Checks if a given (x, y) coordinate is inside the track (free space).

        Args:
            x (float): x-coordinate in world space
            y (float): y-coordinate in world space

        Returns:
            bool: True if the point is inside the track (free space), False otherwise
        """
        # Convert to pixel coordinates
        x_pixel = int((x - self.origin_x) / self.resolution)
        y_pixel = int((y - self.origin_y) / self.resolution)

        # Check if the pixel is within the bounds of the map image
        if 0 <= x_pixel < self.map_img.shape[1] and 0 <= y_pixel < self.map_img.shape[0]:
            return self.map_img[y_pixel, x_pixel] != 0
        else:
            return False

    def _check_done(self, s):
        """
        Check if the current rollout is done
        
        Args:
            obs (np.array): observation of the current step
        """

        # This check is being done only for ego
        lap_info = self.raceline.is_lap_completed(s)
        lap_completed = lap_info['lap_completed']
        lap_time = lap_info['lap_time']
        lap_orientation = lap_info['lap_orientation']

        if lap_completed and lap_orientation == 'forward':
            self.ego_lap_count += 1

        for i in range(self.num_agents):
            self.off_track[i] = 0.
            if not self.is_inside_track(self.poses_x[i], self.poses_y[i]):
                self.off_track[i] = 1.
                self.throttled_printer.print(f"Agent {i} is off track: ({self.poses_x[i]}, {self.poses_y[i]})", 'yellow')

            if F110Env.renderer is not None:
                F110Env.renderer.draw_point(self.poses_x[i], self.poses_y[i], size=10)

        done_collisions = bool(self.collisions[self.ego_idx])
        done_laps_ego = bool(self.ego_lap_count >= 2)
        done_off_track_ego = bool(self.off_track[self.ego_idx])

        done = (self.collisions[self.ego_idx]) or self.ego_lap_count >= 2 or self.off_track[self.ego_idx] == 1.
        
        return bool(done), done_collisions, done_laps_ego, done_off_track_ego, lap_info

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (np.array): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxiliary information dictionary
        """
        
        # call simulation step
        obs = self.sim.step(action)

        # reward = self.timestep
        reward = -1
        
        # update data member
        self._update_state(obs)

        observation = self._get_obs()
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        x, y = observation[self.ego_idx, 0], observation[self.ego_idx, 1]

        s, d, status = self.raceline.get_nearest_index(x, y, self.previous_s)

        # check done
        done, done_collisions, done_laps_ego, done_off_track_ego, lap_info = self._check_done(s)

        # reward = ... # TODO: based on raceline `s` and `d`
        terminated = done # if not recoverable off track or laps completed
        truncated = False  # if time limit is reached

        # temporary reward system
        if terminated:
            if done_collisions or done_off_track_ego:
                reward -= 100
            if done_laps_ego:
                reward += 100

        info = self._get_info()
        info.update({'legacy_obs': obs})
        info.update(lap_info)

        self.previous_s = s

        self.lap_times[self.ego_idx] = lap_info['lap_time'] if lap_info['lap_completed'] else 0
        self.lap_counts[self.ego_idx] = self.ego_lap_count
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
        }

        F110Env.current_obs = obs

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        super().reset(seed=seed)

        poses = np.zeros((self.num_agents, 3))
        if options is not None:
            if 'pose' in options:
                poses = options['pose']
            elif 'poses' in options:
                poses = options['poses']

        # check that poses are valid and not off track
        for i in range(self.num_agents):
            if not self.is_inside_track(float(poses[i, 0]), float(poses[i, 1])):
                raise gym.error.Error(f"Agent {i} pose is off track: ({poses[i, 0]}, {poses[i, 1]})")

        # reset counters and data members
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        self.off_track = np.zeros((self.num_agents,))

        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))

        self.render_obs = None

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        self.render_obs = {
            'ego_idx': info['legacy_obs']['ego_idx'],
            'poses_x': info['legacy_obs']['poses_x'],
            'poses_y': info['legacy_obs']['poses_y'],
            'poses_theta': info['legacy_obs']['poses_theta'],
            'lap_times': info['legacy_obs']['lap_times'],
            'lap_counts': info['legacy_obs']['lap_counts']
            }

        s, _, _ = self.raceline.get_nearest_index(obs[self.ego_idx, 0], obs[self.ego_idx, 1], previous_s=None)
        self.raceline.reset(s)
        self.previous_s = None
        self.ego_lap_count = 0
        
        return obs, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """

        if self.render_mode is None:
            return
        
        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H, self.points_in_foreground)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            F110Env.renderer.update_raceline(self.raceline)
            
        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
        
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if self.render_mode == 'human':
            time.sleep(1.0 / self.metadata['render_fps'])
        elif self.render_mode == 'human_fast':
            pass
