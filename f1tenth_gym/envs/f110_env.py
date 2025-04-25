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

"""
Author: Hongrui Zheng
"""

# gym imports
import gymnasium as gym

# others
import numpy as np

from .action import CarAction, from_single_to_multi_action_space

# base classes
from .base_classes import DynamicModel, Simulator
from .integrator import IntegratorType
from .observation import observation_factory
from .rendering import make_renderer
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment.

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

    # NOTE: change matadata with default rendering-modes, add definition of render_fps
    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 100}

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        self.seed = self.config["seed"]
        self.map = self.config["map"]
        self.params = self.config["params"]
        self.num_agents = self.config["num_agents"]
        self.timestep = self.config["timestep"]
        self.ego_idx = self.config["ego_idx"]
        self.integrator = IntegratorType.from_string(self.config["integrator"])
        self.model = DynamicModel.from_string(self.config["model"])
        self.observation_config = self.config["observation_config"]
        self.action_type = CarAction(self.config["control_input"], params=self.params)

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            time_step=self.timestep,
            integrator=self.integrator,
            model=self.model,
            action_type=self.action_type,
        )
        self.sim.set_map(self.map, self.config["scale"])

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(
                self.map,
                track_scale=self.config["scale"],
            )  # load track in gym env for convenience

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        assert (
            "type" in self.observation_config
        ), "observation_config must contain 'type' key"
        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()

        # action space
        self.action_space = from_single_to_multi_action_space(
            self.action_type.space, self.num_agents
        )

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.render_obs = None
        self.render_mode = render_mode

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

    @classmethod
    def fullscale_vehicle_params(cls) -> dict:
        params = {
            "mu": 1.0489,
            "C_Sf": 20.89,
            "C_Sr": 20.89,
            "lf": 0.88392,
            "lr": 1.50876,
            "h": 0.557,
            "m": 1225.8878467253344,
            "I": 1538.8533713561394,
            "width": 1.674,
            "length": 4.298,
            # steering constraints
            "s_min": -0.91,
            "s_max": 0.91,
            "sv_min": -0.4,
            "sv_max": 0.4,
            # maximum curvature change
            "kappa_dot_max": 0.4,
            # maximum curvature rate rate
            "kappa_dot_dot_max": 20,
            # Longitudinal constraints
            "v_switch": 4.755,
            "a_max": 11.5,
            "v_min": -13.9,
            "v_max": 45.8,
            # maximum longitudinal jerk [m/s^3]
            "j_max": 10.0e3,
            # maximum longitudinal jerk change [m/s^4]
            "j_dot_max": 10.0e3,
            # Extra parameters (for future use in multibody simulation)
            # sprung mass [kg]  SMASS
            "m_s": 1094.542720290477,
            # unsprung mass front [kg]  UMASSF
            "m_uf": 65.67256321742863,
            # unsprung mass rear [kg]  UMASSR
            "m_ur": 65.67256321742863,
            # moments of inertia of sprung mass
            # moment of inertia for sprung mass in roll [kg m^2]  IXS
            "I_Phi_s": 244.04723069965206,
            # moment of inertia for sprung mass in pitch [kg m^2]  IYS
            "I_y_s": 1342.2597688480864,
            # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
            "I_z": 1538.8533713561394,
            # moment of inertia cross product [kg m^2]  IXZ
            "I_xz_s": 0.0,
            # suspension parameters
            # suspension spring rate (front) [N/m]  KSF
            "K_sf": 21898.332429625985,
            # suspension damping rate (front) [N s/m]  KSDF
            "K_sdf": 1459.3902937206362,
            # suspension spring rate (rear) [N/m]  KSR
            "K_sr": 21898.332429625985,
            # suspension damping rate (rear) [N s/m]  KSDR
            "K_sdr": 1459.3902937206362,
            # geometric parameters
            # track width front [m]  TRWF
            "T_f": 1.389888,
            # track width rear [m]  TRWB
            "T_r": 1.423416,
            # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS
            "K_ras": 175186.65943700788,
            # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
            "K_tsf": -12880.270509148304,
            # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
            "K_tsr": 0.0,
            # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
            "K_rad": 10215.732056044453,
            # vertical spring rate of tire [N/m]  KZT
            "K_zt": 189785.5477234252,
            # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
            "h_cg": 0.5577840000000001,
            # height of roll axis above ground (front) [m]  HRAF
            "h_raf": 0.0,
            # height of roll axis above ground (rear) [m]  HRAR
            "h_rar": 0.0,
            # M_s center of gravity above ground [m]  HS
            "h_s": 0.59436,
            # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
            "I_uf": 32.53963075995361,
            # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
            "I_ur": 32.53963075995361,
            # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]
            "I_y_w": 1.7,
            # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
            "K_lt": 1.0278264878518764e-05,
            # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]
            "R_w": 0.344,
            # split of brake and engine torque
            "T_sb": 0.76,
            "T_se": 1,
            # suspension parameters
            # [rad/m]  DF
            "D_f": -0.6233595800524934,
            # [rad/m]  DR
            "D_r": -0.20997375328083986,
            # [needs conversion if nonzero]  EF
            "E_f": 0,
            # [needs conversion if nonzero]  ER
            "E_r": 0,
            # tire parameters from ADAMS handbook
            # longitudinal coefficients
            "tire_p_cx1": 1.6411,  # Shape factor Cfx for longitudinal force
            "tire_p_dx1": 1.1739,  # Longitudinal friction Mux at Fznom
            "tire_p_dx3": 0,  # Variation of friction Mux with camber
            "tire_p_ex1": 0.46403,  # Longitudinal curvature Efx at Fznom
            "tire_p_kx1": 22.303,  # Longitudinal slip stiffness Kfx/Fz at Fznom
            "tire_p_hx1": 0.0012297,  # Horizontal shift Shx at Fznom
            "tire_p_vx1": -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
            "tire_r_bx1": 13.276,  # Slope factor for combined slip Fx reduction
            "tire_r_bx2": -13.778,  # Variation of slope Fx reduction with kappa
            "tire_r_cx1": 1.2568,  # Shape factor for combined slip Fx reduction
            "tire_r_ex1": 0.65225,  # Curvature factor of combined Fx
            "tire_r_hx1": 0.0050722,  # Shift factor for combined slip Fx reduction
            # lateral coefficients
            "tire_p_cy1": 1.3507,  # Shape factor Cfy for lateral forces
            "tire_p_dy1": 1.0489,  # Lateral friction Muy
            "tire_p_dy3": -2.8821,  # Variation of friction Muy with squared camber
            "tire_p_ey1": -0.0074722,  # Lateral curvature Efy at Fznom
            "tire_p_ky1": -21.92,  # Maximum value of stiffness Kfy/Fznom
            "tire_p_hy1": 0.0026747,  # Horizontal shift Shy at Fznom
            "tire_p_hy3": 0.031415,  # Variation of shift Shy with camber
            "tire_p_vy1": 0.037318,  # Vertical shift in Svy/Fz at Fznom
            "tire_p_vy3": -0.32931,  # Variation of shift Svy/Fz with camber
            "tire_r_by1": 7.1433,  # Slope factor for combined Fy reduction
            "tire_r_by2": 9.1916,  # Variation of slope Fy reduction with alpha
            "tire_r_by3": -0.027856,  # Shift term for alpha in slope Fy reduction
            "tire_r_cy1": 1.0719,  # Shape factor for combined Fy reduction
            "tire_r_ey1": -0.27572,  # Curvature factor of combined Fy
            "tire_r_hy1": 5.7448e-006,  # Shift factor for combined Fy reduction
            "tire_r_vy1": -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
            "tire_r_vy3": -0.27568,  # Variation of Svyk/Muy*Fz with camber
            "tire_r_vy4": 12.12,  # Variation of Svyk/Muy*Fz with alpha
            "tire_r_vy5": 1.9,  # Variation of Svyk/Muy*Fz with kappa
            "tire_r_vy6": -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
        }
        return params

    @classmethod
    def f1fifth_vehicle_params(cls) -> dict:
        params = {
            "mu": 1.1,
            "C_Sf": 5.3507,
            "C_Sr": 5.3507,
            "lf": 0.2725,
            "lr": 0.2585,
            "h": 0.1825,
            "m": 15.32,
            "I": 0.64332,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.55,
            "length": 0.8,
        }
        return params

    @classmethod
    def f1tenth_vehicle_params(cls) -> dict:
        params = {
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.15875,
            "lr": 0.17145,
            "h": 0.074,
            "m": 3.74,
            "I": 0.04712,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.31,
            "length": 0.58,
        }
        return params

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().

        Args:
            None

        Returns:
            a configuration dict
        """
        return {
            "seed": 12345,
            "map": "Spielberg",
            "scale": 1.0,
            "params": cls.f1tenth_vehicle_params(),
            "num_agents": 2,
            "timestep": 0.01,
            "ego_idx": 0,
            "integrator": "rk4",
            "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": None},
            "reset_config": {"type": None},
            "scale": 1.0,
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config = deep_update(self.config, config)
            self.params = self.config["params"]

            if hasattr(self, "sim"):
                self.sim.update_params(self.config["params"])

            if hasattr(self, "action_space"):
                # if some parameters changed, recompute action space
                self.action_type = CarAction(
                    self.config["control_input"], params=self.params
                )
                self.action_space = from_single_to_multi_action_space(
                    self.action_type.space, self.num_agents
                )

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

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
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

        return bool(done), self.toggle_list >= 4

    def _update_state(self):
        """
        Update the env's states according to observations.
        """
        self.poses_x = self.sim.agent_poses[:, 0]
        self.poses_y = self.sim.agent_poses[:, 1]
        self.poses_theta = self.sim.agent_poses[:, 2]
        self.collisions = self.sim.collisions

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)

        # observation
        obs = self.observation_type.observe()

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "steering_angles": self.sim.agent_steerings,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
        }

        # check done
        done, toggle_list = self._check_done()
        truncated = False
        info = {"checkpoint_done": toggle_list}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        return obs, info

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        self.sim.set_map(map_name)
        self.track = Track.from_track_name(map_name)

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

        self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()
