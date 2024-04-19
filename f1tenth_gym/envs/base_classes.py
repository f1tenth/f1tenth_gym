from __future__ import annotations
import numpy as np
from .dynamic_models import DynamicModel
from .action import CarAction
from .collision_models import collision_multiple, get_vertices
from .integrator import EulerIntegrator, IntegratorType
from .laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from .track import Track


class RaceCar(object):
    """Base level race car class, handles the physics and laser scan of a single vehicle

    Parameters
    ----------
        params : _type_
            vehicle parameters dictionary
        seed : _type_
            random seed
        action_type : CarAction
            action type for the cars
        integrator : _type_, optional
            integrator type, by default EulerIntegrator()
        model : _type_, optional
            vehicle model type, by default DynamicModel.ST
        is_ego : bool, optional
            ego identifier, by default False
        time_step : float, optional
            physics sim time step, by default 0.01
        num_beams : int, optional
            number of beams in the laser scan, by default 1080
        fov : float, optional
            field of view of the laser, by default 4.7


    Raises
    ------
    ValueError
        No Control Action Type Specified.
    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(
        self,
        params,
        seed,
        action_type: CarAction,
        integrator=EulerIntegrator(),
        model=DynamicModel.ST,
        is_ego=False,
        time_step=0.01,
        num_beams=1080,
        fov=4.7,
    ):

        # initialization
        self.params = params
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        self.action_type = action_type
        self.model = model

        # state of the vehicle
        self.state = self.model.get_initial_state()

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0,))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams,))
            RaceCar.scan_angles = np.zeros((num_beams,))
            RaceCar.side_distances = np.zeros((num_beams,))

            dist_sides = params["width"] / 2.0
            dist_fr = (params["lf"] + params["lr"]) / 2.0

            for i in range(num_beams):
                angle = -fov / 2.0 + i * scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi / 2.0)
                        to_fr = dist_fr / np.sin(angle - np.pi / 2.0)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi / 2)
                        to_fr = dist_fr / np.sin(-angle - np.pi / 2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """Updates the physical parameters of the vehicle

        Parameters
        ----------
        params : dict
            new parameters for the vehicle
        """
        self.params = params

    def set_map(self, map: str | Track):
        """Sets the map for scan simulator

        Parameters
        ----------
        map : str | Track
            name of the map, or Track object
        """
        RaceCar.scan_simulator.set_map(map)

    def reset(self, pose):
        """Resets the vehicle to a pose

        Parameters
        ----------
        pose : np.ndarray
            pose to reset the vehicle to
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # init state from pose
        self.state = self.model.get_initial_state(pose=pose)

        self.steer_buffer = np.empty((0,))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """Ray cast onto other agents in the env, modify original scan

        Parameters
        ----------
        scan : np.ndarray
            original scan range array

        Returns
        -------
        np.ndarray
            modified scan
        """
        # starting from original scan
        new_scan = scan

        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(
                opp_pose, self.params["length"], self.params["width"]
            )

            new_scan = ray_cast(
                np.append(self.state[0:2], self.state[4]),
                new_scan,
                self.scan_angles,
                opp_vertices,
            )

        return new_scan

    def check_ttc(self, current_scan):
        """Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        Parameters
        ----------
        current_scan : np.ndarray
            current laser scan

        Returns
        -------
        bool
            whether the scan given indicates the vehicle is in collision with environment
        """
        in_collision = check_ttc_jit(
            current_scan,
            self.state[3],
            self.scan_angles,
            self.cosines,
            self.side_distances,
            self.ttc_thresh,
        )

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.0
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, vel):
        """Steps the vehicle's physical simulation

        Parameters
        ----------
        raw_steer : float
            desired steering angle, or desired steering velocity
        vel : float
            desired longitudinal velocity, or desired longitudinal acceleration

        Returns
        -------
        np.ndarray
            current laser scan

        Raises
        ------
        ValueError
            No Control Action Type Specified.
        """
        # steering delay
        steer = 0.0
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.0
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)

        if self.action_type.type is None:
            raise ValueError("No Control Action Type Specified.")

        accl, sv = self.action_type.act(
            action=(vel, steer), state=self.state, params=self.params
        )

        u_np = np.array([sv, accl])

        f_dynamics = self.model.f_dynamics
        self.state = self.integrator.integrate(
            f=f_dynamics, x=self.state, u=u_np, dt=self.time_step, params=self.params
        )

        # bound yaw angle
        if self.state[4] > 2 * np.pi:
            self.state[4] = self.state[4] - 2 * np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2 * np.pi

        # update scan
        current_scan = RaceCar.scan_simulator.scan(
            np.append(self.state[0:2], self.state[4]), self.scan_rng
        )

        return current_scan

    def update_opp_poses(self, opp_poses):
        """Updates the vehicle's information on other vehicles

        Parameters
        ----------
        opp_poses : np.ndarray
            updated poses of other agents
        """
        self.opp_poses = opp_poses

    def update_scan(self, agent_scans, agent_index):
        """Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Parameters
        ----------
        agent_scans : list[np.ndarray]
            list of scans of each agent
        agent_index : int
            index of agent
        """
        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan


class Simulator(object):
    """Simulator class, handles the interaction and update of all vehicles in the environment

    Parameters
    ----------
    params : dict
        vehicle parameter dictionary
    num_agents : int
        number of agents in the environment
    seed : int
        seed of the rng in scan simulation
    action_type : CarAction
        action type to use for controlling the vehicles
    integrator : Integrator, optional
        integrator to use for vehicle dynamics, by default IntegratorType.RK4
    model : Model, optional
        vehicle dynamics model to use, by default DynamicModel.ST
    time_step : float, optional
        physics time step, by default 0.01
    ego_idx : int, optional
        ego vehicle's index in list of agents, by default 0

    Raises
    ------
    IndexError
        Index given is out of bounds for list of agents.
    ValueError
        Number of poses for reset does not match number of agents.
    """

    def __init__(
        self,
        params,
        num_agents,
        seed,
        action_type: CarAction,
        integrator=IntegratorType.RK4,
        model=DynamicModel.ST,
        time_step=0.01,
        ego_idx=0,
    ):
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agent_steerings = np.empty((self.num_agents,))
        self.agents: list[RaceCar] = []
        self.collisions = np.zeros((self.num_agents,))
        self.collision_idx = -1 * np.ones((self.num_agents,))
        self.model = model

        # initializing agents
        for i in range(self.num_agents):
            car = RaceCar(
                params,
                self.seed,
                is_ego=bool(i == ego_idx),
                time_step=self.time_step,
                integrator=integrator,
                model=model,
                action_type=action_type,
            )
            self.agents.append(car)

        # initialize agents scan, to be accessed from observation types
        num_beams = self.agents[0].scan_simulator.num_beams
        self.agent_scans = np.empty((self.num_agents, num_beams))

    def set_map(self, map: str | Track):
        """Sets the map of the environment and sets the map for scan simulator of each agent

        Parameters
        ----------
        map : str | Track
            name of the map, or Track object
        """
        for agent in self.agents:
            agent.set_map(map)

    def update_params(self, params, agent_idx=-1):
        """Updates the params of agents, if an index of an agent is given, update only that agent's params

        Parameters
        ----------
        params : dict
            dictionary of params
        agent_idx : int, optional
            index for agent that needs param update, if negative, update all agents, by default -1

        Raises
        ------
        IndexError
            Index given is out of bounds for list of agents.
        """
        self.params = params
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError("Index given is out of bounds for list of agents.")

    def check_collision(self):
        """Checks for collision between agents using GJK and agents' body vertices"""
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(
                np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                self.params["length"],
                self.params["width"],
            )
        self.collisions, self.collision_idx = collision_multiple(all_vertices)

    def step(self, control_inputs):
        """Steps the simulation environment

        Parameters
        ----------
        control_inputs : np.ndarray
            control inputs of all agents, first column is desired steering angle, second column is desired velocity
        """

        # looping over agents
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            self.agent_scans[i, :] = current_scan

            # update sim's information of agent poses
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])
            self.agent_steerings[i] = agent.state[2]

        # check collisions between all agents
        self.check_collision()

        for i, agent in enumerate(self.agents):
            # update agent's information on other agents
            opp_poses = np.concatenate(
                (self.agent_poses[0:i, :], self.agent_poses[i + 1 :, :]), axis=0
            )
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(self.agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.0

    def reset(self, poses):
        """Resets the simulation environment by given poses

        Parameters
        ----------
        poses : np.ndarray
            poses to reset agents to

        Raises
        ------
        ValueError
            Number of poses for reset does not match number of agents.
        """
        if poses.shape[0] != self.num_agents:
            raise ValueError(
                "Number of poses for reset does not match number of agents."
            )

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])
