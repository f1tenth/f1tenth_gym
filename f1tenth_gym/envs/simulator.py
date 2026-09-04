"""Core F110 simulator handling multi-agent dynamics, LiDAR, and collision logic."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np

from .action import (
    LongitudinalActionType,
    SteerActionType,
    longitudinal_action_from_type,
    steer_action_from_type,
)
from .collision_models import CollisionCheckMode, get_vertices
from .dynamic_models import DynamicModel, VehicleParameters
from .env_config import EnvConfig
from .lidar import ray_cast
from .lidar.segment_scan import SegmentScanSimulator2D
from .state import SimulationState
from .track import Track

DynamicsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
IntegratorFn = Callable[[DynamicsFn, np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]
AccelerationFn = Callable[[float, np.ndarray, VehicleParameters], float]
SteeringFn = Callable[[float, np.ndarray, VehicleParameters], float]

@dataclass
class ScanCache:
    """Precomputed beam angles used for opponent occlusion.

    Attributes:
        angles: Beam angles relative to vehicle heading (used by ray_cast).
    """

    angles: np.ndarray


def _make_scan_simulator(lidar_cfg):
    """Build the exact segment scanner from the LiDAR configuration."""
    kwargs = dict(
        angle_min=lidar_cfg.angle_min,
        angle_max=lidar_cfg.angle_max,
        std_dev=lidar_cfg.noise_std,
        min_range=lidar_cfg.range_min,
        max_range=lidar_cfg.range_max,
    )
    return SegmentScanSimulator2D(
        lidar_cfg.num_beams,
        lidar_cfg.field_of_view,
        device=lidar_cfg.scan_device,
        **kwargs,
    )


class F110Simulator:
    """Core simulator for F1TENTH multi-agent racing.

    Handles vehicle dynamics integration, LiDAR simulation, and collision detection
    for all agents in a single state-driven update loop.

    Attributes:
        state: Current simulation state containing poses, velocities, scans.
        track: The racing track being simulated.
        vehicle_params: Physical parameters of the vehicle.
        num_agents: Number of agents in the simulation.
    """

    def __init__(
        self,
        *,
        env_config: EnvConfig,
        vehicle_params: VehicleParameters,
        model: DynamicModel,
        dynamics_fn: DynamicsFn,
        integrator_fn: IntegratorFn,
        longitudinal_type: LongitudinalActionType,
        steering_type: SteerActionType,
        track: Optional[Track],
        seed: int,
    ) -> None:
        self.config = env_config
        self.vehicle_params = vehicle_params
        self.model = model
        self.dynamics_fn = dynamics_fn
        self.integrator_fn = integrator_fn
        self.track = track
        self.seed = seed

        self.num_agents = env_config.num_agents
        self.ego_idx = env_config.ego_index
        self.time_step = env_config.simulation_config.timestep
        self.integrator_dt = env_config.simulation_config.integrator_timestep
        # `time_step % integrator_dt` rejects exact multiples IEEE754 cannot
        # represent (0.03 % 0.01 is 0.00999...), so compare the ratio instead.
        ratio = self.time_step / self.integrator_dt
        self.substeps = max(1, int(round(ratio)))
        if not np.isclose(ratio, self.substeps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"timestep ({self.time_step}) must be an integer multiple of "
                f"integrator_timestep ({self.integrator_dt}), got a ratio of {ratio}"
            )

        self.collision_check_mode: CollisionCheckMode = env_config.collision_check
        self.longitudinal_fn: AccelerationFn = longitudinal_action_from_type(longitudinal_type)
        self.steering_fn: SteeringFn = steer_action_from_type(
            steering_type, steer_kp=env_config.control_config.steer_kp
        )

        self.longitudinal_type = longitudinal_type
        self.steering_type = steering_type

        self.params_array = self.vehicle_params.to_array()

        # Allocate simulation state buffers
        initial_state = self.model.get_initial_state(params=self.params_array)
        self.state_dim = initial_state.shape[0]
        self.control_dim = self.model.control_dim
        scan_size = env_config.lidar_config.num_beams if env_config.lidar_config.enabled else 1
        self.state = SimulationState.allocate(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            scan_size=scan_size,
            control_dim=self.control_dim,
            delay_steps=env_config.control_config.steer_delay_steps,
        )

        # actuation realism: command noise + longitudinal (throttle) delay
        cc = env_config.control_config
        self._steer_noise_std = float(cc.steer_noise_std)
        self._accl_noise_std = float(cc.accl_noise_std)
        self._throttle_delay_steps = int(cc.throttle_delay_steps)
        # A None seed (EnvConfig.seed unset) still needs a concrete int here:
        # the per-agent scan RNGs below are derived as seed + agent_index, and
        # reset(noise_seed=None) falls back to self.seed the same way.
        if self.seed is None:
            self.seed = int(np.random.SeedSequence().generate_state(1)[0])
            seed = self.seed
        self.control_rng = np.random.default_rng(seed)  # command-noise stream
        if self._throttle_delay_steps > 0:
            self._throttle_buffer = np.zeros((self.num_agents, self._throttle_delay_steps), dtype=np.float32)
            self._throttle_head = np.zeros((self.num_agents,), dtype=np.int32)
        else:
            self._throttle_buffer = None
            self._throttle_head = None

        # Static helpers
        self.standard_state_fn = self.model.get_standardized_state_fn()
        self.scan_enabled = env_config.lidar_config.enabled
        self.scan_max_range = env_config.lidar_config.maximum_range

        self.scan_sims: list[SegmentScanSimulator2D] = []
        self.scan_rngs: list[np.random.Generator] = []
        self.scan_cache: list[ScanCache] = []
        # per-agent, per-beam systematic range bias (drawn per episode at reset)
        self.scan_bias = np.zeros((self.num_agents, env_config.lidar_config.num_beams), dtype=np.float32)
        if self.scan_enabled:
            lidar_cfg = env_config.lidar_config
            for agent_index in range(self.num_agents):
                rng = np.random.default_rng(seed + agent_index)
                simulator = _make_scan_simulator(lidar_cfg)
                if self.track is not None:
                    simulator.set_map(self.track, env_config.map_scale)
                cache = self._build_scan_cache(simulator)
                self.scan_sims.append(simulator)
                self.scan_rngs.append(rng)
                self.scan_cache.append(cache)

        self._collision_body_dx, self._collision_body_dy = self._compute_collision_body_offset(
            self.vehicle_params
        )
        self.contact = None
        self._build_contact()

    def _standardize(self, state: np.ndarray) -> np.ndarray:
        """Return the common CoG-referenced seven-column state."""
        return self.standard_state_fn(state).astype(np.float32)

    # --- Public API ---
    def set_map(self, track: Track, map_scale: float = 1.0) -> None:
        """Set or update the track used for simulation.

        Args:
            track: Track object with occupancy map and reference lines.
            map_scale: Scale factor applied to the map.
        """
        self.track = track
        # Walls come from the track, so contact has to be rebuilt whenever it changes.
        self._build_contact()
        if not self.scan_enabled:
            return
        for simulator in self.scan_sims:
            simulator.set_map(track, map_scale)

    def update_params(self, vehicle_params: VehicleParameters, agent_idx: int = -1) -> None:
        """Update vehicle parameters for all agents.

        Args:
            vehicle_params: New vehicle physical parameters.
            agent_idx: Agent index (-1 for all agents, per-agent not supported).
        """
        if agent_idx >= 0:
            raise NotImplementedError("Per-agent parameter updates are not supported")
        self.vehicle_params = vehicle_params
        self.params_array = vehicle_params.to_array()
        self._collision_body_dx, self._collision_body_dy = self._compute_collision_body_offset(
            self.vehicle_params
        )
        self._build_contact()

    def reset(self, poses: np.ndarray, *, option: str = "pose", noise_seed: int | None = None) -> None:
        """Reset all agents to initial positions.

        Args:
            poses: Initial positions, shape (num_agents, 3) for poses or
                   (num_agents, state_dim) for full state.
            option: Reset mode - "pose" for (x, y, theta) or "state" for full state.
            noise_seed: Seed for the per-agent LiDAR-noise RNGs. ``None`` falls
                back to ``self.seed``, which is the simulator's *construction*
                seed — and that is drawn from OS entropy when ``EnvConfig.seed``
                is ``None`` (the default), since the RNGs below need a concrete
                int. So ``noise_seed=None`` is only reproducible when the config
                carries a seed. ``F110Env.reset`` always passes a value derived
                from ``reset(seed=...)``, so the env path is unaffected and the
                noise stream is controlled by the reset seed per the gymnasium
                contract; this fallback matters only when driving
                ``F110Simulator`` directly.
        """
        if poses.shape[0] != self.num_agents:
            raise ValueError("Number of poses does not match number of agents")

        self.state.reset()
        base_seed = self.seed if noise_seed is None else int(noise_seed)
        # command-noise stream, deterministic in the reset seed (offset well away
        # from the per-agent scan seeds base_seed+idx)
        self.control_rng = np.random.default_rng(base_seed + 2 ** 20)
        if self._throttle_buffer is not None:
            self._throttle_buffer.fill(0.0)
            self._throttle_head.fill(0)
        if self.scan_enabled:
            bias_std = self.config.lidar_config.range_bias_std
            for idx in range(self.num_agents):
                self.scan_rngs[idx] = np.random.default_rng(base_seed + idx)
                # per-episode per-beam systematic bias (constant across the rollout)
                if bias_std > 0.0:
                    self.scan_bias[idx] = self.scan_rngs[idx].normal(
                        0.0, bias_std, size=self.scan_bias.shape[1]
                    ).astype(np.float32)
                else:
                    self.scan_bias[idx] = 0.0
        for i in range(self.num_agents):
            if option == "pose":
                self.state.state[i] = self.model.get_initial_state(
                    pose=poses[i], params=self.params_array
                ).astype(np.float32)
            elif option == "state":
                if poses.shape[1] != self.state_dim:
                    raise ValueError("State reset has incorrect dimension")
                self.state.state[i] = poses[i].astype(np.float32)
            else:
                raise ValueError("Unsupported reset option")
            self.state.standard_state[i] = self._standardize(self.state.state[i])
            self.state.poses[i] = np.array(
                [self.state.state[i, 0], self.state.state[i, 1], self.state.state[i, 4]],
                dtype=np.float32,
            )
            if self.config.simulation_config.compute_frenet_frame and self.track is not None:
                # anchor Frenet at the CoG (standard_state), matching the
                # observed pose_x/pose_y whatever the model's native frame
                self.state.frenet[i] = np.array(
                    self.track.cartesian_to_frenet(
                        float(self.state.standard_state[i, 0]),
                        float(self.state.standard_state[i, 1]),
                        float(self.state.standard_state[i, 4]),
                        use_s_guess=False,
                    ),
                    dtype=np.float32,
                )

        # One LiDAR sweep so the first observation is not all zeros (#15).
        if self.scan_enabled:
            self._update_scans()

    def step(self, control_inputs: np.ndarray) -> None:
        """Advance simulation by one timestep.

        Args:
            control_inputs: Control commands, shape (num_agents, 2) with
                            [steering, acceleration/speed] per agent.
        """
        if control_inputs.shape != (self.num_agents, self.control_dim):
            raise ValueError("Control input has incorrect shape")

        steer_commands = control_inputs[:, 0].astype(np.float32)
        accel_commands = control_inputs[:, 1].astype(np.float32)

        # actuator command noise (added at command time, before the lag buffers)
        if self._steer_noise_std > 0.0:
            steer_commands = steer_commands + self.control_rng.normal(
                0.0, self._steer_noise_std, self.num_agents
            ).astype(np.float32)
        if self._accl_noise_std > 0.0:
            accel_commands = accel_commands + self.control_rng.normal(
                0.0, self._accl_noise_std, self.num_agents
            ).astype(np.float32)

        # longitudinal (throttle) actuator delay
        if self._throttle_buffer is not None:
            accel_commands = self._push_throttle_delay(accel_commands)
        self.state.control_input[:, 1] = accel_commands

        if self.state.delay_buffer is not None:
            delayed_steer = self.state.push_delay(steer_commands)
        else:
            delayed_steer = steer_commands
        self.state.control_input[:, 0] = delayed_steer

        for agent_idx in range(self.num_agents):
            state = self.state.state[agent_idx]
            steer_effort = self.steering_fn(delayed_steer[agent_idx], state, self.vehicle_params)
            accel_effort = self.longitudinal_fn(accel_commands[agent_idx], state, self.vehicle_params)
            control_vector = np.array([steer_effort, accel_effort], dtype=np.float32)

            for _ in range(self.substeps):
                state = self.integrator_fn(
                    self.dynamics_fn,
                    state,
                    control_vector,
                    self.integrator_dt,
                    self.params_array,
                )
            state[4] = (state[4] + np.pi) % (2 * np.pi) - np.pi
            self.state.state[agent_idx] = state.astype(np.float32)
            self.state.standard_state[agent_idx] = self._standardize(state)
            self.state.poses[agent_idx] = np.array(
                [state[0], state[1], state[4]], dtype=np.float32
            )

        self.state.collisions.fill(0.0)
        # Before the Frenet block, so the corrected pose is what the Frenet frame,
        # the scan and the observation all see, with nothing to re-derive.
        if self.contact is not None:
            self._resolve_contacts()

        if self.config.simulation_config.compute_frenet_frame and self.track is not None:
            for agent_idx in range(self.num_agents):
                # CoG-anchored (standard_state), matching the observed pose
                std = self.state.standard_state[agent_idx]
                # Anchor the search to THIS agent's own previous s: the shared
                # Track.s_guess windows it around the previous agent instead.
                prev_s = float(self.state.frenet[agent_idx, 0])
                self.state.frenet[agent_idx] = np.array(
                    self.track.cartesian_to_frenet(
                        float(std[0]), float(std[1]), float(std[4]),
                        s_guess=prev_s, use_s_guess=True,
                    ),
                    dtype=np.float32,
                )

        if self.scan_enabled:
            self._update_scans()
        else:
            self.state.scans.fill(0.0)
        self.state.sim_time += self.time_step

    # --- Convenience accessors ---
    @property
    def agent_scans(self) -> np.ndarray:
        return self.state.scans

    @property
    def collisions(self) -> np.ndarray:
        return self.state.collisions

    @property
    def scan_num_beams(self) -> int:
        return self.config.lidar_config.num_beams if self.scan_enabled else 0

    # --- Internal helpers ---

    def _build_scan_cache(self, simulator: SegmentScanSimulator2D) -> ScanCache:
        """Build beam angles used by opponent-body ray casting."""
        num_beams = simulator.num_beams
        increment = simulator.get_increment()
        angle_min = simulator.angle_min
        beam_angles = (
            angle_min + np.arange(num_beams, dtype=np.float64) * increment
        ).astype(np.float32)
        return ScanCache(angles=beam_angles)


    def _lidar_pose_from_cog(self, pose: np.ndarray) -> np.ndarray:
        """Transform a model's CoG pose into the configured LiDAR frame."""
        tf = self.config.lidar_config.base_link_to_lidar_tf
        # Model poses are CoG-referenced, while base_link is at the rear axle.
        # ``lr`` is the longitudinal distance from the rear axle to the CoG.
        dx = float(tf[0]) - float(self.vehicle_params.lr)
        dy, dtheta = tf[1:]
        if dx == 0.0 and dy == 0.0 and dtheta == 0.0:
            return pose
        cos_yaw = math.cos(pose[2])
        sin_yaw = math.sin(pose[2])
        scan_x = pose[0] + dx * cos_yaw - dy * sin_yaw
        scan_y = pose[1] + dx * sin_yaw + dy * cos_yaw
        scan_theta = pose[2] + dtheta
        return np.array([scan_x, scan_y, scan_theta], dtype=pose.dtype)

    @staticmethod
    def _compute_collision_body_offset(
        vehicle_params: VehicleParameters,
    ) -> tuple[float, float]:
        base_dx = -float(vehicle_params.lr)
        if not math.isfinite(base_dx):
            base_dx = 0.0
        dx = base_dx + float(vehicle_params.collision_body_center_x)
        dy = float(vehicle_params.collision_body_center_y)
        return dx, dy

    def _collision_pose_from_base(self, pose: np.ndarray) -> np.ndarray:
        dx, dy = self._collision_body_dx, self._collision_body_dy
        if dx == 0.0 and dy == 0.0:
            return pose
        cos_yaw = math.cos(pose[2])
        sin_yaw = math.sin(pose[2])
        body_x = pose[0] + dx * cos_yaw - dy * sin_yaw
        body_y = pose[1] + dx * sin_yaw + dy * cos_yaw
        return np.array([body_x, body_y, pose[2]], dtype=pose.dtype)

    def _push_throttle_delay(self, values: np.ndarray) -> np.ndarray:
        """Ring-buffer delay for the longitudinal command (mirrors the steering
        delay in SimulationState.push_delay)."""
        buf = self._throttle_buffer
        head = self._throttle_head
        n = buf.shape[1]
        delayed = np.empty_like(values)
        for i in range(values.shape[0]):
            idx = head[i]
            delayed[i] = buf[i, idx]
            buf[i, idx] = values[i]
            head[i] = (idx + 1) % n
        return delayed

    def _compute_all_vertices(self) -> list:
        """Body vertices used to occlude each agent's LiDAR scan."""
        verts = []
        for i in range(self.num_agents):
            cp = self._collision_pose_from_base(self.state.poses[i])
            verts.append(get_vertices(
                np.array([cp[0], cp[1], cp[2]], dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            ))
        return verts

    def _update_scans(self) -> None:
        # LiDAR is a sensor only. Contact flags come exclusively from the
        # geometric contact solver and never rewind or freeze an agent.
        self._all_vertices = self._compute_all_vertices()
        all_vertices = self._all_vertices

        for agent_idx, simulator in enumerate(self.scan_sims):
            pose = self.state.poses[agent_idx]
            scan_pose = self._lidar_pose_from_cog(pose)

            # Clean wall scan, then opponent occlusion.
            scan_clean = simulator.scan(scan_pose, rng=None)
            cache = self.scan_cache[agent_idx]

            # Ray cast against other agents (also noise-free)
            origin = scan_pose.astype(np.float64)
            adjusted_scan = scan_clean
            for opp_idx in range(self.num_agents):
                if opp_idx == agent_idx:
                    continue
                adjusted_scan = ray_cast(origin, adjusted_scan, cache.angles, all_vertices[opp_idx])

            # Sensor noise is applied to the observed scan only: Gaussian range
            # noise + per-beam systematic bias, then random per-beam dropout
            # (no-return -> max_range).
            rng = self.scan_rngs[agent_idx]
            lidar_cfg = self.config.lidar_config
            noisy_scan = adjusted_scan + rng.normal(0.0, simulator.std_dev, size=simulator.num_beams)
            if lidar_cfg.range_bias_std > 0.0:
                noisy_scan = noisy_scan + self.scan_bias[agent_idx]
            noisy_scan = np.clip(noisy_scan, simulator.min_range, simulator.max_range)
            if lidar_cfg.dropout_prob > 0.0:
                dropped = rng.random(simulator.num_beams) < lidar_cfg.dropout_prob
                noisy_scan[dropped] = simulator.max_range
            self.state.scans[agent_idx] = noisy_scan.astype(np.float32)

    def _build_contact(self) -> None:
        """Compile the wall-contact kernels for the current track and body."""
        self.pair_contact = None
        if self.collision_check_mode is not CollisionCheckMode.SEGMENT_CONTACT:
            self.contact = None
            return
        if self.track is None:
            # No map yet, so no walls to extract; set_map builds the kernels as
            # soon as there is a track.
            self.contact = None
            return
        from .contact.adapter import build

        self.contact = build(
            self.track,
            self.vehicle_params,
            self.config.contact_config,
            self.time_step,
            self.config.domain_randomization_config,
        )
        if self.num_agents > 1:
            from .contact.adapter import BodyPairContact
            from .contact.solver import ContactParams

            cfg = self.config.contact_config
            self.pair_contact = BodyPairContact(
                ContactParams(
                    restitution=cfg.restitution,
                    friction=cfg.friction,
                    restitution_threshold=cfg.restitution_threshold,
                    baumgarte=cfg.baumgarte,
                    slop=cfg.slop,
                ),
                cfg.solver_iterations,
                cfg.device,
            )

    def _world_velocity(self, state):
        """(vx, vy, omega) in world axes for the model's native state."""
        yaw = float(state[4])
        if self.model is DynamicModel.ST:
            speed, beta, omega = float(state[3]), float(state[6]), float(state[5])
            course = yaw + beta
            return np.array([speed * math.cos(course), speed * math.sin(course)]), omega
        speed = float(state[3])
        wheelbase = float(self.vehicle_params.lf) + float(self.vehicle_params.lr)
        beta = math.atan(
            math.tan(float(state[2])) * float(self.vehicle_params.lr) / wheelbase
        )
        course = yaw + beta
        omega = speed * math.cos(beta) * math.tan(float(state[2])) / wheelbase
        return np.array([speed * math.cos(course), speed * math.sin(course)]), omega

    def _apply_contact(self, state, velocity, omega, correction):
        """Write a corrected world velocity back into the model's native state.

        ST carries slip and yaw rate, so the projection is exact. KS carries
        neither, so lateral and angular impulse components are projected onto
        its steering-defined kinematic course.
        """
        state[0] += correction[0]
        state[1] += correction[1]
        yaw = float(state[4])
        if self.model is DynamicModel.ST:
            speed = float(np.hypot(velocity[0], velocity[1]))
            course = math.atan2(velocity[1], velocity[0])
            beta = (course - yaw + math.pi) % (2 * math.pi) - math.pi
            # The tyre model is linear in beta, so the wrong branch of this
            # two-fold ambiguity produces enormous restoring moments.
            if abs(beta) > math.pi / 2:
                speed = -speed
                beta = (beta - math.copysign(math.pi, beta) + math.pi) % (2 * math.pi) - math.pi
            state[3] = speed
            state[5] = omega
            state[6] = beta
            return state
        wheelbase = float(self.vehicle_params.lf) + float(self.vehicle_params.lr)
        beta = math.atan(
            math.tan(float(state[2])) * float(self.vehicle_params.lr) / wheelbase
        )
        course = yaw + beta
        state[3] = velocity[0] * math.cos(course) + velocity[1] * math.sin(course)
        return state

    def _resolve_contacts(self) -> None:
        """Resolve wall contact for every agent and refresh the SoA mirrors."""
        for agent_idx in range(self.num_agents):
            state = self.state.state[agent_idx].astype(np.float64)
            pose = self.state.poses[agent_idx]
            verts = get_vertices(
                np.asarray(self._collision_pose_from_base(pose), dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            )
            yaw = float(state[4])
            centre = np.array([float(state[0]), float(state[1])])

            velocity, omega = self._world_velocity(state)
            velocity, omega, correction, hit = self.contact(
                verts, centre, velocity, omega,
                float(self.vehicle_params.m), float(self.vehicle_params.I),
            )
            self.state.collisions[agent_idx] = 1.0 if hit else 0.0
            if not hit:
                continue
            self._write_back(agent_idx, self._apply_contact(state, velocity, omega, correction))

        if self.pair_contact is not None:
            self._resolve_agent_contacts()

    def _resolve_agent_contacts(self) -> None:
        """Impulse-resolve every overlapping vehicle pair.

        A cheap numpy box test first, so the common case of nobody touching costs no
        JAX dispatch at all.
        """
        verts = self._compute_all_vertices()
        lows = [v.min(axis=0) for v in verts]
        highs = [v.max(axis=0) for v in verts]
        mass = float(self.vehicle_params.m)
        inertia = float(self.vehicle_params.I)

        def centre_of(state):
            return np.array([float(state[0]), float(state[1])])

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.any(highs[i] < lows[j]) or np.any(highs[j] < lows[i]):
                    continue
                state_i = self.state.state[i].astype(np.float64)
                state_j = self.state.state[j].astype(np.float64)
                c_i, c_j = centre_of(state_i), centre_of(state_j)
                v_i, w_i = self._world_velocity(state_i)
                v_j, w_j = self._world_velocity(state_j)
                v_i, w_i, v_j, w_j, sep, hit = self.pair_contact(
                    verts[i], verts[j], c_i, c_j, v_i, w_i, v_j, w_j, mass, inertia
                )
                if not hit:
                    continue
                self.state.collisions[i] = 1.0
                self.state.collisions[j] = 1.0
                self._write_back(i, self._apply_contact(state_i, v_i, w_i, -sep))
                self._write_back(j, self._apply_contact(state_j, v_j, w_j, sep))
                verts[i] = self._body_vertices(i)
                verts[j] = self._body_vertices(j)
                lows[i], highs[i] = verts[i].min(axis=0), verts[i].max(axis=0)
                lows[j], highs[j] = verts[j].min(axis=0), verts[j].max(axis=0)

    def _body_vertices(self, agent_idx):
        """Collision-body corners for one agent, from its current pose."""
        cp = self._collision_pose_from_base(self.state.poses[agent_idx])
        return get_vertices(
            np.asarray(cp, dtype=np.float64),
            self.vehicle_params.length,
            self.vehicle_params.width,
        )

    def _write_back(self, agent_idx, state) -> None:
        """Refresh the three SoA mirrors after a contact correction."""
        state[4] = (state[4] + np.pi) % (2 * np.pi) - np.pi
        self.state.state[agent_idx] = state.astype(np.float32)
        self.state.standard_state[agent_idx] = self._standardize(state)
        self.state.poses[agent_idx] = np.array(
            [state[0], state[1], state[4]], dtype=np.float32
        )
