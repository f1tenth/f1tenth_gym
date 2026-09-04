from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .dynamic_models import (
    PARAMETER_ORDER,
    VehicleParameters,
)
from .simulator import F110Simulator
from .env_config import (
    AgentTerminationMode,
    EnvConfig,
    LoopCounterMode,
    RewardMode,
)
from .integrators import integrator_from_type
from .action import (
    get_action_space,
    from_single_to_multi_action_space,
)
from .observation import ObservationType, observation_factory
from .reset import make_reset_fn
from .rendering import make_renderer
from .track import Track


_INF = float("inf")


class RenderClock:
    """Decouples rendering from stepping.

    Two clocks driven from ``F110Env.render()``: a sim-time accumulator deciding
    which steps emit a distinct rgb_array frame, and a wall-clock gate capping
    render fps. Pacing matches rtf ``sim_time / wall_time == real_time_factor``.
    """

    def __init__(self, render_fps: float, real_time_factor: float, timestep: float):
        self.render_fps = float(render_fps)
        self.frame_period = 1.0 / self.render_fps  # sim seconds (video) / wall seconds (human cap)
        self.rtf = float(real_time_factor)
        self.timestep = float(timestep)
        self.reset()

    def reset(self) -> None:
        self._accum = 0.0
        self._frame_is_new = True  # first frame after reset always emits
        self._wall_anchor = None
        self._sim_anchor = None
        self._last_draw_wall = None

    def advance(self) -> bool:
        """Advance the sim-time accumulator by one timestep. Call once per ``step()``."""
        self._accum += self.timestep
        if self._accum + 1e-9 >= self.frame_period:
            self._accum -= self.frame_period
            if self._accum >= self.frame_period:
                # timestep coarser than a frame period: emit every step, no backlog
                self._accum = 0.0
            self._frame_is_new = True
        else:
            self._frame_is_new = False
        return self._frame_is_new

    @property
    def frame_is_new(self) -> bool:
        return self._frame_is_new

    def display_due(self) -> bool:
        """Human-mode wall-clock gate: True at most ``render_fps`` times per wall-second."""
        now = time.perf_counter()
        if self._last_draw_wall is None or (now - self._last_draw_wall) >= self.frame_period - 1e-4:
            self._last_draw_wall = now
            return True
        return False

    def pace(self, sim_time: float) -> None:
        """Sleep to hold ``sim/wall == rtf`` (human modes only). No-op if ``rtf == inf``."""
        if self.rtf == _INF:
            return
        now = time.perf_counter()
        if self._wall_anchor is None:
            self._wall_anchor, self._sim_anchor = now, sim_time
            return
        target = self._wall_anchor + (sim_time - self._sim_anchor) / self.rtf
        slack = target - now
        if slack > 0:
            time.sleep(slack)
        elif slack < -0.25:
            # fell far behind (e.g. a slow step or the sim can't keep up): re-anchor
            self._wall_anchor, self._sim_anchor = now, sim_time

    def set_rtf(self, rtf: float) -> None:
        self.rtf = float(rtf)
        self._wall_anchor = None  # re-anchor so a mid-episode change has no catch-up spike


class F110Env(gym.Env):
    """
    OpenAI Gym environment for F1TENTH autonomous racing.

    Simulates 1/10th scale autonomous race cars with realistic physics,
    LiDAR sensing, and collision detection.

    Attributes:
        track: The racing track used for simulation.
        sim: The underlying F110Simulator instance.
        num_agents: Number of agents in the environment.
        ego_idx: Index of the ego agent.
        vehicle_params: Vehicle parameters for dynamics.
    """

    metadata = {"render_modes": ["human", "human_fast", "rgb_array", "unlimited"], "render_fps": 100}

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        render_mode=None,
    ):
        super().__init__()
        if isinstance(config, EnvConfig):
            resolved_config = config
        else:
            raise TypeError("config must be an EnvConfig instance")

        # `metadata` is a class attribute; copy it per-instance so that mutating
        # metadata["render_fps"] on one env never aliases another env's value.
        self.metadata = dict(type(self).metadata)

        self.env_config = resolved_config
        self.render_mode = render_mode
        self.renderer = None
        self.render_config = None
        self.render_obs = None

        self._apply_env_config()
        self._initialize_components()

    def configure(self, config: EnvConfig | None) -> None:
        """
        Reconfigure the environment with new settings.

        Args:
            config: New environment configuration, or None to skip.
        """
        if config is None:
            return
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig or None")

        self.env_config = config
        self._apply_env_config()
        self._initialize_components()

    def _apply_env_config(self) -> None:
        cfg = self.env_config

        self.seed = cfg.seed
        # re-armed on every (re)configure: a fresh config seed covers the next
        # unseeded reset again
        self._config_seed_used = False
        self.map = cfg.map_name
        self.map_scale = cfg.map_scale

        self.vehicle_params = cfg.params

        self.num_agents = cfg.num_agents
        self.ego_idx = cfg.ego_index

        self.control_cfg = cfg.control_config
        self.simulation_cfg = cfg.simulation_config
        self.observation_cfg = cfg.observation_config
        self.reset_cfg = cfg.reset_config
        self.lidar_cfg = cfg.lidar_config
        self.contact_cfg = cfg.contact_config
        self.render_cfg = cfg.render_config
        self.termination_cfg = cfg.termination_config
        self.reward_cfg = cfg.reward_config
        self.dr_cfg = cfg.domain_randomization_config

        self.max_episode_steps = self.termination_cfg.max_episode_steps
        self.terminate_on_collision = self.termination_cfg.terminate_on_collision
        self.agent_termination_mode = self.termination_cfg.agent_mode

        self.longitudinal_action_type = self.control_cfg.longitudinal_mode
        self.steer_action_type = self.control_cfg.steering_mode
        self.steer_delay_steps = self.control_cfg.steer_delay_steps

        self.timestep = self.simulation_cfg.timestep
        self.integrator_fn = integrator_from_type(self.simulation_cfg.integrator)
        self.model = self.simulation_cfg.dynamics_model
        self.loop_counter_mode = self.simulation_cfg.loop_counter
        self.compute_frenet = self.simulation_cfg.compute_frenet_frame
        self.max_laps = self.simulation_cfg.max_laps
        self.count_partial_first_lap = self.simulation_cfg.count_partial_first_lap

        self.collision_check_mode = cfg.collision_check
        self.render_enabled = cfg.render_enabled

    def _resolve_track(self) -> Track:
        map_source = self.map
        if isinstance(map_source, Track):
            return map_source
        if isinstance(map_source, (str, Path)):
            map_str = str(map_source)
            map_path = Path(map_source)
            if "/" in map_str or "\\" in map_str or map_path.suffix:
                return Track.from_track_path(map_path, track_scale=self.map_scale)
            return Track.from_track_name(map_str, track_scale=self.map_scale)
        raise TypeError("map must be a Track instance or a path/name string")

    def _initialize_components(self) -> None:
        if self.render_enabled and self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.render_config = None

        self.track = self._resolve_track()

        self.sim = F110Simulator(
            env_config=self.env_config,
            vehicle_params=self.vehicle_params,
            model=self.model,
            dynamics_fn=self.model.f_dynamics,
            integrator_fn=self.integrator_fn,
            longitudinal_type=self.longitudinal_action_type,
            steering_type=self.steer_action_type,
            track=self.track,
            seed=self.seed,
        )
        # NOTE: F110Simulator.__init__ already sets the map from `track` above
        # (and builds the scan cache from it), so no explicit set_map is needed.

        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        if self.loop_counter_mode is LoopCounterMode.FRENET_BASED:
            self.cumulative_s = np.zeros((self.num_agents,))
            self.agents_prev_s = np.zeros((self.num_agents,))
        elif self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE:
            self.cumulative_angle = np.zeros((self.num_agents,))
            self.agents_prev_angle = np.zeros((self.num_agents, 2))
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_times_finish = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        # Per-agent terminal status is episode bookkeeping, not simulator
        # state. It latches for ALL semantics but never freezes a vehicle.
        self.terminated_agents = np.zeros((self.num_agents,), dtype=np.bool_)
        # Finish-line crossings, which lead lap_counts by one under the out-lap rule.
        self._line_crossings = np.zeros((self.num_agents,))
        # per-agent previous Frenet s for the progress reward (independent of the
        # lap counter so it works under any LoopCounterMode)
        self._reward_prev_s = np.zeros((self.num_agents,))
        self.sim_time = 0.0

        if self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE and self.track is not None:
            cl = self.track.centerline
            # Polygon area centroid (Shoelace formula) — true centroid of the enclosed area,
            # guaranteed inside for convex tracks and robust for typical racing circuits.
            xs = np.asarray(cl.xs, dtype=np.float64)
            ys = np.asarray(cl.ys, dtype=np.float64)
            x = np.append(xs, xs[0])
            y = np.append(ys, ys[0])
            cross = x[:-1] * y[1:] - x[1:] * y[:-1]
            area = 0.5 * np.sum(cross)
            cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * area)
            cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * area)
            self._winding_point = np.array([float(cx), float(cy)])
            # Determine CW vs CCW from the first two raceline points relative to winding point.
            rl = self.track.raceline
            fp = np.array([rl.xs[0], rl.ys[0]]) - self._winding_point
            sp = np.array([rl.xs[1], rl.ys[1]]) - self._winding_point
            self._winding_direction = float(np.sign(
                np.arctan2(fp[0] * sp[1] - fp[1] * sp[0], fp.dot(sp))
            ))
        else:
            self._winding_point = None
            self._winding_direction = 1.0

        # Spaces are fixed for the env's lifetime but DR redraws params each
        # episode, so build them from the widest params across the DR ranges.
        self.space_vehicle_params = self.dr_cfg.widest_params(self.vehicle_params)

        obs_kwargs: dict[str, Any] = {"type": self.observation_cfg.type}
        if self.observation_cfg.features is not None:
            obs_kwargs["features"] = self.observation_cfg.features
        self.observation_type = observation_factory(env=self, **obs_kwargs)
        self.observation_space = self.observation_type.space()
        # Built on first use: render callbacks always see the DEFAULT vocabulary,
        # so this is only read when the configured type is not DEFAULT.
        self._render_obs_type = None
        self.render_obs = None

        single_action_space = get_action_space(
            self.longitudinal_action_type,
            self.steer_action_type,
            self.space_vehicle_params,
        )
        self.action_space = from_single_to_multi_action_space(
            single_action_space, self.num_agents
        )

        self.reset_fn = make_reset_fn(
            track=self.track,
            num_agents=self.num_agents,
            type=self.reset_cfg.strategy,
            **self.reset_cfg.reset_kwargs(),
        )

        # RecordVideo *container* framerate. One frame is captured per step, so
        # real-time playback needs round(1/timestep), not render_config.render_fps.
        base_fps = int(round(1.0 / self.timestep)) if self.timestep > 0 else 0
        self.metadata["render_fps"] = base_fps

        # render_mode picks the initial real-time factor; plain "human" uses the
        # configured value, "human_fast" is legacy 10x sugar, "unlimited" is free-run.
        if self.render_mode == "human_fast":
            initial_rtf = 10.0
        elif self.render_mode == "unlimited":
            initial_rtf = float("inf")
        else:
            initial_rtf = self.render_cfg.real_time_factor

        self._render_clock = RenderClock(
            render_fps=self.render_cfg.render_fps,
            real_time_factor=initial_rtf,
            timestep=self.timestep,
        )
        self._last_frame = None
        self._elapsed_steps = 0

        if self.render_enabled:
            self.renderer, self.render_config = make_renderer(
                params=self.vehicle_params,
                track=self.track,
                agent_ids=self.agent_ids,
                render_mode=self.render_mode,
                render_config=self.render_cfg,
            )
        else:
            self.renderer = None
            self.render_config = None

    def _record_crossing(self, ind: int, crossings: int) -> None:
        """Bank a finish-line crossing for one agent.

        Every crossing splits the lap time, so the out-lap rule still times full
        circuits; only the lap count skips the partial first lap.
        """
        if crossings <= self._line_crossings[ind] or self.sim_time <= self.timestep:
            return
        self._line_crossings[ind] = crossings
        split = self.sim_time - self.lap_times_finish[ind]
        self.lap_times_finish[ind] = self.sim_time
        laps = crossings if self.count_partial_first_lap else crossings - 1
        if laps > self.lap_counts[ind]:
            self.lap_counts[ind] = laps
            self.lap_times[ind] = split

    def _check_done(self):
        """
        Check if the current rollout is done
        """
        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.compute_frenet
            and self.track is not None
        ):
            s_frame_max = self.track.centerline.spline.s_frame_max
            for ind in range(self.num_agents):
                current_s = float(self.sim.state.frenet[ind, 0])
                delta_s = current_s - self.agents_prev_s[ind]
                # Correct for wraparound: a forward crossing makes delta_s
                # sharply negative; a backward crossing makes it sharply positive.
                if delta_s < -0.5 * s_frame_max:
                    delta_s += s_frame_max
                elif delta_s > 0.5 * s_frame_max:
                    delta_s -= s_frame_max
                self.cumulative_s[ind] += delta_s
                self.agents_prev_s[ind] = current_s
                # cumulative_s is seeded with the spawn arclength at reset, so this
                # counts crossings of the s=0 datum, not loops back to the spawn.
                self._record_crossing(ind, int(self.cumulative_s[ind] / s_frame_max))

        elif (
            self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE
            and self._winding_point is not None
        ):
            wp = self._winding_point
            wd = self._winding_direction
            for ind in range(self.num_agents):
                pose = self.sim.state.poses[ind]
                curr_vec = np.array([pose[0] - wp[0], pose[1] - wp[1]])
                prev_vec = self.agents_prev_angle[ind]  # stored as 2-vector
                # Signed angle from prev to curr, scaled by racing direction
                cross = float(prev_vec[0] * curr_vec[1] - prev_vec[1] * curr_vec[0])
                dot = float(prev_vec.dot(curr_vec))
                delta_angle = np.arctan2(cross, dot) * wd
                self.cumulative_angle[ind] += delta_angle
                self.agents_prev_angle[ind] = curr_vec
                # Seeded at reset with the angle from the s=0 ray, so a full 2pi is
                # a crossing of that ray rather than a loop back to the spawn.
                self._record_crossing(ind, int(self.cumulative_angle[ind] / (2.0 * np.pi)))

        terminal_now = np.zeros((self.num_agents,), dtype=np.bool_)
        if self.terminate_on_collision:
            terminal_now |= np.asarray(self.sim.collisions, dtype=np.bool_)
        if self.max_laps is not None:
            terminal_now |= self.lap_counts >= self.max_laps
        self.terminated_agents |= terminal_now

        if self.agent_termination_mode is AgentTerminationMode.EGO:
            return bool(self.terminated_agents[self.ego_idx])
        if self.agent_termination_mode is AgentTerminationMode.ANY:
            return bool(np.any(self.terminated_agents))
        if self.agent_termination_mode is AgentTerminationMode.ALL:
            return bool(np.all(self.terminated_agents))
        raise RuntimeError(
            f"Unsupported agent termination mode: {self.agent_termination_mode!r}"
        )

    def _sample_vehicle_params(self):
        """Draw a randomized VehicleParameters between the DR bounds (env RNG).

        One vectorized draw over every finite field, so the RNG stream does not
        depend on which fields vary. Non-finite fields (the multi-body block on
        the small-scale presets) pass through untouched: ``uniform`` rejects NaN.
        """
        low, high = self.dr_cfg.bounds_arrays()
        drawn = low.copy()
        finite = np.isfinite(low) & np.isfinite(high)
        drawn[finite] = self.np_random.uniform(low[finite], high[finite])
        return VehicleParameters(
            **{name: float(value) for name, value in zip(PARAMETER_ORDER, drawn)}
        )

    def _compute_progress(self) -> np.ndarray:
        """Per-agent forward Frenet arclength progress (metres) since the last
        step, wrap-corrected. Zeros when the Frenet frame is not computed."""
        progress = np.zeros(self.num_agents)
        if not (self.compute_frenet and self.track is not None):
            return progress
        s_max = self.track.centerline.spline.s_frame_max
        for ind in range(self.num_agents):
            s = float(self.sim.state.frenet[ind, 0])
            ds = s - self._reward_prev_s[ind]
            if ds < -0.5 * s_max:
                ds += s_max
            elif ds > 0.5 * s_max:
                ds -= s_max
            progress[ind] = ds
            self._reward_prev_s[ind] = s
        return progress

    def _compute_reward(self, obs, action, progress, info, terminated, truncated) -> float:
        cfg = self.reward_cfg
        if cfg.mode is RewardMode.CUSTOM:
            return float(cfg.reward_fn(obs, action, info, terminated, truncated))
        if cfg.mode is RewardMode.SURVIVAL:
            return self.timestep
        # PROGRESS
        ego = self.ego_idx
        reward = cfg.progress_weight * float(progress[ego])
        if cfg.velocity_weight:
            reward += cfg.velocity_weight * float(self.sim.state.standard_state[ego, 3])
        if cfg.timestep_weight:
            reward += cfg.timestep_weight * self.timestep
        if cfg.collision_penalty and self.sim.collisions[ego] > 0:
            reward -= cfg.collision_penalty
        return reward

    def step(self, action):
        """Advance the simulation by one timestep.

        Args:
            action: Array of shape ``(num_agents, 2)``, columns
                ``[steering, longitudinal]`` — steering FIRST. Neither column
                is clipped against the action space; only the shape is checked.

        Returns:
            The gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``:
            ``obs`` per the configured observation type; ``reward`` per
            ``RewardConfig`` (SURVIVAL: the timestep; PROGRESS: weighted Frenet
            progress; CUSTOM: your ``reward_fn``); ``terminated`` on collision
            or ``max_laps``; ``truncated`` when ``max_episode_steps`` is
            reached; ``info`` with ``lap_times``, ``lap_counts``, ``sim_time``,
            ``collisions``, ``terminated_agents`` and ``progress`` (all copies).
        """

        # call simulation step
        self.sim.step(action)
        self._elapsed_steps += 1

        # advance the render clock's sim-time frame accumulator (drives render fps); decoupled from how often the user calls render()).
        self._render_clock.advance()

        # check done
        terminated = self._check_done()

        # per-agent forward progress this step (also feeds the progress reward)
        progress = self._compute_progress()

        # observation
        obs = self.observation_type.observe()
        if self.render_enabled and self.renderer is not None:
            if self.observation_cfg.type is ObservationType.DEFAULT:
                self.render_obs = copy.deepcopy(obs)
            else:
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        self.sim_time = self.sim.state.sim_time
        truncated = (
            self.max_episode_steps is not None
            and self._elapsed_steps >= self.max_episode_steps
        )
        # copy: these are the env's live arrays, mutated every step; without a
        # copy a stored info dict would change retroactively (breaks logging).
        info = {
            "lap_times": self.lap_times.copy(),
            "lap_counts": self.lap_counts.copy(),
            "sim_time": self.sim_time,
            "collisions": self.sim.collisions.copy(),  # per-agent collision flags
            "terminated_agents": self.terminated_agents.copy(),
            "progress": progress,  # per-agent forward arclength this step (m)
        }
        # reward is computed last so a CUSTOM reward_fn sees the final info
        reward = self._compute_reward(obs, action, progress, info, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args:
            seed: Seeds ``np_random`` (spawn draw, domain randomization, LiDAR
                noise). ``None`` uses ``EnvConfig.seed`` for the first reset if
                set, else OS entropy.
            options: Optional spawn override. ``{"poses": array (n, 3)}``
                places agents at ``[x, y, theta]`` in the model's native frame
                (zero speed and steering). ``{"states": array (n, state_dim)}``
                writes full native state rows verbatim — the only way to spawn
                at speed. Without either, the configured reset strategy samples
                the poses.

        Returns:
            ``(obs, info)``: the first observation (with a real LiDAR sweep)
            and an info dict with ``lap_times``, ``lap_counts``, ``sim_time``
            and ``terminated_agents`` — note ``collisions``/``progress``
            appear only in ``step()``'s info.
        """
        # EnvConfig.seed covers the FIRST unseeded reset; later ones continue the
        # stream. An explicit seed always wins.
        if seed is None and self.seed is not None and not self._config_seed_used:
            seed = int(self.seed)
        self._config_seed_used = True
        super().reset(seed=seed)

        self.sim_time = 0.0
        self._elapsed_steps = 0
        self._render_clock.reset()
        self._last_frame = None
        if self.loop_counter_mode is LoopCounterMode.FRENET_BASED:
            self.cumulative_s.fill(0.0)
            self.agents_prev_s.fill(0.0)
        elif self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE:
            self.cumulative_angle.fill(0.0)
            self.agents_prev_angle.fill(0.0)
        self.lap_counts.fill(0.0)
        self.terminated_agents.fill(False)
        self.lap_times.fill(0.0)
        self.lap_times_finish.fill(0.0)
        self._line_crossings.fill(0.0)
        if options is not None and "poses" in options:
            poses = options["poses"]
            option = "pose"
        elif options is not None and "states" in options:
            poses = options["states"]
            option = "state"
        else:
            poses = self.reset_fn.sample(self.np_random)
            option = "pose"


        if option == "pose":
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                3,
            ), "Initial poses must be a numpy array of shape (num_agents, 3)"
        elif option == "state":
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                self.model.state_dim,
            ), f"Initial full state must be a numpy array of shape (num_agents, {self.model.state_dim})"
        else:
            raise ValueError("Invalid reset option.")

        # Domain randomization: sample vehicle params for this episode (from the
        # env RNG, so it is reproducible with reset(seed=...)) and push them to
        # the simulator and renderer.
        if self.dr_cfg.randomized_fields():
            episode_params = self._sample_vehicle_params()
            self.sim.update_params(episode_params)
            if self.renderer is not None:
                self.renderer.update_params(episode_params)

        # Derive the LiDAR-noise seed from the env RNG (which gymnasium seeds
        # from reset(seed=...)), so the noise stream is controlled by the reset
        # seed rather than being byte-identical every episode.
        noise_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim.reset(poses, option=option, noise_seed=noise_seed)

        # seed the progress-reward reference from the spawn arclength so the
        # first step's progress is ~0, not the whole spawn s
        if self.compute_frenet and self.track is not None:
            self._reward_prev_s[:] = self.sim.state.frenet[:, 0]

        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.compute_frenet
            and self.track is not None
        ):
            # prev_s stops the first delta being the whole spawn arclength; seeding
            # cumulative_s with it makes a lap a crossing of s=0, not a loop.
            self.agents_prev_s[:] = self.sim.state.frenet[:, 0]
            self.cumulative_s[:] = self.sim.state.frenet[:, 0]

        if (
            self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE
            and self._winding_point is not None
        ):
            wp = self._winding_point
            wd = self._winding_direction
            cl = self.track.centerline
            datum = np.array([float(cl.xs[0]) - wp[0], float(cl.ys[0]) - wp[1]])
            for ind in range(self.num_agents):
                pose = self.sim.state.poses[ind]
                spawn_vec = np.array([pose[0] - wp[0], pose[1] - wp[1]])
                self.agents_prev_angle[ind] = spawn_vec
                # Angle already travelled from the s=0 ray, so the first 2pi lands on
                # that ray instead of back at the spawn bearing.
                cross = float(datum[0] * spawn_vec[1] - datum[1] * spawn_vec[0])
                dot = float(datum.dot(spawn_vec))
                self.cumulative_angle[ind] = float(np.arctan2(cross, dot) * wd) % (2.0 * np.pi)

        obs = self.observation_type.observe()
        if self.render_enabled and self.renderer is not None:
            if self.observation_cfg.type is ObservationType.DEFAULT:
                self.render_obs = copy.deepcopy(obs)
            else:
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        # copy: these are the env's live arrays, mutated every step; without a
        # copy a stored info dict would change retroactively (breaks logging).
        info = {
            "lap_times": self.lap_times.copy(),
            "lap_counts": self.lap_counts.copy(),
            "sim_time": self.sim_time,
            "terminated_agents": self.terminated_agents.copy(),
        }

        return obs, info

    def update_map(self, map_name: Track | str):
        """
        Updates the map used by simulation

        Args:
            map_name (Track | str): name of the map, path to map, or Track instance

        Returns:
            None
        """
        new_config = self.env_config.with_updates(map_name=map_name)
        self.configure(new_config)

    def update_params(self, params, index=-1):
        """
        Update the shared vehicle parameters used by the simulator and renderers.

        Args:
            params (VehicleParameters): new vehicle parameters.
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        if index >= 0:
            raise NotImplementedError(
                "Per-agent parameter updates are not supported in the simplified simulator"
            )
        if isinstance(params, VehicleParameters):
            vehicle_params = params
        else:
            raise TypeError("params must be a VehicleParameters instance")

        # Validate BEFORE mutating anything.
        new_config = self.env_config.with_updates(params=vehicle_params)

        self.vehicle_params = vehicle_params
        self.env_config = new_config

        self.sim.update_params(self.vehicle_params)
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.update_params(self.vehicle_params)
        self.space_vehicle_params = self.dr_cfg.widest_params(self.vehicle_params)
        if hasattr(self, "action_space"):
            single_action_space = get_action_space(
                self.longitudinal_action_type,
                self.steer_action_type,
                self.space_vehicle_params,
            )
            self.action_space = from_single_to_multi_action_space(
                single_action_space, self.num_agents
            )
        if hasattr(self, "observation_space"):
            self.observation_space = self.observation_type.space()

    @property
    def render_obs_type(self):
        """The DEFAULT-vocabulary observation handed to render callbacks.

        Lazy: under the default observation type ``step``/``reset`` deep-copy the
        real observation instead, so this is never built for the common config.
        """
        if self._render_obs_type is None:
            self._render_obs_type = observation_factory(
                env=self, type=ObservationType.DEFAULT
            )
        return self._render_obs_type

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """
        if self.render_enabled and self.renderer is not None:
            self.renderer.add_renderer_callback(callback_func)

    def set_real_time_factor(self, real_time_factor: float) -> None:
        """Change the real-time factor at runtime (mid-episode is fine).

        Sim-seconds per wall-second in the human render modes (``inf`` = free-run);
        no effect on physics or rgb_array output. The pacer re-anchors on change,
        so there is no catch-up sleep or fast-forward spike.

        Args:
            real_time_factor: positive float, or ``float("inf")`` for free-run.
        """
        if not (real_time_factor > 0):
            raise ValueError(
                f"real_time_factor must be > 0 (or float('inf')), got {real_time_factor}"
            )
        self._render_clock.set_rtf(real_time_factor)

    @property
    def real_time_factor(self) -> float:
        """Current real-time factor (see :meth:`set_real_time_factor`)."""
        return self._render_clock.rtf

    @property
    def render_fps(self) -> float:
        """Target fixed frame rate (display cap in human modes, emit cadence in rgb_array)."""
        return self._render_clock.render_fps

    @property
    def frame_is_new(self) -> bool:
        """True on steps that emit a distinct frame at the render_fps sim-time cadence.

        Use this for manual, exact-fps video capture (append a frame only when
        it is True) rather than relying on RecordVideo's one-frame-per-step.
        """
        return self._render_clock.frame_is_new

    def render(self, mode=None):
        """
        Render the current state. Rendering is decoupled from stepping by an
        internal render clock (see ``RenderClock``): the frame rate is fixed by
        ``render_config.render_fps`` and the pace is set by the real-time factor,
        both independent of how fast the user steps the simulation.

        Args:
            mode (str, optional): overrides the mode set at ``gym.make`` for this
                call. Defaults to the configured ``render_mode``. One of
                "human", "human_fast", "unlimited", "rgb_array".

        Returns:
            np.ndarray | None: an RGB frame (H, W, 3) in "rgb_array" mode; None in
            the human modes.
        """
        m = mode or self.render_mode
        if (
            m not in self.metadata["render_modes"]
            or not self.render_enabled
            or self.renderer is None
        ):
            return None

        clk = self._render_clock

        if m in ("human", "human_fast", "unlimited"):
            # Wall-clock gate: draw at most render_fps times/second regardless of
            # how fast the sim is stepped. Then pace the loop to the real-time factor.
            if clk.display_due():
                self.renderer.update(obs=self.render_obs)
                self.renderer.render()
            clk.pace(self.sim_time)
            return None

        # rgb_array: never sleep, always return a frame. Grab a fresh one only on
        # the sim-time cadence and cache it between, so RecordVideo stays smooth.
        if clk.frame_is_new or self._last_frame is None:
            self.renderer.update(obs=self.render_obs)
            self._last_frame = self.renderer.render()
        return self._last_frame

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.render_enabled and self.renderer is not None:
            self.renderer.close()
        super().close()
