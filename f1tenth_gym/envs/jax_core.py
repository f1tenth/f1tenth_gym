"""Composed device-resident reset and transition for the functional gym."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .contact.functional import ContactParams, WallContactConfig
from .dynamic_models.jax_core import (
    DynamicsConfig,
    DynamicsRuntimeParams,
    DynamicsState,
    make_dynamics_state,
    model_state_from_poses,
    step_dynamics,
)
from .dynamic_models.jax import standardize_state
from .episode import (
    BuiltinRewardMode,
    EpisodeConfig,
    EpisodeEvents,
    EpisodeMetrics,
    EpisodeParams,
    EpisodeState,
    EpisodeStatus,
    advance_episode,
    reset_episode_state,
)
from .contact.geometry import BodyParams
from .lidar.functional import (
    ScanConfig,
    ScanParams,
    ScanState,
    clean_scan,
    observed_scan,
    reset_scan_state,
)
from .contact.pairs import PairContactConfig, PairTable, resolve_contacts
from .reset.functional import (
    ResetSamplingConfig,
    ResetTable,
    sample_reset_poses,
)
from .track.functional import (
    FrenetProjectionConfig,
    TrackTable,
    cartesian_to_frenet,
    cartesian_to_frenet_local,
)


@dataclass(frozen=True)
class CoreConfig:
    """Hashable structural choices for one compiled environment topology."""

    dynamics: DynamicsConfig
    reset: ResetSamplingConfig
    scan: ScanConfig
    wall_contact: WallContactConfig
    pair_contact: PairContactConfig
    episode: EpisodeConfig
    frenet: FrenetProjectionConfig = FrenetProjectionConfig()
    contact_enabled: bool = True
    scan_enabled: bool = True
    frenet_enabled: bool = True

    def __post_init__(self) -> None:
        counts = {
            self.dynamics.num_agents,
            self.reset.num_agents,
            self.scan.num_agents,
            self.wall_contact.num_agents,
            self.pair_contact.num_agents,
            self.episode.num_agents,
        }
        if len(counts) != 1:
            raise ValueError("every core config must use the same num_agents")
        if (
            self.wall_contact.state_dim != self.dynamics.state_dim
            or self.pair_contact.state_dim != self.dynamics.state_dim
        ):
            raise ValueError("dynamics and contact configs must use the same state_dim")
        for name in ("contact_enabled", "scan_enabled", "frenet_enabled"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if (
            not self.frenet_enabled
            and self.episode.reward_mode is BuiltinRewardMode.PROGRESS
        ):
            raise ValueError("PROGRESS reward requires frenet_enabled=True")
        if not self.scan_enabled and self.scan.num_beams != 1:
            raise ValueError(
                "disabled sensing requires a one-beam internal placeholder"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoreTables:
    """Fixed-shape device tables produced by host preprocessing."""

    reset: ResetTable
    track: TrackTable
    pairs: PairTable


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoreParams:
    """Traced physical, sensor, contact, and episode values."""

    dynamics: DynamicsRuntimeParams
    body: BodyParams
    contact: ContactParams
    scan: ScanParams
    episode: EpisodeParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoreState:
    """Complete immutable carry for one functional environment."""

    dynamics: DynamicsState
    scan: ScanState
    episode: EpisodeState
    scans: jax.Array
    collisions: jax.Array
    observation_sim_time: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoreObservation:
    """Canonical agent-batched values from which adapters package fields."""

    scans: jax.Array
    state: jax.Array
    standard_state: jax.Array
    collisions: jax.Array
    frenet: jax.Array
    lap_times: jax.Array
    lap_counts: jax.Array
    sim_time: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoreMetrics:
    """Post-transition episode measurements and global end status."""

    episode: EpisodeMetrics
    status: EpisodeStatus


def observe_core(state: CoreState) -> CoreObservation:
    """Build the canonical observation vocabulary without recomputing physics."""
    model = state.dynamics.model
    standard = jax.vmap(standardize_state)(model)
    dtype = model.dtype
    return CoreObservation(
        scans=jnp.asarray(state.scans, dtype=dtype),
        state=model,
        standard_state=standard,
        collisions=state.collisions.astype(dtype),
        frenet=state.episode.frenet,
        lap_times=state.episode.lap_times,
        lap_counts=state.episode.lap_counts.astype(dtype),
        sim_time=jnp.asarray(state.observation_sim_time, dtype=dtype),
    )


def _reset_frenet(
    poses: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
) -> jax.Array:
    if config.frenet_enabled:
        return jax.vmap(
            lambda pose: cartesian_to_frenet(tables.track.centerline, pose)
        )(poses)
    return jnp.zeros((config.dynamics.num_agents, 3), dtype=poses.dtype)


def _step_frenet(
    model: jax.Array,
    previous: EpisodeState,
    tables: CoreTables,
    config: CoreConfig,
) -> jax.Array:
    if config.frenet_enabled:
        poses = model[:, jnp.asarray((0, 1, 4))]
        return jax.vmap(
            lambda pose, prior: cartesian_to_frenet_local(
                tables.track.centerline,
                pose,
                prior,
                config.frenet,
            )
        )(poses, previous.frenet[:, 0])
    return jnp.zeros_like(previous.frenet)


def _observed_scans(
    key: jax.Array,
    model: jax.Array,
    scan_state: ScanState,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> jax.Array:
    if config.scan_enabled:
        clean = clean_scan(
            model,
            tables.track,
            params.body,
            config.scan,
            params.scan,
            params.dynamics.vehicle.lr,
        )
        return observed_scan(
            key,
            clean,
            scan_state,
            config.scan,
            params.scan,
        )
    return jnp.zeros(
        (config.dynamics.num_agents, config.scan.num_beams),
        dtype=model.dtype,
    )


def _finish_reset(
    bias_key: jax.Array,
    scan_key: jax.Array,
    dynamics: DynamicsState,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> tuple[CoreObservation, CoreState]:
    """Initialize every non-dynamics carry from one zero-clock model state."""
    poses = dynamics.model[:, jnp.asarray((0, 1, 4))]
    frenet = _reset_frenet(poses, tables, config)
    episode = reset_episode_state(frenet, config.episode)
    scan_state = reset_scan_state(
        bias_key,
        config.scan,
        params.scan,
        dtype=dynamics.model.dtype,
    )
    scans = _observed_scans(
        scan_key,
        dynamics.model,
        scan_state,
        tables,
        config,
        params,
    )
    state = CoreState(
        dynamics=dynamics,
        scan=scan_state,
        episode=episode,
        scans=scans,
        collisions=jnp.zeros((config.dynamics.num_agents,), dtype=jnp.bool_),
        observation_sim_time=dynamics.sim_time,
    )
    return observe_core(state), state


def reset_core(
    key: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> tuple[CoreObservation, CoreState]:
    """Sample and initialize one environment, including its first real scan."""
    pose_key, bias_key, scan_key = jax.random.split(key, 3)
    poses = sample_reset_poses(pose_key, tables.reset, config.reset)
    model = model_state_from_poses(poses, config.dynamics)
    dynamics = make_dynamics_state(model, config.dynamics)
    return _finish_reset(
        bias_key,
        scan_key,
        dynamics,
        tables,
        config,
        params,
    )


def reset_core_from_poses(
    key: jax.Array,
    poses: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> tuple[CoreObservation, CoreState]:
    """Initialize from explicit CoG poses ``[x, y, yaw]`` with zero motion."""
    poses = jnp.asarray(poses)
    expected = (config.dynamics.num_agents, 3)
    if poses.shape != expected:
        raise ValueError(f"poses must have shape {expected}, got {poses.shape}")
    poses = poses.astype(tables.reset.waypoints.dtype)
    model = model_state_from_poses(poses, config.dynamics)
    dynamics = make_dynamics_state(model, config.dynamics)
    _pose_key, bias_key, scan_key = jax.random.split(key, 3)
    return _finish_reset(
        bias_key,
        scan_key,
        dynamics,
        tables,
        config,
        params,
    )


def reset_core_from_state(
    key: jax.Array,
    model_state: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> tuple[CoreObservation, CoreState]:
    """Initialize from complete native KS/ST state rows without alteration."""
    model = jnp.asarray(model_state)
    expected = (config.dynamics.num_agents, config.dynamics.state_dim)
    if model.shape != expected:
        raise ValueError(
            f"model_state must have shape {expected}, got {model.shape}"
        )
    dynamics = make_dynamics_state(
        model.astype(tables.reset.waypoints.dtype), config.dynamics
    )
    _pose_key, bias_key, scan_key = jax.random.split(key, 3)
    return _finish_reset(
        bias_key,
        scan_key,
        dynamics,
        tables,
        config,
        params,
    )


def step_core(
    key: jax.Array,
    state: CoreState,
    actions: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    params: CoreParams,
) -> tuple[
    CoreObservation,
    CoreState,
    jax.Array,
    EpisodeEvents,
    CoreMetrics,
]:
    """Compose actuation, physics, sensing, and episode semantics once."""
    dynamics_key, scan_key = jax.random.split(key)
    dynamics = step_dynamics(
        dynamics_key,
        state.dynamics,
        actions,
        config.dynamics,
        params.dynamics,
    )
    if config.contact_enabled:
        model, collisions = resolve_contacts(
            dynamics.model,
            tables.track,
            tables.pairs,
            params.body,
            params.dynamics.vehicle,
            params.contact,
            params.dynamics.timestep,
            config.wall_contact,
            config.pair_contact,
        )
    else:
        model = dynamics.model
        collisions = jnp.zeros(
            (config.dynamics.num_agents,), dtype=jnp.bool_
        )
    dynamics = replace(dynamics, model=model)
    frenet = _step_frenet(model, state.episode, tables, config)
    scans = _observed_scans(
        scan_key,
        model,
        state.scan,
        tables,
        config,
        params,
    )
    episode_params = params.episode
    if not config.frenet_enabled:
        # No winding-angle implementation exists in the functional core.  A
        # disabled Frenet frame therefore has no lap counter; collision and
        # timeout policies remain active, while lap termination is explicit off.
        episode_params = replace(episode_params, lap_limit_enabled=False)
    episode, rewards, events, metrics, status = advance_episode(
        state.episode,
        frenet,
        collisions,
        model[:, 3],
        tables.track.centerline.length,
        state.dynamics.sim_time,
        dynamics.sim_time,
        params.dynamics.timestep,
        config.episode,
        episode_params,
    )
    next_state = CoreState(
        dynamics=dynamics,
        scan=state.scan,
        episode=episode,
        scans=scans,
        collisions=collisions,
        observation_sim_time=state.dynamics.sim_time,
    )
    return (
        observe_core(next_state),
        next_state,
        rewards,
        events,
        CoreMetrics(episode=metrics, status=status),
    )


__all__ = [
    "CoreConfig",
    "CoreMetrics",
    "CoreObservation",
    "CoreParams",
    "CoreState",
    "CoreTables",
    "observe_core",
    "reset_core",
    "reset_core_from_poses",
    "reset_core_from_state",
    "step_core",
]
