Configuration
=============

:class:`~f1tenth_gym.envs.f110_env.F110Env` takes exactly one constructor
argument — a frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig` — and
anything else (a dict, ``None``, loose keyword arguments) raises ``TypeError``.
Every default is readable off the config object before an environment exists,
and so is the type check on each of the nine nested sections:

>>> from f1tenth_gym.envs.env_config import EnvConfig
>>> cfg = EnvConfig()
>>> cfg.num_agents, cfg.map_name, cfg.seed
(1, 'Spielberg', None)
>>> EnvConfig(control_config={"steer_delay_steps": 2})
Traceback (most recent call last):
    ...
TypeError: control must be a ControlConfig instance

The error names each section by a short name (``control``), not by the actual
keyword argument (``control_config``).

Top-level fields
----------------

``EnvConfig`` declares 17 fields; nine of them are config objects with a
section of their own further down.

.. list-table::
   :header-rows: 1
   :widths: 25 28 47

   * - Field
     - Default
     - Meaning / validation
   * - ``seed``
     - ``None``
     - Coerced to ``int`` when set; the first *unseeded* ``reset()`` then
       behaves as ``reset(seed=seed)``, making the whole run a function of
       the config. An explicit ``reset(seed=...)`` always wins, and ``None``
       seeds from OS entropy. See :doc:`reproducibility`.
   * - ``map_name``
     - ``"Spielberg"``
     - A track name (downloaded on first use), a path — track directory,
       stem, or YAML file, in either naming convention — or a prebuilt
       ``Track`` instance, the fast path for vectorized envs. Not validated
       here; a bad value fails at ``gym.make``. See :doc:`tracks`.
   * - ``map_scale``
     - ``1.0``
     - Coerced to ``float``; must be ``> 0``. See :doc:`tracks` for map/world
       scaling.
   * - ``params``
     - ``F1TENTH_VEHICLE_PARAMETERS``
     - Must be a ``VehicleParameters`` instance — the first check to run.
       See :doc:`dynamics`.
   * - ``num_agents``
     - ``1``
     - Coerced to ``int``; must be ``>= 1``.
   * - ``ego_index``
     - ``0``
     - Coerced to ``int``; must satisfy ``0 <= ego_index < num_agents``.
   * - ``control_config``
     - ``ControlConfig()``
     - Action interpretation, actuator lag and noise.
   * - ``simulation_config``
     - ``SimulationConfig()``
     - Physics clock, integrator, dynamics model, lap rule.
   * - ``observation_config``
     - ``ObservationConfig()``
     - Observation preset or custom field tuple.
   * - ``reset_config``
     - ``ResetConfig()``
     - Spawn strategy, spacing, reference line.
   * - ``lidar_config``
     - ``LiDARConfig()``
     - LiDAR geometry and noise.
   * - ``render_config``
     - ``RenderConfig()``
     - Rendering pacing and frame output.
   * - ``termination_config``
     - ``TerminationConfig()``
     - Termination and truncation rules.
   * - ``reward_config``
     - ``RewardConfig()``
     - Per-step reward.
   * - ``domain_randomization_config``
     - ``DomainRandomizationConfig()``
     - Per-episode vehicle-parameter randomization.
   * - ``collision_check``
     - ``CollisionCheckMode.SEGMENT_CONTACT``
     - ``SEGMENT_CONTACT`` applies geometric wall/body contact response;
       ``NONE`` disables collision detection and response. Raw integer values
       are coerced through the enum. Import it from
       ``f1tenth_gym.envs.collision_models``.
   * - ``render_enabled``
     - ``True``
     - Coerced to ``bool``. When ``False`` no renderer is built:
       ``render()`` returns ``None`` and ``add_render_callback`` does
       nothing, whatever ``render_mode`` was passed to ``gym.make``.

Deriving a config
-----------------

Every config class is a frozen dataclass: nothing is mutated in place, and
``with_updates(**changes)`` returns a fresh copy whose ``__post_init__``
re-validates it. ``with_updates`` replaces only top-level fields, so changing
a field inside a nested section means rebuilding that section too:

>>> base = EnvConfig()
>>> cfg = base.with_updates(
...     simulation_config=base.simulation_config.with_updates(max_laps=None)
... )
>>> cfg.simulation_config.max_laps, base.simulation_config.max_laps
(None, 1)

The default ``max_laps=1`` terminates after a single lap; ``None`` is the
endless-rollout setting for RL training and long evaluations.

Nothing is re-exported from the package roots — ``from f1tenth_gym import
EnvConfig`` and ``from f1tenth_gym.envs import EnvConfig`` both raise
``ImportError``. Import from the defining modules: every class on this page
lives in ``f1tenth_gym.envs.env_config`` except ``LiDARConfig``
(``f1tenth_gym.envs.lidar``), and the enums come from the modules named in
each section.

When validation runs
--------------------

A constructed ``EnvConfig`` proves less than it appears to. Checks are split
across three tiers, and only the first fires at construction time:

1. Each dataclass's ``__post_init__`` — every rule quoted in the tables on
   this page.
2. ``F110Simulator.__init__`` — the substep rule below.
3. Component construction inside ``gym.make`` — the reset-strategy check, the
   action-space bounds, the observation factory (a ``FEATURES`` tuple naming
   a dropped field raises here), and, when a renderer is built, an unknown
   ``focus_on`` id.

Tiers 2 and 3 raise from ``gym.make``, after the map has been resolved — on a
fresh machine that includes the download.

``timestep`` must be an exact multiple of ``integrator_timestep``; the ratio
sets the number of integrator substeps taken per environment step. The check
is validated against that ratio, so any pair that divides evenly is accepted
— ``timestep=0.03, integrator_timestep=0.01`` gives 3 substeps. The pairing is
checked when the simulator is built, not when the config is constructed, so an
invalid pair raises from ``gym.make`` rather than from ``SimulationConfig``:

>>> import gymnasium as gym
>>> from f1tenth_gym.envs.env_config import SimulationConfig
>>> cfg = EnvConfig(
...     simulation_config=SimulationConfig(timestep=0.025, integrator_timestep=0.01)
... )                                    # constructs: tier 1 has no ratio rule
>>> gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
Traceback (most recent call last):
    ...
ValueError: timestep (0.025) must be an integer multiple of ...

``ControlConfig``
-----------------

The two mode fields choose what each action column means and the bounds the
action space declares — :doc:`actions` is the column-by-column reference —
while the delay and noise fields corrupt the command before it reaches the
actuator (:doc:`sim2real`). Enums come from ``f1tenth_gym.envs.action``.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``longitudinal_mode``
     - ``LongitudinalActionType.SPEED``
     - ``SPEED`` (column 1 commands a target speed, tracked by a
       proportional controller) or ``ACCL`` (column 1 is a raw
       acceleration).
   * - ``steering_mode``
     - ``SteerActionType.STEERING_ANGLE``
     - ``STEERING_ANGLE`` (column 0 commands a target angle, realised by a
       saturated P controller) or ``STEERING_SPEED`` (column 0 is a steering
       rate).
   * - ``steer_delay_steps``
     - ``0``
     - Ring-buffer lag, in steps, on the steering command; ``>= 0``.
   * - ``throttle_delay_steps``
     - ``0``
     - Ring-buffer lag on the longitudinal command; ``>= 0``.
   * - ``steer_noise_std``
     - ``0.0``
     - Std of Gaussian noise added to the steering command each step;
       ``>= 0``. Noise is applied before the delay buffers.
   * - ``accl_noise_std``
     - ``0.0``
     - Std of Gaussian noise added to the longitudinal command each step;
       ``>= 0``.
   * - ``steer_kp``
     - ``None``
     - Gain of the ``STEERING_ANGLE`` controller,
       ``sv = clip(kp * error, -sv_max, sv_max)``. ``None`` derives
       ``10 * sv_max / (s_max - s_min)`` (about 38.2 for F1TENTH); ``<= 0``
       selects the legacy bang-bang relay, which limit-cycles by roughly
       ``sv_max * timestep``. Deliberately unvalidated — any float is legal.

The delay fields index a ring buffer, so a fractional value is coerced to
``int`` rather than stored verbatim:

>>> from f1tenth_gym.envs.env_config import ControlConfig
>>> ControlConfig(steer_delay_steps=2.7).steer_delay_steps
2

``SimulationConfig``
--------------------

The physics clock, integrator, dynamics model and lap rule. Enums:
``IntegratorType`` from ``f1tenth_gym.envs.integrators``, ``DynamicModel``
from ``f1tenth_gym.envs.dynamic_models``, ``LoopCounterMode`` from
``f1tenth_gym.envs.env_config``.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``timestep``
     - ``0.01``
     - Sim timestep in seconds; ``> 0``. Paired with
       ``integrator_timestep`` by the substep rule above.
   * - ``integrator_timestep``
     - ``0.01``
     - Integration substep in seconds (may be smaller than ``timestep``);
       ``> 0``.
   * - ``integrator``
     - ``IntegratorType.RK4``
     - ``RK4`` or ``EULER``.
   * - ``dynamics_model``
     - ``DynamicModel.ST``
     - ``KS`` (kinematic single-track, 5-state) or ``ST`` (single-track,
       7-state). ``MB`` is unsupported in this version. See :doc:`dynamics`.
   * - ``loop_counter``
     - ``LoopCounterMode.FRENET_BASED``
     - ``FRENET_BASED``, or ``WINDING_ANGLE`` (cumulative angle around the
       track centroid; reliable on convex tracks only).
   * - ``compute_frenet_frame``
     - ``True``
     - Whether per-agent Frenet ``(s, ey, ephi)`` coordinates are computed
       each step.
   * - ``max_laps``
     - ``1``
     - Laps before ``terminated=True``; ``None`` = no lap limit; ``>= 1``
       when set. Watches the ego's lap count only — there is no per-agent
       variant.

On its own the class accepts ``FRENET_BASED`` lap counting with the Frenet
frame off; its ``with_updates`` and ``EnvConfig`` both repair the combination
by coercing ``compute_frenet_frame=True``:

>>> sim = SimulationConfig(compute_frenet_frame=False)
>>> sim.compute_frenet_frame
False
>>> EnvConfig(simulation_config=sim).simulation_config.compute_frenet_frame
True

One cross-rule runs ahead of that repair: ``RewardMode.PROGRESS`` with
``compute_frenet_frame=False`` raises rather than coerces (see
``RewardConfig`` below).

``ObservationConfig``
---------------------

What ``obs`` looks like; the field vocabulary, shapes and space bounds of
each type are in :doc:`observations`. ``ObservationType`` comes from
``f1tenth_gym.envs.observation``.

.. list-table::
   :header-rows: 1
   :widths: 20 28 52

   * - Field
     - Default
     - Allowed values / notes
   * - ``type``
     - ``ObservationType.DEFAULT``
     - ``DEFAULT`` (per-agent dict of named fields),
       ``DIRECT`` (raw agent-batched arrays since v1.0.0 —
       selecting it warns), ``FEATURES`` (custom subset), or the fixed
       presets ``KINEMATIC_STATE`` / ``DYNAMIC_STATE`` /
       ``FRENET_DYNAMIC_STATE``. Not itself validated at construction; a
       non-``ObservationType`` value raises ``TypeError`` at ``gym.make``.
   * - ``features``
     - ``None``
     - Tuple of field names, valid only with ``type=FEATURES``. Not coerced
       to a tuple: a list is accepted and makes the config unhashable.

>>> from f1tenth_gym.envs.env_config import ObservationConfig
>>> ObservationConfig(features=("pose_x",))
Traceback (most recent call last):
    ...
ValueError: observation `features` only applies to ObservationType.FEATURES, ...

Two keys are conditional whatever the type: ``scan`` is dropped when
``lidar_config.enabled=False``, and ``frenet_pose`` is dropped when
``compute_frenet_frame=False``. Naming a dropped field in a ``FEATURES``
tuple raises ``ValueError`` at ``gym.make``.

``ResetConfig``
---------------

Where agents spawn at ``reset()``. ``ResetStrategy`` and ``ReferenceLine``
come from ``f1tenth_gym.envs.reset``.

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Field
     - Default
     - Allowed values / notes
   * - ``strategy``
     - ``ResetStrategy.RL_GRID_STATIC``
     - ``RL_GRID_STATIC`` / ``RL_RANDOM_STATIC`` / ``RL_GRID_RANDOM`` /
       ``RL_RANDOM_RANDOM`` / ``MAP_RANDOM_STATIC``.
   * - ``min_dist``
     - ``None``
     - Minimum spawn spacing in metres; ``None`` = strategy default (1.5
       for ``RL_*``, 0.5 for MAP); ``>= 0``.
   * - ``max_dist``
     - ``None``
     - Maximum spawn spacing in metres; ``None`` = strategy default (2.5
       for ``RL_*``, 1.0 for MAP); ``> 0``, and ``> min_dist`` when both
       are set.
   * - ``shuffle``
     - ``None``
     - ``None`` = strategy default (STATIC strategies keep order).
       Permutes only which agent gets which slot — the pose set itself is
       identical.
   * - ``move_laterally``
     - ``None``
     - ``None`` = strategy default: ``False`` for ``RL_*``; ``True`` for
       MAP, where the flag is never read. No effect with a single agent.
   * - ``reference_line``
     - ``None``
     - ``None`` = ``ReferenceLine.RACELINE``. ``CENTERLINE`` spawns on the
       line the Frenet frame measures against, giving ``ey == 0`` at spawn
       (:doc:`tracks`). Rejected for the MAP strategy.
   * - ``start_width``
     - ``None``
     - ``None`` = 1.0 m: length of the spawn window along the reference
       line, in absolute metres, clamped to at least one waypoint so scaled
       maps stay legal. GRID strategies only; ``ValueError`` otherwise.
       Must be ``> 0``.

``reset_kwargs()`` forwards the non-``None`` optional fields to the
reset-function builder:

>>> from f1tenth_gym.envs.env_config import ResetConfig
>>> ResetConfig(min_dist=2.0).reset_kwargs()
{'min_dist': 2.0}

``LiDARConfig``
---------------

The simulated LiDAR casts exact rays against oriented wall segments. Import
from ``f1tenth_gym.envs.lidar``; every angular field is in radians, and an
angle beyond ±π raises with a did-you-pass-degrees hint.

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Field
     - Default
     - Allowed values / notes
   * - ``enabled``
     - ``True``
     - When ``False`` the ``scan`` key disappears from observations. Geometric
       contact remains active and is independent of LiDAR.
   * - ``num_beams``
     - ``1080``
     - ``>= 1``. Beams evenly span the exact stored angle pair, including
       both endpoints when there is more than one beam.
   * - ``angle_min``
     - ``-2.3561945`` (-135°)
     - Must be ``>= -π``.
   * - ``angle_max``
     - ``2.3561945`` (+135°)
     - Must be ``<= π`` and ``> angle_min``.
   * - ``range_min``
     - ``0.0``
     - Readings below are clipped; ``>= 0`` and ``< range_max``.
   * - ``range_max``
     - ``30.0``
     - Readings above are clipped; ``> 0``.
   * - ``noise_std``
     - ``0.01``
     - Std of per-beam Gaussian range noise, drawn each step; ``>= 0``.
   * - ``dropout_prob``
     - ``0.0``
     - Per-beam, per-step no-return probability (dropped beams read
       ``range_max``); in ``[0, 1]``.
   * - ``range_bias_std``
     - ``0.0``
     - Std of a per-beam systematic bias drawn once per episode; ``>= 0``.
   * - ``base_link_to_lidar_tf``
     - ``(0.275, 0.0, 0.0)``
     - ``(x, y, yaw)`` sensor offset from ``base_link`` in metres/radians. The
       simulator subtracts the active vehicle's ``lr`` from x when converting
       it to the model state frame. Unvalidated.
   * - ``scan_device``
     - ``"cpu"``
     - ``"cpu"`` or ``"gpu"`` for the JAX segment-intersection kernel.
       Async vector workers must use a spawn context after JAX initializes.

The three noise fields shape only the *observed* scan. Collision and response
come from segment contact geometry independently of LiDAR (:doc:`sim2real`).
``reset()`` ends with one sweep, so the first observation's ``scan`` is already
real and noise-bearing; reset does not run the contact solver. Convenience
read-only properties: ``angle_increment`` and ``maximum_range``.

``angle_min`` and ``angle_max`` are the sensor's true extent and the only stored
geometry. ``field_of_view`` is a read-only property equal to
``angle_max - angle_min``, so the two can never disagree. Build from a symmetric
sweep with :meth:`~f1tenth_gym.envs.lidar.LiDARConfig.from_fov`:

>>> import numpy as np
>>> from f1tenth_gym.envs.lidar import LiDARConfig
>>> print(LiDARConfig.from_fov(np.deg2rad(180)).angle_min)
-1.5707963267948966
>>> LiDARConfig(angle_min=-0.5, angle_max=1.5).field_of_view
2.0
>>> LiDARConfig().with_updates(angle_min=-1.0).field_of_view
3.3561945

Giving both a field of view and an explicit angle is over-determined and raises.
``field_of_view`` used to be a constructor argument, which made the same number
an input, validated, and then overwritten from the angles — so
``LiDARConfig(field_of_view=5.0, angle_min=-0.5, angle_max=0.5)`` silently
stored ``1.0``.

There is no backend selector. The former raster/EDT scanner was removed after
the exact segment implementation passed its differential gates. Delete a stale
``backend`` key from serialized development configurations; it now raises
``TypeError`` instead of silently selecting different sensor physics.

``RenderConfig``
----------------

Pacing and frame output; :doc:`rendering` explains the clocks these fields
drive. Rendering also needs ``render_enabled=True`` *and* a ``render_mode``
at ``gym.make`` — with either missing no renderer is built and these fields
are inert.

.. list-table::
   :header-rows: 1
   :widths: 28 24 48

   * - Field
     - Default
     - Notes
   * - ``render_fps``
     - ``60``
     - Coerced to ``int``; ``> 0``. Caps human-mode redraws per wall-clock
       second and sets the rgb_array distinct-frame cadence in sim time.
   * - ``real_time_factor``
     - ``1.0``
     - Coerced to ``float``; ``> 0``; ``float("inf")`` = free-run, ``nan``
       rejected. Sim-seconds per wall-second in human modes; togglable at
       runtime via ``env.unwrapped.set_real_time_factor(x)``.
   * - ``window_size``
     - ``800``
     - Coerced to ``int``; ``> 0``. Square window / frame / video size in
       pixels.
   * - ``focus_on``
     - ``"agent_0"``
     - Agent id the camera follows. Unvalidated; an unknown id raises
       ``ValueError`` when the renderer is built. ``None`` is not a map
       view — it parks the camera at the world origin (the map view is the
       renderer's middle-click toggle).
   * - ``vehicle_palette``
     - 10 hex colours
     - Per-agent car colours, cycled by index.
   * - ``show_wheels``
     - ``True``
     - Stored but never read by the GL backend — frames are byte-identical
       either way.
   * - ``render_map_img``
     - ``True``
     - Draw the occupancy-map image under the scene.
   * - ``car_thickness``
     - ``1``
     - Coerced to ``int``. Stored but never read, like ``show_wheels``.
   * - ``bigger_car_when_map_centered``
     - ``True``
     - Scale cars up in the zoomed-out map view.
   * - ``show_lap_info``
     - ``True``
     - Overlay the lap-time / lap-count label.

``TerminationConfig``
---------------------

When an episode ends, and on whose account.

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Field
     - Default
     - Allowed values / notes
   * - ``max_episode_steps``
     - ``None``
     - ``truncated=True`` once this many steps elapse since reset — a
       truncation limit without a gymnasium ``TimeLimit`` wrapper (the
       registered spec sets none). ``None`` = no limit; ``>= 1`` when set.
   * - ``terminate_on_collision``
     - ``True``
     - Whether a collision sets ``terminated=True``.
   * - ``agent_mode``
     - ``AgentTerminationMode.EGO``
     - Reduce per-agent collision and lap-completion status with ``EGO``,
       ``ANY`` or ``ALL``.

>>> from f1tenth_gym.envs.env_config import TerminationConfig
>>> TerminationConfig(agent_mode="all")
Traceback (most recent call last):
    ...
TypeError: agent_mode must be an AgentTerminationMode

Per-agent collision flags arrive in ``info["collisions"]`` on every ``step``
(the ``reset`` info dict has no such key). ``info["terminated_agents"]`` is a
copied Boolean array available on reset and step. It latches collision and lap
completion status for the episode without stopping or freezing individual
vehicles. The configured reduction always returns one environment-wide
``terminated`` value.

``RewardConfig``
----------------

The per-step scalar; the arithmetic of each mode, with measured magnitudes,
lives in :doc:`rl`. ``RewardMode`` comes from ``f1tenth_gym.envs.env_config``.

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Field
     - Default
     - Allowed values / notes
   * - ``mode``
     - ``RewardMode.SURVIVAL``
     - ``SURVIVAL`` (``reward = timestep``, the historical default),
       ``PROGRESS`` (weighted Frenet progress + speed + survival −
       collision penalty), or ``CUSTOM``.
   * - ``progress_weight``
     - ``1.0``
     - Reward per metre of forward arclength (PROGRESS). Unvalidated — any
       float, either sign.
   * - ``velocity_weight``
     - ``0.0``
     - Reward per m/s of *signed* ego speed (PROGRESS); reversing pays
       negative. Unvalidated.
   * - ``timestep_weight``
     - ``0.0``
     - Survival bonus per step, scaled by the timestep (PROGRESS).
       Unvalidated.
   * - ``collision_penalty``
     - ``0.0``
     - Subtracted on every step the ego is in contact, not once per crash
       (PROGRESS); ``>= 0``.
   * - ``reward_fn``
     - ``None``
     - For ``CUSTOM``: a callable
       ``(obs, action, info, terminated, truncated) -> float``.

``CUSTOM`` requires ``reward_fn``, a non-callable ``reward_fn`` raises
``TypeError``, and a ``reward_fn`` set alongside ``SURVIVAL`` or ``PROGRESS``
is stored and never called. ``PROGRESS`` needs the Frenet frame —
``EnvConfig`` raises when ``compute_frenet_frame`` is ``False``. Whatever the
mode, ``info`` carries the raw ``progress`` and ``collisions`` signals every
step:

>>> from f1tenth_gym.envs.env_config import RewardConfig, RewardMode
>>> RewardConfig(mode=RewardMode.CUSTOM)
Traceback (most recent call last):
    ...
ValueError: RewardMode.CUSTOM requires reward_fn to be set

``DomainRandomizationConfig``
-----------------------------

Per-episode vehicle-parameter randomization, redrawn from the environment RNG
at every Gymnasium ``reset()`` — reproducible with ``reset(seed=...)``;
:doc:`sim2real` covers where the draw lands in the control loop. The native JAX
batch adapter uses the same bounds with one explicit device key per environment
instead; see :doc:`rl`.

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Field
     - Default
     - Allowed values / notes
   * - ``enabled``
     - ``False``
     - Whether to randomize at each reset.
   * - ``low`` / ``high``
     - ``None``
     - The per-field lower and upper end of the range, each an ordinary
       ``VehicleParameters`` in absolute physical units. Both are required
       when ``enabled``. Every field needs ``low <= high``, and a field
       where they are equal is not randomized — so building both from your
       base parameters and changing only what should vary is the idiom.

>>> from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as base
>>> from f1tenth_gym.envs.env_config import DomainRandomizationConfig
>>> dr = DomainRandomizationConfig(
...     enabled=True,
...     low=base.with_updates(m=3.0, mu=0.9),
...     high=base.with_updates(m=4.0, mu=1.1),
... )
>>> dr.randomized_fields()
('mu', 'm')

Randomizing an actuation limit (``v_max``, ``s_max``, …) is supported: the
action and observation spaces are built from ``widest_params(base)``, a fixed
superset of every randomized episode, so gymnasium's fixed-space contract
holds. The physics ground truth for the current episode is
``env.unwrapped.sim.params_array`` — ``env.vehicle_params`` keeps the nominal
values.

.. note::

   Names are the raw ``VehicleParameters`` fields — ``m`` (mass) and ``h``
   (CoG height), not ``mass``/``h_cg``. A range over an unsupported MB-only
   field (``K_zt``, …) raises at ``EnvConfig`` construction. ``width``,
   ``length``, ``lr`` and the collision-body offsets take effect through
   opponent occlusion and contact geometry.

Reconfiguring a live environment
--------------------------------

``env.unwrapped.configure(new_cfg)`` swaps the whole config on a running env
and rebuilds the affected components — track, simulator, spaces, reset
function, renderer:

>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> print(f"{env.unwrapped.sim.params_array[0]:.4f}")   # mu, physics ground truth
1.0489
>>> cfg = env.unwrapped.env_config
>>> env.unwrapped.configure(
...     cfg.with_updates(params=cfg.params.with_updates(mu=1.0))
... )
>>> print(f"{env.unwrapped.sim.params_array[0]:.4f}")
1.0000
>>> obs, info = env.reset(seed=0)   # reset after reconfiguring
>>> env.close()

Two costs are invisible in the return value: a rebuilt renderer starts with
an empty callback list, so re-register anything added through
``add_render_callback``; and switching ``render_enabled`` from ``True`` to
``False`` leaks the old GL context, because the guard that would close it
reads the new flag. ``configure`` also re-arms ``EnvConfig.seed`` to cover
the next unseeded reset.
