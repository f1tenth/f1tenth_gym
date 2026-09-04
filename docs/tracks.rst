Maps, racelines and the Frenet frame
====================================

.. seealso::

   :mod:`f1tenth_gym.envs.track` — ``Track``, ``Raceline``, ``CubicSplineND``.

A track is an occupancy grid plus two closed reference lines, and they are not
the same loop: Spielberg's centerline runs 343.32 m and its raceline 338.13 m.
Most of the numbers that look wrong at first — a lateral error of 0.81 m at
standstill, a lap that costs 343 m of progress for 338 m of driving — fall out
of that one gap.

Two lines, and they are not the same loop
-----------------------------------------

A :class:`~f1tenth_gym.envs.track.track.Track` carries two
:class:`~f1tenth_gym.envs.track.raceline.Raceline` objects, each backed by a
periodic cubic spline over ``[x, y, cos ψ, sin ψ, k, vx, ax]``. ``centerline``
is the geometric middle of the tarmac; ``raceline`` is an optimized line with a
speed profile attached:

>>> from f1tenth_gym.envs.track.track import Track
>>> track = Track.from_track_name("Spielberg")
>>> (round(float(track.centerline.spline.s[-1]), 4),
...  round(float(track.raceline.spline.s[-1]), 4))
(343.3222, 338.1253)
>>> track.centerline.vxs[:3]
array([1., 1., 1.], dtype=float32)
>>> track.raceline.vxs[:3]
array([8., 8., 8.], dtype=float32)

The 1 m/s on the centerline is not a measurement. ``Raceline.from_centerline_file``
fills ``vxs`` from a hard-coded ``fixed_speed=1.0``, because a bare
``{stem}_centerline.csv`` carries only ``x_m`` and ``y_m`` columns — so
``centerline.vxs`` is geometry, never a speed target. The raceline's 8 m/s comes
from the ``vx_mps`` column of its own CSV.

Spawning on one line, measured against the other
------------------------------------------------

Every ``RL_*`` reset strategy places its cars on ``track.raceline``, while the
Frenet transform projects onto ``track.centerline``. The two lines do not
coincide, so the reported lateral deviation is already large before the car has
moved:

>>> import gymnasium as gym
>>> from f1tenth_gym.envs.env_config import EnvConfig
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> s, ey, ephi = obs["agent_0"]["frenet_pose"]
>>> print(f"{s:.4f} {ey:.4f} {ephi:.4f}")
0.2630 0.8086 -0.0008
>>> env.close()

``ey`` is non-zero at spawn by construction, not by luck of the draw: across
seeds 0–49 on Spielberg the spawn ``ey`` spans only ``[0.8080, 0.8086]``, while
``x`` and ``y`` move through a window about a metre long (:doc:`reproducibility`).
Subtract the spawn value if a lateral controller needs a zero baseline, or spawn
on the line the frame actually measures against:

>>> from f1tenth_gym.envs.env_config import ResetConfig
>>> from f1tenth_gym.envs.reset import ReferenceLine
>>> cfg = EnvConfig(
...     reset_config=ResetConfig(reference_line=ReferenceLine.CENTERLINE),
...     render_enabled=False,
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> print(f"{obs['agent_0']['frenet_pose'][1]:.4f}")
0.0000
>>> env.close()

Lap counting is untouched by the gap. The per-agent ``s`` is centerline
arclength and the lap check divides it by the centerline's own
``s_frame_max``, so one physical lap advances the projection by exactly one
centerline length whatever line the car drives. What the gap does change is the
odometer: ``info["progress"]`` and the PROGRESS reward (:doc:`rl`) are
denominated in centerline arclength too, so a lap accrues 343.3 m even though
the raceline reads 338.1 m — a longer racing line is not paid more for being
longer. A lap also fires one centerline length after the spawn ``s``, not at a
start/finish line.

Reading ``(s, ey, ephi)``
-------------------------

With ``SimulationConfig.compute_frenet_frame`` left at its default ``True``,
each step reports one 3-vector per agent, in metres, metres and radians:

* ``s`` — arclength along the centerline spline, wrapped into
  ``[0, s_frame_max)``.
* ``ey`` — signed lateral deviation, positive to the **left** of the direction
  of travel; the sign comes from the spline normal ``[-sin ψ, cos ψ]``.
* ``ephi`` — heading minus the centerline's heading there, wrapped to
  ``[-π, π)``.

The pose handed to the transform is the same physical reference point under
every dynamics model, so ``frenet_pose`` is directly comparable across a KS/ST
switch (:doc:`dynamics`). Turning ``compute_frenet_frame`` off removes
``frenet_pose`` from the observation dict rather than zeroing it, and naming it
explicitly in ``ObservationConfig(features=...)`` then raises at ``gym.make``
(:doc:`observations`).

:meth:`~f1tenth_gym.envs.track.track.Track.cartesian_to_frenet` and
:meth:`~f1tenth_gym.envs.track.track.Track.frenet_to_cartesian` are available on
any track and round-trip. Offsetting a centerline point 0.5 m along the normal
lands at ``ey = +0.5``:

>>> import numpy as np
>>> x, y = track.centerline.spline.calc_position(100.0)
>>> yaw = float(track.centerline.spline.calc_yaw(100.0))
>>> left = (float(x - 0.5 * np.sin(yaw)), float(y + 0.5 * np.cos(yaw)))
>>> s, ey, ephi = track.cartesian_to_frenet(*left, yaw, use_s_guess=False)
>>> print(f"{s:.4f} {ey:.4f} {ephi:.4f}")
100.0000 0.5000 0.0000
>>> print("{:.4f} {:.4f} {:.4f}".format(*track.frenet_to_cartesian(s, ey, ephi)))
-69.4765 44.2699 2.3436

``use_s_guess=False`` above buys correctness with a global search. The simulator
cannot afford that every step, so it seeds each agent's projection with that
agent's previous ``s`` and searches a window of roughly ±5 m of arclength around
it — ±4.76 m on Spielberg, quantised to whole spline segments. The width comes
from ``Track.frenet_search_range``, a plain attribute set to 10 m with no
``EnvConfig`` field behind it: an agent that translates further than the window
in a single step locks onto the wrong stretch of the loop and stays there. Both
methods also take ``use_raceline=True``, but nothing in the simulator passes it,
so the reported frame is always the centerline.

Where the map comes from
------------------------

``EnvConfig.map_name`` accepts three kinds of value and ``_resolve_track``
dispatches on which one it got:

* a bare name — no path separator and no suffix, such as ``"Spielberg"`` —
  goes to :meth:`Track.from_track_name
  <f1tenth_gym.envs.track.track.Track.from_track_name>`, which looks under the
  repository-root ``maps/`` directory and downloads the track if it is absent;
* a path — anything containing ``/``, ``\`` or a file suffix — goes to
  :meth:`Track.from_track_path
  <f1tenth_gym.envs.track.track.Track.from_track_path>`, which accepts the track
  directory, a stem inside it, or the map YAML itself, under either the
  ``{stem}.yaml`` or the legacy ``{stem}_map.yaml`` naming convention;
* a ``Track`` instance is used as-is, with no I/O at all.

All three path forms reach the same map:

>>> import pathlib
>>> track_dir = pathlib.Path(Track.from_track_name("Spielberg").filepath).parent
>>> stem = track_dir / "Spielberg"
>>> for candidate in (track_dir, stem, stem.with_suffix(".yaml")):
...     round(float(Track.from_track_path(candidate).centerline.s_frame_max), 4)
343.3222
343.3222
343.3222

The first request for a built-in name fetches
``https://api.f1tenth.org/<name>.tar.xz`` and extracts it — with tar's
``filter="data"`` hardening, and no checksum — into a ``maps/`` directory
resolved four levels up from the package source, which an editable clone
provides and a built wheel does not (:doc:`installation`). A name the endpoint
does not serve raises ``FileNotFoundError("No maps exists for <name>.")``.

Reaching for the third form is the highest-leverage change available when
building many environments. Constructing the LiDAR's Euclidean distance
transform over Spielberg's 2000×2000 grid dominates setup, and the result is
cached on the ``Track``, so environments sharing one instance pay for it once:
measured here, ``gym.make(config=EnvConfig(map_name=track))`` takes 1.2 ms
against about 210 ms for the same map by name.

Reference lines are optional files looked up beside the YAML:
``{stem}_centerline.csv`` (comma-delimited, columns ``x_m, y_m, …``) and
``{stem}_raceline.csv`` (semicolon-delimited, ``s_m; x_m; y_m; psi_rad;
kappa_radpm; vx_mps; ax_mps2``). Ship only one and both loaders use it for the
other, so a raceline-only directory loads and drives — with the frame measured
against the same line the cars spawn on, and therefore ``ey = 0`` at spawn. Ship
neither and loading raises ``ValueError``, naming both files it looked for.
``EnvConfig.map_scale`` scales the YAML's ``resolution`` and ``origin[:2]``
together with the reference-line coordinates, so the whole world grows or
shrinks as a unit (:doc:`configuration`).

What the occupancy image encodes
--------------------------------

A track directory pairs the YAML metadata with the greyscale bitmap it names.
Three keys place the geometry in the world: ``resolution`` is metres per pixel;
``origin`` is the world coordinate of the grid's bottom-left corner, of which
only the first two components are used and the third, a rotation, is ignored;
and ``image`` names the bitmap, which is loaded ``FLIP_TOP_BOTTOM`` so that grid
row 0 is the smallest world ``y``.

The image is binarised following ROS ``map_server`` semantics. Occupancy probability is
``(255 - pixel) / 255``, or ``pixel / 255`` when ``negate: 1``, and a cell is an
obstacle exactly when that probability exceeds ``occupied_thresh``; everything else —
including the ROS "unknown" band — is free. In the resulting grid ``0`` is occupied
and ``255`` is free. Exact LiDAR and contact geometry is extracted as oriented wall
segments along the occupied/free boundary; occupancy queries still use the binary
grid. Releases before v1.0.0 ignored the YAML and cut at a hard-coded pixel value of
128; set
``occupied_thresh: 0.495`` to reproduce that grid exactly.

Only those two values survive the binarisation, and walls are the rare ones —
0.85% of Spielberg's four million pixels:

>>> np.unique(track.occupancy_map)
array([  0., 255.], dtype=float32)
>>> float((track.occupancy_map == 0.0).mean())
0.0084995

``Track.load_spec`` validates the rest of the YAML before any of this runs.
``image``, ``resolution`` and ``origin`` are required and their absence raises
``ValueError`` naming the file and the missing keys; ``negate``,
``occupied_thresh`` and ``free_thresh`` fall back to the ROS defaults ``0``,
``0.65`` and ``0.196``; ``mode: trinary`` is accepted while ``raw`` and
``scale`` are rejected; and an unrecognised key is dropped with a warning rather
than crashing the load.

Synthetic tracks are always closed
----------------------------------

Racing a shape with no map file takes one call. :meth:`Track.from_refline
<f1tenth_gym.envs.track.track.Track.from_refline>` fits a periodic spline
through ``x``, ``y`` and ``velx`` arrays, sets both reference lines to it, and
synthesizes an all-free occupancy grid sized to the extents plus a 5 m margin.
Pass the instance as ``map_name``:

>>> from f1tenth_gym.envs.env_config import SimulationConfig
>>> theta = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
>>> circle = Track.from_refline(
...     x=10.0 * np.cos(theta), y=10.0 * np.sin(theta), velx=np.full(200, 5.0)
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(
...     map_name=circle,
...     simulation_config=SimulationConfig(max_laps=None),
...     render_enabled=False,
... ))
>>> obs, info = env.reset(seed=42)
>>> for _ in range(50):
...     obs, *_ = env.step(np.array([[0.0, 3.0]], dtype=np.float32))
>>> s, ey, ephi = obs["agent_0"]["frenet_pose"]
>>> print(f"{circle.centerline.s_frame_max:.4f} {s:.4f} {ey:.4f}")
62.8293 0.8667 -0.0249
>>> env.close()

A 10 m radius makes a 62.83 m loop, the car has covered 0.87 m of it, and it
sits 2.5 cm off the line rather than the 0.81 m of the previous section — on a
synthetic track both reference lines are the same object, so that offset cannot
arise.

.. warning::

   ``from_refline`` **force-closes the reference line into a loop** — there is
   no open-path mode. If the last point does not coincide with the first, an
   extra point is appended to close the loop (required by the periodic spline).
   A "straight line" input such as ``x=np.linspace(0, 10, N), y=np.zeros(N)``
   therefore becomes a closed ~20 m path with a phantom return leg, and ``s``,
   curvature and lap counting all run over that closed loop. Design your
   reference line as an intentionally closed circuit.

Fifty points along a 10 m segment come back as 51 points over a 20 m loop:

>>> straight = Track.from_refline(
...     x=np.linspace(0.0, 10.0, 50), y=np.zeros(50), velx=np.full(50, 5.0)
... )
>>> straight.centerline.spline.s.shape
(51,)
>>> float(straight.centerline.s_frame_max)
20.0

``env.unwrapped.update_map(track_or_name)`` swaps the map on a live environment
by re-running ``configure`` with the new ``map_name``, which rebuilds the
simulator, the spaces and the renderer around it — so any render callbacks have
to be registered again afterwards (:doc:`rendering`).

Functional device tables
------------------------

The functional JAX layer never loads a map inside a compiled function.
``f1tenth_gym.envs.track.preprocessing.preprocess_track`` runs on the host and
converts an existing ``Track`` into fixed-shape spline coefficients, reference
points, oriented walls, contact/ray tile candidates and explicit masks. The
returned ``TrackTable`` is a JAX pytree; pure transforms in
``f1tenth_gym.envs.track.functional`` evaluate its splines and convert global
Cartesian/Frenet poses without NumPy conversion.

``IndexedJaxSimulator`` preprocesses repeated ``Track`` object identities once
and groups heterogeneous maps by exact table shape and dtype. Each group gets
its own compiled indexed transition, avoiding a second stand-alone bucketing
API and avoiding global padding to the largest map.

``f1tenth_gym.envs.reset.preprocessing.preprocess_reset`` separately
preprocesses current grid/all-track waypoint and spacing choices.
``sample_reset_poses`` consumes only an explicit JAX key, fixed arrays and the
internal ``ResetSamplingConfig`` topology. That name deliberately distinguishes
the compiled sampling choices from the user-facing ``EnvConfig.reset_config``
and its public ``ResetConfig`` type.
``MAP_RANDOM_STATIC`` is not yet part of this device surface; its current host
sampler has a known row/column bug that must be resolved as an explicit behavior
decision rather than copied into the new backend.

``JaxSimulator`` selects the centerline or raceline, spacing, grid window,
shuffle and lateral reset policy from ``EnvConfig`` and materializes those
reference-line choices. It builds an acceleration table only for an enabled subsystem:
collision ``NONE`` gets constant-size masked contact and pair placeholders,
and disabled LiDAR gets a masked ray placeholder. When both are disabled, wall
extraction is skipped as well. An enabled contact table is sized to the
farthest possible model-pose-relative collision-body corner, including
configured body offsets and domain-randomization bounds.

Exact device scans
------------------

``clean_scan`` gathers the ray-tile candidates for every LiDAR pose, applies
the candidate and wall masks, and casts analytically against oriented wall
segments. It then shortens those ranges against every edge of every opponent
body in one fixed-shape calculation. The result is a noise-free ``(agents,
beams)`` array. ``reset_scan_state`` samples the episode-fixed per-beam bias;
``observed_scan`` then adds per-step Gaussian noise and that bias, clips ranges,
and applies dropout. Collision response uses independent contact geometry.

Build the ray table for at least the sensor's longest range. ``JaxSimulator``
checks that invariant before constructing the traced scan parameters because a
smaller table could silently omit a reachable wall. A larger preprocessed reach
is safe. ``base_link_to_lidar_tf`` is measured from ``base_link``, whereas the
supported models use a different longitudinal origin. Both simulator paths
therefore subtract the active vehicle's ``lr`` from the configured x offset.
The functional path keeps ``lr`` traced, so the mount remains correct under
batching and domain randomization.
