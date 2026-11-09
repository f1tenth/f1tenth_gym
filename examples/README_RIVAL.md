# R.I.V.A.L. defensive baseline

A first two-car simulation, **not proof of safety** and not a reproduction of IDM,
MOBIL, or an official IAC rulebook. No Gym physics, observation fields, or existing
examples are modified. The default experiment is a short segment on the supplied
156.356 m closed example racing line, not a full-lap benchmark.

## Run (from repository root)

Use the repository's installed dependencies on Python 3.11–3.15. For Bash (Fish
uses `activate.fish`):

```bash
python3.15 -m venv .venv-py315  # only if it does not already exist
source .venv-py315/bin/activate
python -m pip install -e .

# Default: eight simulated seconds, headless, left passing attempt
python examples/rival_baseline.py

# Matched defense / no-defense comparison with identical seeds and initial poses
python examples/rival_baseline.py --scenario left-pass --compare

# All four scripts, with matched controls
python examples/rival_baseline.py --scenario all --compare

# Optional GUI (requires a display and system OpenGL/GLU libraries)
python examples/rival_baseline.py --scenario left-pass --render --compare

# Tests: no display or simulator rollout required
python -m unittest discover -s examples -p 'test_rival_*.py' -v
```

Other options: `--duration 12`, `--no-defense`, `--config path/to/config.yaml`,
`--output path/to/results`. Config-relative track paths are resolved independently
of the shell working directory. `--duration` overrides YAML; `--compare` and
`--no-defense` are mutually exclusive. Reusing an output folder overwrites files
for the same scenario/mode; use distinct `--output` folders for parameter sweeps.
Default generated results are ignored by `examples/.gitignore`.

Rendering labels defender **blue**, attacker **red**, racing line **yellow**, and
pure-pursuit target **green**. A camera follows the defender; the label explains
the colors. Pyglet is imported lazily for the overlay. No display is opened in
default headless mode. GUI close behavior is inherited from the legacy renderer.

## Coordinate and controller conventions

`ClosedRaceline` closes the waypoint polyline, including its final-to-first
segment. Its own arc length is computed from xy, rather than trusting the CSV's
s column. Positive `d` is left of travel; positive curvature is a left turn.
Projection uses the nearest segment, so the intended reference should not
self-intersect or have nearby overlapping branches. Signed longitudinal gap is
`wrap(s_attacker - s_defender)` in `[-L/2, L/2)`; negative means behind. Both
unwrapped progress and pass detection accumulate small wrapped step differences,
so crossing start/finish is not a spurious pass or a negative lap.

Initial poses come from `start_s` and `start_s - initial_gap` on the supplied
waypoints, with heading from the line and a feasible initial lateral offset.
They do not use guessed map coordinates. Both cars start at rest. The seed,
timing, speeds, and scripts are deterministic; there is no randomized attacker.

The observation adapter reads `poses_x/y/theta` and `linear_vels_x/y`, estimating
longitudinal speed by projecting heading-based velocity onto the line tangent.
The legacy simulator reports its speed in `linear_vels_x`, zero in
`linear_vels_y`; slip angle is not exposed by this observation. This is an
approximation of Frenet speed, not an extra state or observation-contract change.

Pure pursuit adapts `waypoint_follow.py`'s body-frame lateral error formula:
`atan2(2 * wheelbase * lateral_error, target_distance_squared)`. Its lookahead
point is sampled forward along the line and shifted by the requested offset.
The actual target distance is used, rather than assuming it equals the arc
lookahead. Steering and target commands are rate-limited independently. Actions
remain `[steering_angle, speed]`, with defender row 0 and attacker row 1. Both
`reset` and `step` are unpacked as `(obs, reward, done, info)`.

## Legal-region interpretation and map margins

There is no authoritative track-centerline or width file in this example. The
racing line is **not** assumed to be the track center. At each queried s:

1. Scan both normal directions through the bitmap, retaining only the connected
   free cross-section containing the racing line. Stop at the first obstacle or
   the configurable `max_offset_search` cap (3 m each side by default).
2. Subtract body half-width and `boundary_margin` when certifying usable centers.
   The midpoint of those two bounds estimates the local track center.
3. Smooth curvature over s-1, s, s+1 m. In left turns, the usable **inner half**
   lies left of that estimated center; right turns use the right half. Offset the
   center boundary by `center_margin`, shrink the inside half by `inner_fraction`,
   and intersect it with configurable raceline-relative `legal_min/max`.
4. Below `curvature_threshold`, there is no defined inside: allow both sides of
   the local usable cross-section (still intersected with `legal_min/max`).

Thus “inner half” means a curvature-dependent half of an **estimated bitmap
cross-section**, not half of a fixed-width road, and not always positive d. This
is a scaled, configurable interpretation, not certified IAC legality. Search
caps, an open area, nearby connected lanes, or a near-tangent cross-section can
bias the inferred center. Supply dedicated track boundary/centerline data before
using this as a regulatory model. The raster map, rather than a hard-coded width,
is the source of all local boundary checks.

The nominal target is the point in the intersection of current/preview legal
bands closest to d=0. Both modes therefore follow the global racing line where
allowed, but use a legal shifted reference where the line lies outside this
interpreted inner half. The **no-defense** control retains the same map, legal,
rate, and emergency collision checks; it disables tactical side-blocking only.

A vectorized distance transform provides conservative bitmap clearance. Candidate
poses additionally certify an orientation-aware grid covering the entire vehicle
rectangle, including interiors, not just corners. Unknown/outside pixels are
blocked. Clearance bounds include pixel/interpolation uncertainty. These checks
can reject a pose that actually fits. Map threshold/image behavior follows the
simulator (>128 is free); image metadata threshold/negate flags are not applied.
`car_length/width` are safety-envelope settings, not physics overrides: values
smaller than the actual Gym vehicle are rejected. The controller wheelbase must
match the unchanged Gym geometry.

## Defensive policy

The IDM-inspired desired **bumper** gap is:

```
s_desired = minimum_gap + max(0, v_attacker * time_headway
              + v_attacker * closing_speed / (2 * sqrt(a_ref * b_comfortable)))
```

The defender reacts only to a car behind, closing faster than
`closing_threshold`, within `reaction_threshold * s_desired`. This is a trigger,
not an IDM acceleration model. MOBIL inspiration is similarly limited to
comparing useful lateral changes after safety rejection, not its highway
lane-changing acceleration equations or politeness term.

Candidates include the nominal target, held target, five points across the
preview-admissible interval, and the attacker's lateral corridor. Utility rewards
occupying that approaching corridor, penalizes movement, and requires
`usefulness_threshold` improvement over a safe held target. Reject candidates
outside the legal region, unsafe map/footprint predictions, or predicted
longitudinal/lateral rectangle overlaps. Check both the eventual candidate and
the **next issued rate-limited target**, so a tracking-lag reversal does not
silently check the opposite direction from the command. Logs distinguish reaction
trigger from an actual non-nominal defensive activation.

Prediction assumes constant current longitudinal speeds and constant attacker d,
with a bounded lateral transition toward a target. Relative interval boxes check
swept overlap between samples. Road/footprint prediction is sampled at six times
over `prediction_horizon`; it does not prove continuous collision avoidance,
especially during acceleration, high curvature, or unmodeled attacker movement.
Current tracking error may lie outside a legal band while the controller attempts
to recover: candidate endpoints and next commands, not perfect physical tracking,
are constrained. If there is no safe reachable command, request zero speed and
hold the previous offset rather than teleporting to a new band. The inherited
vehicle braking dynamics mean this is **not an instantaneous or guaranteed stop**.

Abrupt changes of the curvature-defined legal half can make a long experiment
stall; stopping is deliberate, not evidence of a successful lap. A proper
trajectory planner with braking-distance and region-transition planning is needed
for full-track operation under this interpretation.

## Scripts and feasible comparisons

- `no-attack`: both request defender speed and the nominal feasible line.
- `approach`: attacker requests higher speed but uses a desired-gap rear-end
  guard and does not request a passing side.
- `left-pass`: after `pass_start`, ramp a positive-d request over `pass_ramp`.
- `right-pass`: analogous negative-d request. On this sample segment much of the
  right side is outside the legal band or too narrow for two inflated cars.
  Requests are clipped and logged as `pass_infeasible:follow_clipped_lane`, or
  stopped if no safe reachable lane remains. This clipped approach is the closest
  valid script here, **not a claim that a right-side pass was feasible**.

The attacker follows a timed script, with a simple rear-end speed guard against
ramming. Its speed responds to the gap for avoidance, but it has no tactical
counter-policy. Both cars obey the same interpreted legal corridor. A request
can become infeasible later due to local curvature/boundary changes.

## Outputs and metric definitions

Each run writes `<scenario>_<defense|no_defense>.csv` and a `.json` summary.
CSV contains every control step: measured post-step Frenet states and speed,
pre-step decision gap/closing/lateral separation, desired gap, legal interval,
both targets/actions, reaction/activation flags, reasons, accumulated progress,
center separation, and each car's collision/off-track flags. `decision_time`
distinguishes the pre-step decision from post-step `time`.

Summary and terminal output include:

- Termination reason/time and whether **each** car completed `laps` full racing-line
  lengths from its own start; fractional laps and first completion time. A short
  duration-limited run normally has `completed=false`, `finish_time=null`.
- Gym lap counts/timers separately in JSON. Those use a different start-zone
  heuristic, so they need not equal Frenet laps.
- Simulator collision flags and conservative whole-body off-track failures
  (zero extra rules margin). Stop on either car's failure, even though Gym's done
  logic primarily considers the defender. Conservative off-track checks may
  false-reject poses near raster boundaries.
- A pass when the attacker is ahead by `car_length + separation_margin` in
  unwrapped progress for at least 0.5 s. The timestamp is the beginning of the
  qualifying lead, recorded only after confirmation. It is not just pulling
  alongside; finish-line wrap alone cannot create a pass.
- Minimum **center-to-center** distance, not bumper clearance; defender measured-offset
  and target min/max; non-nominal defensive activation episodes and active steps;
  maximum per-step target/steering changes and their commanded rates.
- Infeasible-request/stop steps (not a count of separate failure episodes), full
  settings, seed, and the baseline-only disclaimer in JSON.

## Verified initial results

On Python 3.14.8 with the repository's actual sample map and dependencies:

- `python -m unittest discover -s examples -p 'test_rival_*.py' -v`: **50 tests
  passed**, including wraparound, Frenet sign/closure, rotated bitmap maps,
  margins/legal bands, swept separation, unsafe candidate rejection, target
  limiting, preview intervals, cache eviction, and command/candidate reversal.
- All four scenarios and their matched no-defense controls ran for **8 seconds**.
  All eight had **no simulator collisions and no conservative off-track failures**.
  None completed a full lap; none met the sustained-pass criterion in that window.
- Left pass: defense activated **4** times, target range **[0, 0.455] m**, minimum
  center distance **1.562 m**; no-defense minimum **0.752 m**. Commanded lateral
  target rate never exceeded **0.25 m/s**. These are observations, not evidence
  that blocking improves safety or will prevent a later pass.
- Right pass: requests were often infeasible/clipped; defense activated **2**
  times. This is a constrained attempt, not a completed valid right-side pass.
- A 0.1 s rendering smoke test ran through the actual overlay using pyglet's
  headless OpenGL backend. Interactive display controls were not manually tested.

Speeds are intentionally low (2.0 / 2.8 m/s). This first baseline has no LiDAR
perception, learned policy, opponent uncertainty, optimal trajectory planning,
formal rule enforcement, or safety guarantee. It uses perfect simulator poses.
The stock Gym NumPy warning still prints; no Gymnasium API is assumed.
