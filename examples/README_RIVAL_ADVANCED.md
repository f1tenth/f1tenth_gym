# R.I.V.A.L. advanced simulation controller

**Research simulation only. Not approved for a real vehicle, not a guarantee of
collision avoidance, and not proof of effective defense against an adversarial
car.** This upgrade adds stronger estimation, prediction, control, and falsification
tests. Its current validation still finds unresolved safety decisions, ineffective
attacks, model-bound violations, and real-time deadline misses. There is no robot
actuation interface or hardware arming path.

The original `rival_baseline.py` and its policy are retained for comparison. Gym
equations, observations, and the four-value `reset`/`step` API are unchanged.
The advanced runner now integrates those equations in smaller substeps for
numerical stability. Actions remain rows of `[steering_angle, speed]`.

## Run

From the repository root, with the project installed (Python 3.15.2 on Fedora Linux was tested):

```bash
# Default: simulated noisy/delayed observations, adaptive attacker, headless
python examples/rival_advanced.py

# Matched tactical defense / no-defense, same sensor seed and initial conditions
python examples/rival_advanced.py --scenario adaptive --compare

# Original Defender logic, same estimator and stronger adversary
python examples/rival_advanced.py --controller legacy --scenario adaptive --compare \
  --output examples/rival_results/legacy_comparison

# New 3D race-control viewer; up to 300 simulated seconds / lap-goal completion
python examples/rival_advanced.py --scenario adaptive --duration 300 --render

# Track-aware faster profile with smooth attacker control
python examples/rival_advanced.py --config examples/rival_advanced_fast.yaml \
  --scenario adaptive --duration 120 --render --output examples/rival_results/fast

# Optional faster presentation (does not change simulation time or physics)
python examples/rival_advanced.py --scenario adaptive --render --render-speed 2

# Keep the original 10 Hz top-down presentation available
python examples/rival_advanced.py --scenario adaptive --render --viewer legacy --compare

# Bounded scenario matrices: four smoke cases or twelve stress cases
python examples/rival_advanced.py --suite smoke --duration 4 --compare
python examples/rival_advanced.py --suite stress --duration 4 \
  --output examples/rival_results/stress

# Experimental scan-only dynamic-object detection (rear blindness matters!)
python examples/rival_advanced.py --sensor lidar --duration 4

# All RIVAL unit tests, including compiled/reference trajectory parity
python -m unittest discover -s examples -p 'test_rival_*.py' -v
```

In Fish, activate with `source .venv-py315/bin/activate.fish`; Bash uses
`source .venv-py315/bin/activate`. Use `--config` for a separate YAML file and
`--output` to avoid overwriting earlier scenarios. `--duration` overrides YAML.
No dependency was added: the implementation uses the repository's NumPy, SciPy,
Numba, Pillow, PyYAML, and legacy Gym/pyglet stack.

The repaired `rival_advanced_fast.yaml` uses 3.0/3.4 m/s defender/attacker cruise
targets instead of blindly requesting 20 m/s. It enables `track_speed_planning`,
uses an 8 m initial gap and 2.5 m minimum following gap, and matches the attacker's
lateral target rate to the defender's 0.45 m/s capability. These are different
initial/following conditions from the conservative default, not a controlled
speed-only performance comparison. Neither profile changes vehicle physics.

The earlier 20 m/s configuration was not a usable high-speed controller: it
caused stop/go recovery, abrupt attacker acceleration changes, and only brief
peak speeds. Earlier 2.5/3.2 and 2.2/3.0 m/s variants also failed long-run stall
checks. Their results do not validate the current profile; see the repaired-profile
validation below. The original default profile remains available unchanged in
its control behavior.

The simulator's default 0.58 m-long F1TENTH vehicle has `v_max = 20 m/s`, so a
literal 26 m/s exceeds its configured limit; the sample track and controller are
also not validated for racing at that speed. Do not reinterpret scaled-car speeds
as verified full-size F1 speeds. `--render-speed` changes wall-time pacing only;
it cannot make a CPU-bound simulation run faster than the machine can compute.

The first trajectory solve may compile/load a Numba kernel and take longer.
Compilation and inference timings are not hidden by the metrics. A hardware
system would need an explicit warmup/arming gate and an independently enforced
control deadline; this runner supplies neither.

## Corner tracking in the normal fast setup

The normal fast configuration now enables `robust.corner_tracking: true` with
`target_tracking_gain: 4.0`. Cruise targets remain 3.0/3.4 m/s, the start/gaps and
inside legal region are unchanged, and Gym's physics and margins are unchanged.

The defender's offset target now includes bounded moving-reference feedforward:
it anticipates the reference offset's change as the car advances, rather than
treating the moving reference as a stationary point at which to stop. **Total**
target rate and acceleration remain bounded by 0.45 m/s and 0.8 m/s². Both cars
also reserve headroom to release positive acceleration before reaching a speed
ceiling, while preserving the 6 m/s³ command-jerk bound. This prevents the old
ceiling overshoot from unnecessarily triggering an attacker lateral correction.
There is no hard speed clip; an abruptly lowered, unreachable cap can still
require a transient braking response.

The shared speed step is in `rival_control.py`. Defender execution/readable/JIT
prediction and attacker readable/JIT prediction use matching arithmetic. The
conservative configuration leaves this opt-in disabled for comparison.

A fixed-seed full-lap trial of the fast adaptive scenario measured:

| Metric | Before corner fix | With corner fix |
|---|---:|---:|
| Defender corner lateral-error p95 | 0.298 m | 0.105 m |
| Defender corner lateral-error maximum | 0.358 m | 0.195 m |
| Attacker corner lateral-error p95 | 0.046 m | 0.040 m |
| Attacker corner lateral-error maximum | 0.225 m | 0.061 m |
| Corner time below 0.3 m/s, after startup, defender / attacker | 1.02 / 0.26 s | 0 / 0 s |
| Both lap goals finished | 82.96 s | 80.02 s |
| Unresolved defender / attacker action cycles | 794 / 10 | 557 / 0 |
| Collision / off-track / numerical failure / local stall | None | None |

Errors are relative to the smooth legal reference, not a claim that every
intentional recovery/defensive offset is tracking error. Metrics exclude the
first two seconds and classify corners by `abs(curvature) >= 0.05` or
`abs(reference_slope) >= 0.05`; RMS and percentiles use actual action durations.
The `rival_corner_metrics.py` utility reports these definitions and also wrapped
heading error and descriptive steering variation/reversals. This is one scenario
and seed—not a general cornering, defensive-performance or safety guarantee.
The adaptive run still deferred attacks and reported unresolved defender decisions
and CPU deadline misses.

Validation: **402 tests passed** in the GL-enabled suite (406 discovered, four
long opt-ins skipped there). The four configuration/lap/controlled-pass regression
tests passed separately, including new corner-error, no-near-stop and total-target
acceleration checks. Seven existing API/dynamics tests, dependency checks and
whitespace checks also passed. Historical feasibility snapshots explicitly retain
the pre-corner controller mode; they are not silently reinterpreted as new states.

```bash
python examples/rival_advanced.py --config examples/rival_advanced_fast.yaml \
  --scenario adaptive --duration 120 --render --output examples/rival_results/corners
python examples/rival_corner_metrics.py \
  --csv examples/rival_results/corners/00_adaptive_robust_defense.csv \
  --config examples/rival_advanced_fast.yaml
```

## Previous timing, feasibility, and passing improvements

The fast profile enables `compiled_attack_preview` as well as
`track_speed_planning`. The conservative default leaves both disabled. The new
`rival_attack_rollout.py` accelerates supported short previews; custom geometry,
policy subclasses, and staged/extended maneuvers retain the reference Python
path. `rival_defender_checks.py` accelerates the same nine opponent hypotheses,
swept SAT separation and progress calculations. Tests compare numeric results,
first commands, failure ordering and map checks against the reference paths.
There is no reduced hypothesis count, larger margin allowance, coarser time grid,
new dependency, or Gym physics change. Initial JIT/cache-load costs are still
reported; this is not a hard-real-time system.

The defender now logs the full rejection conjunction and tries a bounded
midpoint/hazard recovery search after the original candidates fail. New paths
change the action only if every original check accepts them. Failed hazard
probes preserve the established emergency selection. Racecraft compares offsets
at consistent reference stations, avoiding confusion between normal corner
following and a new lateral block; protected-body/overlap checks remain active.

The attacker evaluates staged **entry / pass / exit** opportunities rather than
requiring an offset lane over an unnecessarily long section of track. Candidate
checks predict defender reference motion, speed tracking, and braking. A committed
side cannot be swapped while alongside; aborts and front returns retain their
collision, headway and braking checks. The original short preview remains an
independent requirement. Opportunity geometry and timing are bounded sampled
assumptions—not a proof of full-maneuver safety or arbitrary-adversary coverage.

### Observed before/after on the default fast adaptive case

The same speeds, start, seed, margins, `inside` region and physics timestep were
used. Policies differ, so trajectories and action counts need not be identical.
Timings are observations on the development machine, not portable benchmarks.

| Metric | Earlier repaired fast profile | Latest run |
|---|---:|---:|
| Both lap goals complete | Yes, 82.54 simulated s | Yes, 82.96 simulated s |
| Wall duration, headless | 206.28 s | 79.69 s |
| Pipeline p50 / p95 | 43.21 / 99.55 ms | 8.86 / 44.73 ms |
| 20 ms pipeline misses | 4,117 / 4,127 cycles | 982 / 4,148 cycles |
| Defender unresolved cycles | 660 | **794** |
| Attacker unresolved cycles | 13 | 10 |
| Collision / off-track / numerical failure / local stall | None | None |
| Executed sampled-feasible attacks | 0 | 0 |

**The live defender unresolved count did not improve.** Same-state replays
confirmed several genuine candidate-coverage fixes, but they do not establish a
closed-loop reduction: the changed actions lead to different subsequent states.
The latest run still has no valid executed attack, so it is not evidence of
competitive defensive performance. The pipeline still misses its 20 ms budget,
and simulated sensing/actuation does not acquire wall-time latency from those
misses. There is no claim of safe real-time vehicle operation.

### Controlled passing probe

`rival_passing_demo.yaml` deliberately uses a 0.5 m/s defender and 2.0 m/s attacker,
start station 22 m and a 6 m initial gap, preserving the inside region and margins.
This is a separate speed-advantage probe, not an easier substitute for the default
adaptive evaluation:

```bash
python examples/rival_advanced.py --config examples/rival_passing_demo.yaml \
  --scenario left-pass --duration 20 --render --compare \
  --output examples/rival_results/passing_demo
```

The 20-second headless comparison recorded a sustained pass at **5.12 s** and
401 executed sampled-feasible attack steps in **each** run, with no collisions,
off-track events, numerical failures, local stalls or attacker-unresolved steps.
The defender still reported unresolved decisions. Defense-on and no-defense pass
times were equal: this validates an executable passing path, **not an advantage
for the defensive policy**. Neither car completed a lap in this short probe.

The sample track's minimum inside legal center-band width is about **0.095 m**,
less than the configured **0.43 m** center separation. Many local attempts must
therefore defer. This is a measured local constraint, not a global impossibility
proof. `--region track` remains an explicit opt-in relaxation, never applied
silently to make a pass happen.

### Diagnostics and regression commands

Final validation on Python 3.15.2 / Fedora Linux: **364 tests passed** in the
GL-enabled suite (368 discovered, four long regressions skipped there). The four
configuration/lap/controlled-pass tests then passed with `RIVAL_FAST_TESTS=1` and
`RIVAL_PASS_TESTS=1`; seven existing API/dynamics tests passed separately.
`git diff --check` and `python -m pip check` also passed. The old 60/300-second
conservative-profile opt-ins were not rerun as part of this latest validation.

CSV adds `planner_candidate_*`, `planner_selected_*`, attacker plan/maneuver
latencies, maneuver phase and opportunity reason. Candidate rejection categories
can overlap; JSON aggregates candidates only on solve cycles and selected failures
on every action cycle. These fields expose failure causes rather than renaming
unresolved decisions as safe. JSON also records passing-opportunity counts and
separate attacker planning percentiles.

```bash
# Display-independent tests; expensive real-Gym regressions stay opt-in.
python -m unittest discover -s examples -p 'test_rival_*.py' -v
# Actual lap and controlled matched-pass regressions (can take minutes):
env RIVAL_FAST_TESTS=1 RIVAL_PASS_TESTS=1 python -m unittest discover -s examples -p test_rival_fast_run.py -v
```

## Repaired fast-profile controller (preceding implementation)

`robust.track_speed_planning: true` opts into the shared `SpeedProfile` and the
attacker improvements. It is enabled in `rival_advanced_fast.yaml`, not in the
conservative default. The selected-reference speed plan combines offset-path
curvature, lateral acceleration, lateral-target-rate constraints, and whole-lap
backward braking propagation. Front-preview queries reserve response/jerk distance
and include every intervening sampled node, including across lap closure. Both
Python and compiled defender rollouts use the same cap; attacker previews also
apply it at each predicted station. This remains a reference heuristic, not a
speed guarantee for arbitrary tactical offsets.

The attacker now clones a speed/acceleration state for each candidate and commits
only the selected first command, with bounded acceleration and jerk. Recovery
checks useful nominal/hold speeds before absolute crawl speeds; a previously slow
hold cannot permanently hide a newly feasible faster nominal route. Every
continued recovery is freshly checked. Hard footprint, legal-band, collision and
tracking-budget checks remain active; an already invalid margin still reports
unresolved rather than silently expanding the road.

Fast previews use sampled defender hold/reference-following hypotheses with
constant speed or bounded acceleration, plus oriented-body separation and
between-node padding. A pass also needs estimated room for its full completion,
lateral transition and braking—not just today's short preview. Infeasible passes
are deferred, and a freshly checked nominal return is preferred before the cars
become alongside. Following demand accounts for response, jerk, braking distance
and headway. This deliberately favors controlled following on narrow sections.
It does not model every defensive maneuver or arbitrary opponent braking/ramming.

### Historical repaired-profile validation, before the latest changes

The then-current adaptive fast-profile run completed both lap goals at **82.54 simulated
seconds**, within a 120-second budget, in about **206 wall seconds**:

| Metric | Defender | Attacker |
|---|---:|---:|
| Lap goal finish time | 82.23 s | 82.53 s |
| Maximum measured speed | 3.18 m/s | 3.60 m/s |
| Maximum command jerk | 6.00 m/s³ | 6.00 m/s³ |
| Collision / off-track / numerical failure | 0 / 0 / 0 | 0 / 0 / 0 |
| Maximum local-stall duration | 0 s | 0 s |
| Unresolved decision steps | 660 | 13 |

Cruise targets can be briefly exceeded by the jerk-limited command filter and
simulator speed tracking; they are not hard clamps on measured speed. All 4,127
CSV action rows were checked for failures, stalls, and command-jerk limits. There
was **no pass and no executed sampled-feasible attack**: this run demonstrates
track completion/command smoothness, **not effective defense against an attack**.
The evaluator correctly marks the attack ineligible. CPU deadline misses remain;
OpenGL/NVIDIA does not accelerate the planner. Other starts, seeds, scenarios,
multiple-lap goals, and real-vehicle operation are not established by this run.

The GL-enabled RIVAL suite passed **282 tests**, with three long opt-ins skipped
(285 discovered). The opt-in fast lap regression was then rerun successfully:
its three configuration/integration tests passed in about 226 wall seconds.
Coverage includes Python/JIT speed-cap parity, lap wrap, future narrowing,
jerk-limited stop/restart, candidate command-state isolation, unchanged hard
margins, recovery selection and rendered/headless command parity.

```bash
# Normally skipped because it takes several minutes; uses the actual Gym/map.
env RIVAL_FAST_TESTS=1 python -m unittest discover -s examples -p test_rival_fast_run.py -v
```

CSV includes attacker/defender profile caps, attacker first-command acceleration,
and an explicit label for the assumed defender model in attacker reasons. JSON
records profile construction diagnostics. Those are not safety certificates.

## 3D race-control presentation

`--render` now opens a standalone, resizable OpenGL viewer rather than modifying
Gym's original renderer. It includes lit 3D RC-car bodies and wheels, a dark
occupancy-map-derived track surface, cyan/red car identification, racing line,
target markers, the selected rollout, and a screen-space speed/lap/decision HUD.
The car meshes stay inside the configured XY vehicle footprints; decorative
height does not introduce a new collision shape or change contact physics.

- **C** cycles follow, chase, isometric, and top-down perspective cameras.
- **Mouse wheel** zooms; **H** toggles the HUD and planning overlays.
- **Escape / window close** ends the run as `viewer_closed`, saving the CSV and
  summary. This is not lap completion or a successful defensive result.
- `--render-speed 1` (default) paces presentation toward real time; `2` requests
  double speed; `0` runs unthrottled. Slow CPU planning can still run slower than
  the requested pace. Headless runs are never paced.

The viewer uses existing pyglet 1.5 compatibility OpenGL and needs a display
(or an Xvfb display for offscreen GL tests). No game-engine dependency is added.
It displays actual observation poses, not a separate animated vehicle simulation.
The existing Gym single-track/tire/contact model and RK4 integration remain the
source of motion: no added downforce, banking, suspension, 3D tire forces, or
F1-scale vehicle dynamics are implied by the graphics. NVIDIA/OpenGL may draw
frames; planning and physics remain on the CPU. `--viewer legacy` keeps the
original renderer for troubleshooting and existing workflows.

## Racecraft-inspired defensive model

`rival_racecraft.py` adds stateful permissions and utility above the existing
robust trajectory checks. It borrows **principles**, not an official rule set,
from circuit racing and autonomous racing:

1. **One early lane choice per encounter.** Choose a fixed offset relative to
   the smooth legal reference, favoring the inside of a distant bend or the
   initially threatened side on a straight. Do not continuously chase changes
   in the attacker's lateral position. Only a feasible winning rollout consumes
   the move; an infeasible proposal does not authorize trying the other side.
2. **No late reactive block.** Uncertain closing time must exceed both the TTC
   floor and a lateral-maneuver time budget (translation, settling, measurement
   age, and buffer). A late lock remains latched for that encounter.
3. **Commit before braking/turn-in.** Curvature is previewed over travel during
   `racecraft_corner_time` plus braking distance and pursuit lookahead. A nearby
   corner locks out new tactical moves; checked corner-tracking recovery remains
   available when longitudinal separation allows it.
4. **Respect overlap on either side.** Reject sampled body motion toward the
   occupied side, including residual steering/target momentum. At predicted
   longitudinal overlap, reserve a car-width plus separation/uncertainty margin
   in the existing legal corridor when it can fit two cars. Never widen the
   inside-half band to manufacture passing room. If that band cannot fit the
   extra room reserve, the original map/SAT separation checks still apply;
   neither a legal passing opportunity nor rule compliance is claimed.
5. **Retain encounter memory.** Dropouts, noisy side changes, old commitment
   timers, and defense toggles do not grant a second move. Reset only after a
   continuously observed separated retreat or clear pass, with a dwell period.
   Disabling defense removes tactical lane utility but retains collision and
   overlap checks for a matched no-defense comparison.

Acceleration, jerk, steering and target slew, curvature-based speed limits,
map/whole-body legality, and uncertain opponent hypotheses remain active. A
racecraft rejection can only remove feasibility, never certify a path that
failed another check. If all candidates fail, the output is still an explicitly
unresolved `emergency` fallback—not a claimed safe hold or permission to squeeze.

The `robust` YAML section exposes lane displacement, early reaction buffer,
corner curvature/preview, overlap/room margins, and reset gap/dwell. See
`RobustConfig` for all defaults. Distances are scaled-car metres and speeds m/s,
not full-size F1 values. The old `commitment_time` field remains accepted for
configuration compatibility; it no longer unlocks another defensive move.

CSV adds `racecraft_phase`, `racecraft_reason`, `racecraft_encounter`,
`racecraft_committed_bias`, `racecraft_move_used`, `racecraft_locked`, and
`racecraft_safe`. JSON summarizes encounters, committed encounters, phase counts,
and selected policy-check failures. `racecraft_safe` means only that the additional
sampled racecraft filter accepted the selected path; use the independent overall
`feasible` and evaluator failures too. Neither flag is a safety certificate.

This is a more interpretable simulation baseline, **not a validated F1 or Indy
Autonomous Challenge controller**. No official stewarding/one-move regulation,
right-to-a-corner adjudication, strategic tire degradation, aero/wake effects,
friction estimation, or protection against arbitrary ramming is implemented.
The truth-derived synthetic sensor and approximate prediction limitations remain.
The default `--region inside` still enforces the project's scaled inside-half
interpretation. `--region track` explicitly relaxes it to the full mapped legal
corridor for circuit-racing experiments; it does not activate official F1 rules.

## Verification of the presentation/racecraft upgrade

Tested with Python 3.15.2, Gym 0.26.2 and pyglet 1.5.31 on Fedora Linux:

- At the presentation/racecraft-upgrade stage: 212 tests discovered; the five display/long-run opt-ins are
  skipped by default. With `RIVAL_VIEWER_GL_TEST=1`, **210 passed**, leaving only
  the two long-run tests skipped.
- Final 60-second and 300-second-budget real-Gym adaptive regressions passed.
  The latter took about 204 wall seconds; these tests remain goal-bounded, not
  forced five-minute sessions. Their rolling progress, failure and numerical
  checks remain enabled.
- Hidden-window integration verified identical headless/rendered commands and
  simulated outcomes, plus clean explicit user-close handling. All four camera
  modes, screenshot capture and window events passed the GL smoke test.
- Four-second rendered defense/no-defense runs on the sample map completed
  without collision, off-track or numerical failure. Neither recorded a pass;
  together they still reported **166 unresolved controller steps**. Both short
  runs missed CPU deadlines. This does **not** establish defensive superiority.
- Existing compatibility/API and dynamics tests: **7 passed**. Broad discovery
  over `gym/f110_gym/unittest` also imports interactive demos, so use the targeted
  commands below. A package-qualified dynamics invocation can hit an existing
  Numba cache module-name mismatch; discovery keeps the expected module name.
- `git diff --check` and `python -m pip check` passed. Runtime test success is
  not a claim that all existing editor/static-typing diagnostics are resolved.

```bash
python -m unittest discover -s examples -p 'test_rival_*.py' -v
# OpenGL display required; omit this variable for headless-only unit tests
# In Fish, use env as shown; this also works in Bash.
env RIVAL_VIEWER_GL_TEST=1 python -m unittest discover -s examples -p 'test_rival_*.py' -v
env RIVAL_LONG_TESTS=1 RIVAL_SOAK_TESTS=1 python -m unittest discover -s examples -p test_rival_longrun.py -v
python -m unittest discover -s gym/f110_gym/unittest -p compatibility_test.py -v
python -m unittest discover -s gym/f110_gym/unittest -p dynamics_test.py -v
```

## Longer runs and the corner-stall fix

```bash
# Up to five simulated minutes; also stops when both cars finish the lap goal
python examples/rival_advanced.py --scenario adaptive --duration 300 --render

# Optional fail-fast diagnostic, not an automatic reset or successful finish
python examples/rival_advanced.py --scenario adaptive --duration 300 --stall-timeout 10

# Real-Gym long regressions (normally skipped to keep unit tests quick)
RIVAL_LONG_TESTS=1 RIVAL_SOAK_TESTS=1 python -m unittest discover -s examples -p test_rival_longrun.py -v
```

`--duration` is a **maximum simulated duration**, not a wall-time promise. The
default goal is one lap per car, so the repaired default adaptive run finishes
at about **113.6 simulated seconds**, before a 300-second limit. Long-test names
refer to their time budgets; early genuine lap-goal completion is accepted.
The tests do not silently restart, reset a stuck car, or ignore Gym's done signal.

Several distinct causes were fixed:

1. **Numerical instability:** 20 ms RK4 integration produced exploding yaw/slip
   near the dynamic model's low-speed switch. Replaying identical commands at
   10 ms and 5 ms removed that explosion. The advanced runner retains 20 ms
   commands and uses `robust.max_physics_dt: 0.01` to substep the unchanged Gym.
   Any inner-step collision, off-track, done, or numerical failure stops the action
   immediately and preserves partial elapsed time. Nonfinite observations or
   yaw rates over 100 rad/s are explicitly reported, never called a safe stop.
   The simulator's sample-based steering delay consequently has a different
   physical duration at the smaller integration step; this is recorded rather
   than claimed to be bit-for-bit equivalent to the unstable 20 ms simulation.
2. **Late corner preparation:** a bounded periodic smooth reference anticipates
   required offsets. Lookahead targets are checked against the band at their
   lookahead station; the current body is checked at its own station. Speeds
   anticipate spatial-reference slope as well as curvature.
3. **Safe-looking standstill:** recovery offset/crawl candidates are available
   without a passing threat, and a checked forward route wins over unnecessary
   standstill. Failing map/collision checks still cannot be forced through.
4. **Insufficient full-body reference clearance:** reference refinement checks the
   oriented rectangle, not just the center offset, inside the original legal
   band. `reference_clearance_reserve: 0.12` adds preparation room **on top of**
   the unchanged 0.04 m runtime boundary margin. Empty body-clear bands fail
   explicitly. Nodal clearance is not a swept tracking guarantee.
5. **Attacker recovery deadlocks:** nominal catch-up replaces edge chasing when
   there is no nearby passing opportunity. Guarded recovery searches legal
   corrections/crawl speeds and requires actual predicted progress. Both cars
   use the same explicit body-rule tracking budget; existing rule errors may
   request strictly improving repair, not passing credit. Hard wall-margin,
   collision, and target-legality checks remain separate and observable.

CSV rows are streamed per **action cycle**, with inner-physics-step counts and
partial-cycle timing. Checkpoints are written while running, labeled incomplete;
only final run/aggregate JSON is authoritative. `--progress-interval 10` prints
periodic simulated time, wall time, per-car progress and rolling stall state.
Five-second signed-progress windows detect local stops that whole-run averages
miss; `--stall-timeout` terminates explicitly as `stalled`, never success.

Historical corner-stall-fix validation **before the racecraft upgrade**, Python
3.15.2 / Gym 0.26.2 (these timings are not a benchmark for the new policy):

| Outcome | Defender | Attacker |
|---|---:|---:|
| Lap goal finished at | 83.30 s | 113.60 s |
| Last-five-second progress at finish | 8.210 m | 8.666 m |
| Maximum local-stall duration | 0 s | 0 s |
| Collision / off-track / numerical failure | 0 / 0 / 0 | 0 / 0 / 0 |

At that stage, the standard suite passed 152 tests (the two expensive long tests are skipped
unless enabled); nine existing simulator/API/dynamics tests also passed. The
real-Gym 60-second regression and the up-to-300-second goal-bounded regression
both passed. CI runs the 60-second regression on Python 3.15. They independently reconstruct rolling progress from CSV, check every
step for failures, and do not rely only on the final summary. This establishes
that the repaired **default scenario** keeps moving through the track; it does
not establish all scenarios, official rule compliance, safe real-time deployment,
or superior defensive performance. Unresolved model decisions and CPU deadline
misses remain reported.

## What changed

### 1. The defender no longer consumes exact opponent state

`rival_perception.py` defines an acquisition-timestamped measurement interface,
a bounded filter in wrapped raceline coordinates, uncertainty growth, and two
observation adapters:

- **`simulated` (default):** seeded position/speed noise, acquisition period,
  delivery latency, and packet dropout. This adapter starts from simulator truth.
  It is a sensor-imperfection experiment, **not actual perception**.
- **`lidar`:** converts the actual simulator scan into world points, subtracts
  map returns using bitmap clearance, clusters remaining adjacent points, and
  associates detections with the previous candidate. It never looks up the
  opponent's truth pose to identify a cluster. However, it is only a heuristic
  dynamic-object detector: walls, unmodelled objects, and association errors can
  produce misses or false positives.

The default 4.7-radian LiDAR has a rear blind sector. A car directly behind may be
unobservable; the controller must not invent a rear detection. A real defender
requires validated rear sensing or another independent tracking source.

Out-of-order, future, nonfinite, stale-acquisition, and implausibly jumping packets
are rejected. Track extrapolation carries explicit age, speed uncertainty,
position uncertainty, acceleration, and lateral-rate estimates. These uncertainty
scales are **heuristic, not calibrated confidence intervals**. Own localization
still comes from the simulator. The adversary and evaluator deliberately retain
truth; the defensive policy receives only the tracker output.

### 2. A trajectory lattice replaces offset-only safety checks

`rival_robust.py` replans at 10 Hz by default and issues commands every 20 ms. It
compares offset-bias and speed candidates over a 1.5-second horizon, with:

- Command filters integrated at the **actual actuation cadence**, not the coarser
  planning period. `prediction_dt` is an upper bound on check spacing; the default
  checks use the 20 ms command interval. Actual Gym integration is substepped at
    10 ms by default; command timing and jerk limits are unchanged.
- Bounded acceleration, braking, emitted-command jerk, steering slew, lateral
  target velocity, and lateral target acceleration. Braking reserves speed to
  release deceleration smoothly instead of clipping at zero and violating jerk.
- Body-forward speed reconstructed from the longitudinal projection for XY/yaw
  integration; heading configurations where that projection is unreliable are
  explicitly degraded.
- Curvature-dependent speed limits and longitudinal lead-car headway/braking
  caps. This is no longer just fixed cruise speed or a binary stop request.
- Nine opponent hypotheses combining slow/braking, estimated, and fast/accelerating
  longitudinal behavior with left, measured, and right lateral motion. Initial
  speed uncertainty is propagated through those speed-limited hypotheses, with
  additional position/model inflation.
- Oriented-rectangle separating-axis checks in world coordinates, enlarged for
  uncertainty and relative motion between samples. Road checks certify the body
  rectangle, including its interior, at every predicted pose.

This is finite-hypothesis, kinematic model-predictive **sampling**, not an exhaustive
adversarial reachable-set computation or a formal safety shield. Acceleration,
steering dynamics, tire slip, friction, latency, and attacker policies can differ
from the model. The true Gym dynamics remain the independent evaluator.

`rival_rollout.py` compiles only the numeric trajectory-generation kernel using
Numba without fastmath. Map and opponent rejection still execute unchanged. A
readable Python reference is retained and parity-tested across acceleration,
braking, yaw misalignment, wraps, nonuniform steps, and invalid corridors. Batched
footprint queries accelerate checks without lowering their sampling density.

### 3. Defense is conditional, with explicit failure modes

The original IDM-inspired closing-gap trigger remains interpretable. Candidate
utility now also values predicted progress and penalizes unnecessary movement.
Safety rejection precedes usefulness scoring.

- **Cruise/follow:** use nominal legal geometry and speed/headway limits.
- **Defend:** consider a useful offset only while the attacker is behind, sensing
  is fresh, and estimated time to collision leaves room for a maneuver.
- **Yield:** avoid initiating movement into an alongside/too-close opponent's lane.
  Existing lateral momentum is predicted; excessive residual approach is marked
  unresolved rather than claiming an achieved safe yield.
- **Degraded:** no tactical defense with absent, stale, invalid, or known
  out-of-model measurements. Speed is reduced and available predictions are
  inflated. Lost sensing does **not** imply the road is empty.
- **Emergency:** if no sampled candidate passes all checks, choose among remaining
  map/legal candidates using a minimax-style separation fallback. Depending on the
  rear hazard, this may select escape/progress rather than automatically stopping
  in front of the attacker. It remains explicitly **not modeled-safe**.

Known estimated speed, acceleration, and lateral-rate bound violations disable
blocking. Evaluator truth checks separately reveal bounds violated before the
tracker recognizes them. No bounded prediction protects against arbitrary ramming.

The racecraft encounter state now enforces a single early lane commitment,
late/corner locks, and overlap room checks rather than timer-only side-switch
hysteresis. Disabling tactical defense forces a new plan immediately; invalid
sensor packets cannot pass through as valid state.

### 4. Legal geometry is still an explicit approximation

`TrackEnvelope` precomputes periodic map-derived cross-sections. Unlike the old
abrupt curvature-half switch, it blends from a full near-straight corridor to the
estimated inside half in sustained curvature. This is a **changed, smoother
research interpretation** of the inside-half rule, not official IAC compliance.
Map footprint checking remains online even when envelope interpolation is used.

The default bitmap search is capped by `settings.max_offset_search`; caps and
invalid sections are exposed by the envelope object. A capped cross-section or
connected open area can bias the inferred track center. Constant road width is
not assumed. `--region track` explicitly relaxes the inside-half interpretation;
it must not be silently used to claim inside-region results.

For surveyed boundaries, set `robust.boundaries_path` to a CSV relative to the YAML:

```text
s,lo,hi
```

Each row contains arc length and physical boundary offsets from the racing line,
**before** vehicle half-width and margin subtraction. Supply at least three finite
rows, increasing from `s=0` to a value less than lap length; the last-to-first span
is periodic. Validate density and topology externally. No invented boundary file
is supplied for the sample map.

Predicted legal checks allow `lateral_tracking_margin` and decaying existing
tracking error. They are not perfect physical rule compliance. The evaluator
separately logs measured legal-violation duration/magnitude with a 2 cm numerical
allowance, distinct from raster off-track failure.

### 5. Opponents and evaluation are harder—and their failures are visible

Scenarios: `no-attack`, `approach`, `left-pass`, `right-pass`, `adaptive`, `feint`,
`surge`, `cut-in`, and `brake`. The attacker has perfect simulated state to challenge
the noisy defender; it is not representative of a validated human/robot opponent.

Adaptive attacks reassess the free side every 0.5 s, with 0.8 s commitment and
hysteresis. Feints switch intention once. Surge increases closing speed. Cut-in
and brake behavior apply after gaining a lead. Collision guards examine swept
lateral intervals, not only endpoints. Preview uses an XY bicycle path with the
same pursuit, command timing, and slew limits as attacker execution. Unsafe
requests try reduced-speed hold/nominal paths before unresolved bounded braking.

These guards deliberately make the attacker bounded rather than a suicidal
rammer. They can also make an attack ineffective. The evaluator distinguishes:
requested corridor validity, requested sampled-path feasibility, guard intervention,
selected fallback, executed attack steps, unresolved preview, stalling, and
collision/off-track failure. An ineffective or failed attacker is **not a
successful defense**. The conservative invalid-attack flag includes unresolved
preview during setup as well as the passing attempt; consult step logs for timing.

Smoke/stress suites vary geometry, initial gap, closing speed, seed, sensor latency,
and dropout—not just random seed. They are small deterministic falsification
suites, not a statistical safety demonstration. The default synthetic speed bound
of 4 m/s is an explicit assumption for this suite, not the physical limit of an
unknown adversary.

## Output and comparison semantics

Each action cycle is written to CSV, with counts/timing for its physics substeps,
plus strict JSON run/aggregate summaries:

- Control mode/reason, commands and derivatives, target and actual Frenet states.
- Hypothesis clearance, sensor age/error/availability, and unresolved decisions.
- Per-car collision, raster off-track, legal-region error, progress/laps/time.
- A pass based on **accumulated unwrapped relative progress**, requiring a full
  vehicle-length-plus-margin lead for 0.5 s. Timestamp is the beginning of the
  confirmed qualifying lead, not the finish-line crossing or confirmation time.
- Solver p50/p95/max and misses against `control_period`; detection, tracking,
  full observation-to-command time, and misses against the physics timestep.
- Sensor/model assumption violations evaluated from truth, separately from the
  information available to the defender.
- Invalid/eligible attacks and defender versus attacker failures. Automated
  `defended_attack_credit` is deliberately false: no finite run proves success.

`min_clearance` is a conservative center-disk lower bound, not measured body
clearance: it can be negative even when the finer oriented-rectangle footprint
passes. `min_predicted_separation` includes hypothesis inflation; it is not the
measured car-to-car distance.

Synchronous wall-clock computation does **not** add simulated actuator delay. A
20 ms deadline miss is therefore a real-time deployment warning, not an injected
late-actuation experiment. Pipeline timings exclude `env.step` and rendering.

`--compare` switches tactical defense off while keeping the robust controller's
sensing, shaping, and safety machinery. `--controller legacy` instead runs the
original Defender/TrackRules/pursuit at physics cadence, supplied the same estimator
and adversary. Its shaping, legal-region estimator, and replanning cadence differ;
it is an informative baseline, not a perfectly isolated single-factor ablation.
The original `rival_baseline.py` remains available unchanged.

## Earlier short-run validation: Python 3.15.2, Gym 0.26.2

These measurements describe the earlier implementation before the long-run fixes
above; preserve them as failure evidence, not current timing/performance numbers.

- **103 RIVAL tests passed**; **7 existing Gym/API/dynamics tests passed** at that revision.
- Twelve current-config stress cases, up to 4 simulated seconds each: **0 collision
  runs, 0 raster off-track runs, 0 defender-failure runs**. However, **9 attacks
  were invalid/ineffective**, only **3 were eligible attack tests**, and there were
  **1,408 unresolved defender steps**. No laps or sustained passes completed.
- Stress solver p95 ranged **17.4–115.5 ms**, with up to **8 missed 100 ms planning
  budgets in one run**, plus **1,363 total 20 ms observation-to-command deadline
  misses**. These measurements do **not pass a real-time readiness gate**.
- A 1-second real-scan detector run and a 0.2-second rendered comparison completed
  without collision/off-track. The short LiDAR run supplied no eligible attack
  test. Rendering used headless OpenGL; interactive controls were not hand-tested.
- Earlier eight-second matched adaptive runs exposed both excessive yielding and
  attacker setup failures. They were used to falsify assumptions, not reported as
  performance wins. The evaluation does **not establish superiority over the
  legacy defender**, much less safe real-world defense.

Reproduction commands and generated evidence:

```bash
python examples/rival_advanced.py --suite stress --duration 4 \
  --output examples/rival_results/release_stress_final
python examples/rival_advanced.py --sensor lidar --duration 1 \
  --output examples/rival_results/release_lidar_final
```

Results are intentionally ignored by Git. Each output folder contains
`summary.json`, run JSON, and per-step CSV. Timing is machine/load/JIT-cache
sensitive. Preserve config, code revision, and environment when comparing runs.

## What blocks real-car use

Do not connect this runner to a vehicle. Before even supervised closed-course
hardware work, this needs calibrated front/rear tracking and uncertainty, surveyed
track boundaries and authoritative rules, identified dynamic/friction/braking
models, tested worst-case timing and actuator delays, full-lap and randomized
adversarial evaluation with competent attackers, and independent watchdog,
emergency-stop, geofence, and low-speed safety procedures. Start with log replay
and hardware-in-the-loop under qualified supervision. The current code does not
provide those validations or safety systems.
