# R.I.V.A.L. advanced simulation controller

**Research simulation only. Not approved for a real vehicle, not a guarantee of
collision avoidance, and not proof of effective defense against an adversarial
car.** This upgrade adds stronger estimation, prediction, control, and falsification
tests. Its current validation still finds unresolved safety decisions, ineffective
attacks, model-bound violations, and real-time deadline misses. There is no robot
actuation interface or hardware arming path.

The original `rival_baseline.py` and its policy are retained for comparison. Gym
physics, observations, and the four-value `reset`/`step` API are unchanged. Actions
remain rows of `[steering_angle, speed]`.

## Run

From the repository root, with the project installed (Python 3.15.2 was tested):

```bash
# Default: simulated noisy/delayed observations, adaptive attacker, headless
python examples/rival_advanced.py

# Matched tactical defense / no-defense, same sensor seed and initial conditions
python examples/rival_advanced.py --scenario adaptive --compare

# Original Defender logic, same estimator and stronger adversary
python examples/rival_advanced.py --controller legacy --scenario adaptive --compare \
  --output examples/rival_results/legacy_comparison

# Render at 10 Hz without slowing every physics step with human-mode sleep
python examples/rival_advanced.py --scenario adaptive --render --compare

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

The first trajectory solve may compile/load a Numba kernel and take longer.
Compilation and inference timings are not hidden by the metrics. A hardware
system would need an explicit warmup/arming gate and an independently enforced
control deadline; this runner supplies neither.

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
  checks use the smaller 20 ms physics interval.
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

Commitment/hysteresis limits side-switching. Disabling tactical defense forces a
new plan immediately; invalid sensor packets cannot pass through as valid state.

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

Each physics step is written to CSV, plus strict JSON run/aggregate summaries:

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

## Final validation: Python 3.15.2, Gym 0.26.2

- **103 RIVAL tests passed**; **7 existing Gym/API/dynamics tests passed**.
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
