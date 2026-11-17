# R.I.V.A.L. Simulation — Presentation Notes

## Project overview

- R.I.V.A.L. is a research simulation of two F1TENTH-scale cars: a defending car and an attacking car.
- The advanced controller combines state estimation, trajectory prediction, motion planning, and vehicle control to study their interaction on a track.
- The recent work focused on making both cars track the racing line more smoothly through corners in the normal fast configuration.
- The simulator's vehicle physics and observation interface were left unchanged; improvements were made in the controller and its predictions.

## What was changed

- Enabled corner tracking in the fast configuration, with a gain of `4.0`.
- Added bounded feedforward to the defender's lateral target. This lets it anticipate how its reference line moves along the track instead of treating the changing target as stationary.
- Kept the defender's total lateral target rate and acceleration bounded at `0.45 m/s` and `0.8 m/s²`.
- Improved acceleration release near the desired-speed ceiling for both cars. This reduces unnecessary overshoot and abrupt follow-up corrections.
- Kept the command-jerk bound at `6 m/s³` and applied matching control calculations in the controller's readable and JIT prediction paths.
- Retained the fast profile's moderate cruise targets (`3.0 m/s` defender, `3.4 m/s` attacker) rather than requesting the simulator's maximum speed. The faster target is not automatically a faster or more reliable lap.
- Left the conservative configuration's corner-tracking option disabled so it remains available for comparison.

## Measured cornering results

Fixed-seed, full-lap trial in the adaptive fast scenario:

| Metric | Before | After |
|---|---:|---:|
| Defender corner lateral-error 95th percentile | 0.298 m | 0.105 m |
| Defender maximum corner lateral error | 0.358 m | 0.195 m |
| Attacker corner lateral-error 95th percentile | 0.046 m | 0.040 m |
| Attacker maximum corner lateral error | 0.225 m | 0.061 m |
| Time below 0.3 m/s in corners after startup, defender / attacker | 1.02 / 0.26 s | 0 / 0 s |
| Time for both lap goals to finish | 82.96 s | 80.02 s |
| Unresolved defender / attacker action cycles | 794 / 10 | 557 / 0 |

- Lower corner-tracking errors mean the cars stayed closer to their smooth legal reference through the measured corners.
- Neither car had a corner stop below `0.3 m/s` after the first two seconds in the updated run.
- The defender's unresolved action cycles decreased, but **557 remained**. This is an important limitation, not a solved safety problem.
- No collision, off-track event, numerical failure, or local stall was recorded in this particular trial.

## Validation

- The GL-enabled test suite reported **402 passed**; four opt-in tests were skipped in that suite.
- A separate fast-profile lap and controlled-pass regression suite reported **4 passed**.
- Existing Gym API/dynamics tests reported **7 passed**.
- These results validate specific tests and scenarios. They do not establish performance across all tracks, seeds, speeds, or adversarial behavior.

## Limitations and honest interpretation

- This is a research simulation, not a controller approved for a real vehicle. There is no hardware actuation interface or real-world safety certification.
- Sampled feasibility checks are not a guarantee of collision avoidance.
- Unresolved decisions and CPU deadline misses remain; the planner does not always find an accepted action on time.
- The measured adaptive run did not demonstrate a pass. A separate deliberately slow-defender probe demonstrated that an attack could pass, but defense-on and no-defense runs passed at the same time. That probe does **not** demonstrate defensive superiority.
- The simulator's configured maximum speed is `20 m/s`; the fast profile's `3.0/3.4 m/s` values are controller cruise targets, not maximum-speed claims. A request for `26 m/s` exceeds the configured vehicle limit and has not been validated.
- Rendering on an NVIDIA GPU affects visualization only; the simulation's physics and planning are CPU work.

## Suggested short spoken summary

> “I improved corner tracking for both cars by making the defender anticipate its moving reference line and by smoothing how both cars release acceleration near their speed targets. In a fixed-seed full-lap simulation, the defender's 95th-percentile corner error dropped from 0.298 to 0.105 meters, and the attacker's maximum corner error dropped from 0.225 to 0.061 meters. Both cars completed the lap goals without a recorded collision or off-track event in that run. However, the planner still reported unresolved defender decisions, so these results are a measured improvement in one simulation—not a safety guarantee or proof that the defender can always stop an attack.”

## Related files

- `examples/rival_advanced_fast.yaml` — fast controller configuration.
- `examples/rival_control.py` — shared longitudinal command logic.
- `examples/rival_corner_metrics.py` — corner-tracking metric calculations.
- `examples/README_RIVAL_ADVANCED.md` — controller details, run instructions, and validation context.

