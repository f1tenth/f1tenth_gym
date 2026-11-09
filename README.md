![Python 3.11–3.15](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
Use Python 3.11–3.15 in a fresh virtual environment (Python 3.15 verified).
Do not reuse an environment containing the old NumPy/Gym pins. From a checkout:

```bash
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Compatibility and testing

The simulator retains its existing API: both `reset(poses)` and `step(actions)`
return `(observation, reward, done, info)`. Physics, observations, control inputs,
and rendering behavior are unchanged. Gym's newer API-checking/order wrappers are
disabled for this legacy interface. Do not assume compatibility with Gym's modern
wrappers or third-party tooling that requires a five-value step result.

Gym 0.26.2 and pyglet 1.5 are intentionally retained instead of migrating to
Gymnasium or pyglet 2, which would require API and renderer changes. Gym prints
an upstream NumPy 2 warning; the simulator's own unwrapped paths are covered by
regression tests, not all Gym environments/wrappers.

Run the headless compatibility, dynamics, and collision correctness tests with:

```bash
python -m unittest f110_gym.unittest.compatibility_test f110_gym.envs.dynamic_models f110_gym.envs.collision_models.CollisionTests.test_random_collision f110_gym.envs.collision_models.CollisionTests.test_multiple_collisions -v
```

Rendering still requires a working display and system OpenGL/GLU libraries
(e.g. `libgl1` and `libglu1-mesa` on Debian/Ubuntu). Python package installation
alone does not supply these native libraries. Non-rendering simulation does not
need a display. Windows and macOS GUI compatibility has not been verified.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
