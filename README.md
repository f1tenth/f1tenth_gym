# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
You can install the environment by running:

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym.git
$ cd f1tenth_gym
$ git checkout exp_py
$ pip3 install --user -e gym/
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

## Environment Details
### Observation
```python
{
  "edo_idx": 0
  "scans": [array([0.215, ..., 30.028], shape=(1080,))],
  "poses_x": [-46.76866474190183],
  "poses_y": [16.821669212305586],
  "poses_theta": [5.140745068338013], # radian: (0~2π),
  "linear_vels_x": [5.31381482205777],
  "linear_vels_y": [0.0],
  "ang_vels_z": [0.22991898368161137],
  "collisions": array([0.]),
  "lap_times": array([11.14]),
  "lap_counts": array([0.])
}
```
## Action
```python
{
  "steer": 0.0316930846048194,
  "speed": 5.393091216421344
}
```
## Reward
```python
reward = 0.01
```

## Known issues
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.

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
