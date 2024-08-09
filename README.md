![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.20
```
And you might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
```
which could be ignored. The environment should still work without error.

- Invalid metadata: Expected end or smeicolon (after version specifier)
>WARNING: Ignoring version 0.19.0 of gym since it has invalid metadata:
Requested gym==0.19.0 from file:///home/\<user\>/.cache/pip/wheels/c8/24/9d/db9869e09d1fbf12a10ce64362d9f161f09bdb5651e44317fe/gym-0.19.0-py3-none-any.whl (from f110-gym==0.2.1) has invalid metadata: Expected matching RIGHT_PARENTHESIS for LEFT_PARENTHESIS, after version specifier
    opencv-python (>=3.) ; extra == 'all'
                  ~~~~^
Please use pip<24.1 if you need to use this version.
INFO: pip is looking at multiple versions of f110-gym to determine which version is compatible with other requirements. This could take a while.
ERROR: Could not find a version that satisfies the requirement gym==0.19.0 (from f110-gym) (from versions: 0.0.2, 0.0.3, 0.0.4, 0.0.5, 0.0.6, 0.0.7, 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.1.6, 0.1.7, 0.2.0, 0.2.1, 0.2.2, 0.2.3, 0.2.4, 0.2.5, 0.2.6, 0.2.7, 0.2.8, 0.2.9, 0.2.10, 0.2.11, 0.2.12, 0.3.0, 0.4.0, 0.4.1, 0.4.2, 0.4.3, 0.4.4, 0.4.5, 0.4.6, 0.4.8, 0.4.9, 0.4.10, 0.5.0, 0.5.1, 0.5.2, 0.5.3, 0.5.4, 0.5.5, 0.5.6, 0.5.7, 0.6.0, 0.7.0, 0.7.1, 0.7.2, 0.7.3, 0.7.4, 0.8.0.dev0, 0.8.0, 0.8.1, 0.8.2, 0.9.0, 0.9.1, 0.9.2, 0.9.3, 0.9.4, 0.9.5, 0.9.6, 0.9.7, 0.10.0, 0.10.1, 0.10.2, 0.10.3, 0.10.4, 0.10.5, 0.10.8, 0.10.9, 0.10.11, 0.11.0, 0.12.0, 0.12.1, 0.12.4, 0.12.5, 0.12.6, 0.13.0, 0.13.1, 0.14.0, 0.15.3, 0.15.4, 0.15.6, 0.15.7, 0.16.0, 0.17.0, 0.17.1, 0.17.2, 0.17.3, 0.18.0, 0.18.3, 0.19.0, 0.20.0, 0.21.0, 0.22.0, 0.23.0, 0.23.1, 0.24.0, 0.24.1, 0.25.0, 0.25.1, 0.25.2, 0.26.0, 0.26.1, 0.26.2)
ERROR: No matching distribution found for gym==0.19.0

As the warning suggests, this is due to incompatibility between pip versions >=24.1 and gym version 0.19.0. You can resolve this by downgrading the pip used in the virtualenv:

```
pip install --upgrade pip==24.0
```

The installation steps are then as follows:

```bash
virtualenv gym_env
source gym_env/bin/activate
pip install --upgrade pip==24.0
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```
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
