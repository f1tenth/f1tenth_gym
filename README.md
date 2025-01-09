![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
We recommend installing the simulation inside a venv.  
You can install the environment any way you like. Instructions to do it with virtualenv are below.
Make sure to use python version 3.10. We've tested 3.10.11 and 3.10.15 and both have worked.

```bash
virtualenv gym_env
source gym_env/bin/activate
```

Then clone the repo
```bash
git clone https://github.com/WE-Autopilot/f1tenth_gym.git
cd f1tenth_gym
```

Set some versions by hand to avoid magic, tracebackless errors.
```bash
pip install "pip<24.1"
pip install "setuptools==65.5.0"
pip install "wheel<0.40.0"
```

Then run the gym setup
```bash
pip install -e .
```

You can run a quick waypoint follow example by:
```bash
cd examples
python waypoint_follow.py
```

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required and we haven't tested it at all not even once.):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Known issues
You might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
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
