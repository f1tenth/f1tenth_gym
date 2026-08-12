![Python 3.9+](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
![Code Style](https://github.com/f1tenth/f1tenth_gym/actions/workflows/lint.yml/badge.svg)

# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart

### Using uv (recommended)

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:

```bash
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
uv sync
```

Then you can run a quick waypoint follow example by:
```bash
uv run python examples/waypoint_follow.py
```

### Using pip

Alternatively, you can install using pip in a virtual environment:

```bash
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

Then run examples with:
```bash
python examples/waypoint_follow.py
```

### Using Docker

A Dockerfile is also provided. The image renders headless by default (Qt offscreen), no GPU required:
```bash
docker build -t f1tenth_gym .
docker run -it f1tenth_gym
# inside the container:
python3 examples/waypoint_follow.py
```

To render to a display instead, forward your X server and override the Qt platform:
```bash
docker run -it -e QT_QPA_PLATFORM=xcb -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym
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
