# The F1TENTH Gym environment

Zirui branch TODO list:
- [ ] verify collision
- [ ] scan toggle
- [ ] separate time step and integrater time step
- [ ] add frenet
- [ ] use correct loop count
- [ ] simplify observation
- [ ] simplify reset

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