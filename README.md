# The F1TENTH Gym environment

Zirui branch TODO list:
- [x] clip lidar after adding noise, put lidar noise in config
- [x] remove unnecessary use of dictionary. model st is 50% faster, mb is 150% faster.
- [x] toggle renderer
- [ ] verify collision
- [ ] scan toggle
- [ ] separate time step and integrater time step
- [ ] add frenet
- [ ] use correct loop count
- [x] simplify std_state and observation
- [ ] simplify reset
- [ ] control_buffer_size in config
- [ ] use class attribute for config, not dict
- [ ] save dynamic param and default gym config in separate yaml file

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