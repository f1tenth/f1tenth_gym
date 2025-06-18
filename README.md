# The F1TENTH Gym environment

Zirui branch TODO list:
- [x] clip lidar after adding noise, put lidar noise in config
- [x] remove unnecessary use of dictionary. model st is 50% faster, mb is 150% faster.
- [x] toggle renderer
- [ ] verify collision
- [x] scan toggle
- [x] separate time step and integrater time step
- [x] add frenet
- [x] use correct loop count frenet_based, added max loop num
- [ ] also add winding_angle
- [x] simplify std_state and observation
- [x] add option "state" in reset
- [x] control_buffer_size, lidar fov, lidar nums in config
- [ ] double check dynamics result
- [x] implemented new rendering with pyqtgraph.opengl
- [x] added 3d mesh renderering for proof of concept
- [ ] merge lidar scan fix

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