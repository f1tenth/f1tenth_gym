# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{o2020textscf1tenth,
  title={textscF1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```

F1TENTH also provide a guide on building the physical 1/10th scale vehicle:[https://f1tenth.org/build.html](https://f1tenth.org/build.html)

## Installation (With Docker)
We recommend using the provided Dockerfile to create containers for this simulation environment.

To build and run the docker containers, note that you might need to run these with ```sudo``` depending on how you setup docker:
```bash
$ cd f1tenth_gym
$ docker build -t f1tenth_gym -f Dockerfile .
$ docker run -it --name=f1tenth_gym_container --rm f1tenth_gym
```

## Installation (Native)
The environment officially supports Python3, and you'll need several dependencies to run this environment:

### Python packages:

```bash
$ pip3 install --user numpy scipy numba Pillow gym pyyaml pyglet
```

### To install the simulation environment natively, clone this repo.

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym
```

### Then install the env via the following steps:
```bash
$ cd f1tenth_gym
$ pip3 install --user -e gym/
```

## Example Usage
### Basic:
```python
import gym
import numpy as np

# making the environment
racecar_env = gym.make('f110_gym:f110-v0')
obs, step_reward, done, info = racecar_env.reset(np.array([[0., 0., 0.], # pose of ego
                                                           [2., 0., 0.]])) # pose of 2nd agent

# simulation loop
lap_time = 0.
while not done:
    # some agent policy that you created
    actions = planner.plan(obs) # numpy.ndarray (num_agents, 2), columns are steering angle and then velocity

    # stepping through the environment
    obs, step_reward, done, info = racecar_env.step(actions)

    lap_time += step_reward
```

### TODO: More customization:
```python
# loading the map (uses the ROS convention with .yaml and an image file)
map_path = 'your/path/to/the/map/file.yaml'
map_img_ext = '.png' # png extension for example

# loading physical parameters of the car
# These could be identified on your own system
mass= 3.74
l_r = 0.17145
I_z = 0.04712
mu = 0.523
h_cg = 0.074
cs_f = 4.718
cs_r = 5.4562

racecar_env.init_map(map_path, map_img_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir, double_finish=True)

# Initial state (for two cars)
initial_x = [0.0, 2.0]
initial_y = [0.0, 0.0]
initial_theta = [0.0, 0.0]
lap_time = 0.0

# Resetting the environment
obs, step_reward, done, info = racecar_env.reset({'x': initial_x,
                                                  'y': initial_y,
                                                  'theta': initial_theta})
# Simulation loop
while not done:

    # Your agent here
    ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)

    # Stepping through the environment
    action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
    obs, step_reward, done, info = racecar_env.step(action)

    # Getting the lap time
    lap_time += step_reward
```
