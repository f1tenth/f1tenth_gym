# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement. Updated by UBM Driverless for Python 3.12 and replaced `gym` with
`gymnasium`.

You can find the [old documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Installation
To install the environment use `pip install -e .` in the root directory of the repository.
It is advised to use a virtual environment for the installation if you just want to test the package.

## Usage example

```python
import numpy as np
import gymnasium as gym


params = {'mu': 1.0489,'C_Sf': 4.718,'C_Sr': 5.4562,'lf': 0.15875,'lr': 0.17145,'h': 0.074,'m': 3.74,'I': 0.04712,'s_min': -0.46,'s_max': 0.46,'sv_min': -3.2,'sv_max': 3.2,'v_switch': 7.319,'a_max': 9.51,'v_min':-5.0,'v_max': 20.0,'width': 0.31,'length': 0.58}
num_agents = 1


single_agent_low = np.array(
    [-np.inf, -np.inf, params['s_min'], params['v_min'], 0.0, params['sv_min'], -np.inf], dtype=np.float32)
single_agent_high = np.array(
    [+np.inf, +np.inf, params['s_max'], params['v_max'], 2 * np.pi, params['sv_max'], +np.inf], dtype=np.float32)

# Duplicate for all agents to match the shape (num_agents, 7)
obs_low = np.tile(single_agent_low, (num_agents, 1))
obs_high = np.tile(single_agent_high, (num_agents, 1))

# Now create the observation space with the proper dimensions
observation_space = gym.spaces.Box(
    low=obs_low,
    high=obs_high,
    shape=(num_agents, 7),
    dtype=np.float32
)

single_agent_low = np.array([-np.inf, params['s_min']], dtype=np.float32)
single_agent_high = np.array([+np.inf, params['s_max']], dtype=np.float32)

action_low = np.tile(single_agent_low, (num_agents, 1))
action_high = np.tile(single_agent_high, (num_agents, 1))

action_space = gym.spaces.Box(
    low=action_low,
    high=action_high,
    shape=(num_agents, 2),
    dtype=np.float32
)


def test_f110_env_rendering():
    # Initialize the environment
    map_filepath = ...  # Path to the map file
    raceline_filepath = ...  # Path to the raceline file
    
    env = gym.make('f110_gym:f110-v0',
                   map=map_filepath,
                   map_ext='.pgm',
                   num_agents=num_agents,
                   timestep=0.01,               # Simulation timestep
                   render_mode='human',         # None, 'human', 'human_fast'. If None, no rendering is done
                   points_in_foreground=False,  # Whether to render trail points of the car path in the foreground
                   raceline_path=raceline_filepath)

    # Set initial poses for agents
    # Format: [x, y, theta] for each agent
    # init_poses = np.array([
    #     [4.0, 5.0, 0.0],  # Agent 0 (ego)
    #     [0.0, 0.0, 0.0]  # Agent 1
    # ])
    init_poses = observation_space.sample()
    init_poses = init_poses[:, [0, 1, 4]]

    # Reset the environment with initial poses
    observation, info = env.reset(options={"poses": init_poses})
    print('obs reset:\n', observation)
    done = False

    # Run simulation for 1000 steps or until done
    for i in range(1000):
        if done:
            print(f"Episode finished after {i} steps")
            break

        # Generate random actions (steering angle, acceleration)
        # Actions format: [[steer_0, accel_0], [steer_1, accel_1], ...]
        # actions = np.array([
        #     [0, 0],
        #     [0, 0]
        # ])
        actions = action_space.sample()

        # Take action and get new observation
        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Print some information
        if i % 100 == 0:
            print(f"Step {i}")
            print(f"Ego position: ({observation[0,0]:.2f}, {observation[0,1]:.2f})")

        # Render the environment in human mode
        env.render()

if __name__ == "__main__":
    test_f110_env_rendering()

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
