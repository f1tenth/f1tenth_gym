import gym
import numpy as np
import yaml
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator


with open('config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)






# instantiating the environment
racecar_env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    
obs, step_reward, done, info = racecar_env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
racecar_env.render()


# Ottieni lo spazio delle azioni


# Stampa le informazioni sullo spazio delle azioni




class Planner:
    def __init__(self):
        # You can initialize any parameters or variables here
        pass

    def plan(self, observation):
        # This method should take the current observation as input
        # and return the corresponding action or plan
        # Replace this with your actual planning logic

        #wasd to 

        print(observation)

        action = np.array([[3., 0.9]])


        
        return action

# Example usage
planner = Planner()

# simulation loop
lap_time = 0.

# loops when env not done
while not done:
    # get action based on the observation
    actions = planner.plan(obs)

    # stepping through the environment
    obs, step_reward, done, info = racecar_env.step(actions)

    racecar_env.render(mode='human')

    lap_time += step_reward