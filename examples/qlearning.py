import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
import pickle
import os
import logging


"""
Planner Helpers
"""

class QLearningPlanner:
    def __init__(self, state_space, action_space, alpha=0.25, gamma=0.95, epsilon=0.1, load_file=None):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_map = [
            (0.9, -45 * np.pi / 180),   # Speed 0.9, Steer -45 degrees
            (0.9, 0),                   # Speed 0.9, Steer straight
            (0.9, 45 * np.pi / 180),    # Speed 0.9, Steer 45 degrees
            (1.5, -45 * np.pi / 180),   # Speed 1.5, Steer -45 degrees
            (1.5, 0),                   # Speed 1.5, Steer straight
            (1.5, 45 * np.pi / 180),    # Speed 1.5, Steer 45 degrees
            (2.1, -30 * np.pi / 180),   # Speed 2.1, Steer -30 degrees
            (2.1, 0),                   # Speed 2.1, Steer straight
            (2.1, 30 * np.pi / 180),    # Speed 2.1, Steer 30 degrees
            (3.0, -10 * np.pi / 180),   # Speed 3.0, Steer -10 degrees
            (3.0, 0),                   # Speed 3.0, Steer straight
            (3.0, 10 * np.pi / 180),    # Speed 3.0, Steer 10 degrees
        ]
        if load_file:
            self.load_q_table(load_file)
        else:
            self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def save_q_table(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, filename):
        with open(filename, 'rb') as file:
            self.q_table = pickle.load(file)

    def plan(self, scans, linear_vels_x, poses_x, poses_y):
        state = self.compute_state(scans, linear_vels_x, poses_x, poses_y)
        action_index = self.choose_action(state)
        action = self.action_map[action_index]
        print(f"action = {action}")
        return action

    def compute_state(self, scans, linear_vels_x, poses_x, poses_y):
        # Select 12 points from the scan array
        selected_points = np.linspace(0, len(scans)-1, num=7, dtype=int)
        selected_scans = scans[selected_points]

        # convert selected_scans to a low - <0.4, high - >2.0, mid 0.4-2.0
        for i in range(len(selected_scans)):
            if selected_scans[i] < 0.5:
                selected_scans[i] = 0
            elif selected_scans[i] > 2.0:
                selected_scans[i] = 2
            else:
                selected_scans[i] = 1
                
        linear_vels_x = linear_vels_x // 0.2

        # Combine the normalized scans and poses_theta into a single state array
        state = np.concatenate((selected_scans, [linear_vels_x, poses_x, poses_y]))

        # Hash the state array to get the state index
        state_index = hash(tuple(state)) % self.state_space

        return state_index

    def render_waypoints(self, *args, **kwargs):
        pass


class MyRewardWrapper(gym.RewardWrapper):
    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     reward = self.reward(obs)
    #     return obs, reward, done, info
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info
    
    def reward(self, obs):
        reward = -0.001
        # return reward
        scans = obs['scans'][0]
        poses_x = obs['poses_x'][0]
        poses_y = obs['poses_y'][0]
        poses_theta = obs['poses_theta']
        linear_vels_x = obs['linear_vels_x'][0]
        linear_vels_y = obs['linear_vels_y'][0]
        ang_vels_z = obs['ang_vels_z']
        collisions = obs['collisions']
        
        if collisions:
            reward -= 1000
        velocity = np.sqrt(linear_vels_x**2 + linear_vels_y**2)
        reward += velocity * 2.0

        if min(scans) < 0.5:      
            reward -= 1 /(np.min(scans)-0.1) * 1
        
        return float(reward) 


def main(i):
    """
    main entry point
    """
    iteration_count = i
    state_space_size = 2000
    action_space_size = 12
    starting_epsilon = 0.8
    epsilon_decay = 0.01
    epsilon = starting_epsilon / np.exp(epsilon_decay * iteration_count)
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    logging.info(f"Starting iteration {iteration_count} with epsilon = {epsilon}")
    
    if os.path.exists('final_q_table.pkl'): 
        logging.info(f"Loading final_q_table.pkl")
        planner = QLearningPlanner(state_space=state_space_size, 
                                   action_space=action_space_size, 
                                   epsilon=epsilon,
                                   load_file="final_q_table.pkl")
    else:
        logging.info(f"Creating new QLearningPlanner")
        planner = QLearningPlanner(state_space=state_space_size, 
                                   action_space=action_space_size,
                                   epsilon=epsilon)
        


    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, 
                   num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env = MyRewardWrapper(env)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    scans = obs['scans'][0]
    poses_x = obs['poses_x'][0]
    poses_y = obs['poses_y'][0]
    poses_theta = obs['poses_theta']
    linear_vels_x = obs['linear_vels_x'][0]
    env.render()

    laptime = 0.0
    start = time.time()
    
    save_interval = 100
    step_count = 0 
    
    tt_reward = 0

    while not done:
        current_state = planner.compute_state(scans, linear_vels_x, poses_x, poses_y)
        action_output = planner.plan(scans, linear_vels_x, poses_x, poses_y)
        action_index = planner.action_map.index(action_output)
        
        # should be (steer, speed)
        obs, step_reward, done, info = env.step(np.array([[action_output[1], action_output[0]]]))  # Note: the order might need to be adjusted based on your env
        scans = obs['scans'][0]
        poses_x = obs['poses_x'][0]
        poses_y = obs['poses_y'][0]
        poses_theta = obs['poses_theta']
        linear_vels_x = obs['linear_vels_x'][0]
        
        tt_reward += step_reward
        
        ### check step function       
        # print(f"obs = {obs}, step_reward = {step_reward}")
        print(f"step_reward = {step_reward}")
        # print(f"scans = {obs['scans']}, len scans = {len(obs['scans'][0])}")

        # obatin the next_state
        next_state = planner.compute_state(scans, linear_vels_x, poses_x, poses_y)
        
        planner.update_q_table(current_state, action_index, step_reward, next_state)
        
        env.render(mode='human')
        
        if iteration_count % save_interval == 0:
            planner.save_q_table(f'q_table_iter{iteration_count}.pkl')  # Save the Q-table

    selected_points = np.linspace(0, len(scans)-1, num=11, dtype=int)
    selected_scans = scans[selected_points]
    logging.info(f"selected_scans = {selected_scans}")
    logging.info(f"min selected_scans = {np.min(selected_scans)}")
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    logging.info(f'Sim elapsed time: {laptime}, Real elapsed time: {time.time()-start}')
    print(f"Total reward = {tt_reward}")
    logging.info(f"Total reward = {tt_reward}")
    
    planner.save_q_table('final_q_table.pkl')

if __name__ == '__main__':
    # initiate logging file
    logging.basicConfig(filename=f'waypoint_follow_{time.ctime()}.log', level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f'Logging initiated {time.ctime()}')
    
    iterations = 1000
    for i in range(iterations):
        print(f"iteration = {i}")
        main(i)
