import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
import pickle
import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import re



"""
Planner Helpers
"""

class DQNAgent:
    def __init__(self, state_space, action_space, alpha=0.001, gamma=0.95, epsilon=0.1, memory_size=10000):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.model = self.build_model()
        
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

    def build_model(self):
        optimizer = Adam(learning_rate=self.alpha)
        model = Sequential([
            Dense(24, input_dim=self.state_space, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_space, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizer)
        return model
    
    def choose_action_index(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = state.reshape(1, -1).astype('float32')  # Reshape for single prediction

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def compute_state(self, scans, linear_vels_x, ang_vels_z, collision):
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
        state = np.concatenate((selected_scans, [linear_vels_x, ang_vels_z, collision]))

        return state

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
        # If not enough samples, skip this replay cycle
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            next_state = next_state.reshape(1, -1).astype('float32')
            state = state.reshape(1, -1).astype('float32')
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        
    def render_waypoints(self, *args, **kwargs):
        pass






class MyRewardWrapper(gym.RewardWrapper):

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation), done, info
    
    def reward(self, obs):
        reward = -1

        scans = obs['scans'][0]
        linear_vels_x = obs['linear_vels_x'][0]
        ang_vels_z = obs['ang_vels_z'][0]
        collisions = obs['collisions']
        
        if collisions:
            reward -= 1000
        velocity = linear_vels_x
        reward += velocity * 3.0
        reward -= ang_vels_z * 1.0

        if min(scans) < 0.5:      
            reward -= 1 /(np.min(scans)-0.1) * 1
        
        return float(reward) 


def main(i):
    """
    main entry point
    """
    iteration_count = i
    state_space_size = 10
    action_space_size = 12
    epsilon = 0.8
    epsilon_decay = 0.01
    epsilon_min = 0.01
    epsilon = max(epsilon_min, epsilon * np.exp(-epsilon_decay * iteration_count))

    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    logging.info(f"Starting iteration {iteration_count} with epsilon = {epsilon}")
    
    if any(re.match(r'q_table_iter.*\.h5', file) for file in os.listdir()):
        latest_q_table = max([file for file in os.listdir() if re.match(r'q_table_iter.*\.h5', file)], key=lambda x: int(re.search(r'\d+', x).group()))

        logging.info(f"Loading {latest_q_table} as the latest Q-table")
        agent = DQNAgent(state_space=state_space_size, 
                         action_space=action_space_size, 
                         epsilon=epsilon)
        agent.load(latest_q_table)

    else:
        logging.info(f"Creating new DQNAgent")
        agent = DQNAgent(state_space=state_space_size, 
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

        agent.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, 
                   num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env = MyRewardWrapper(env)
    # env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    scans = obs['scans'][0]
    linear_vels_x = obs['linear_vels_x'][0]
    collision = obs['collisions']
    ang_vels_z = obs['ang_vels_z'][0]
    # env.render()

    laptime = 0.0
    start = time.time()
    
    save_interval = 100
    step_count = 0 
    
    tt_reward = 0

    while not done:
        current_state = agent.compute_state(scans, linear_vels_x, ang_vels_z, collision)
        action_index = agent.choose_action_index(current_state)
        action_output = agent.action_map[action_index]
        # print(f"action_output = {action_output}")
        
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
        # print(f"step_reward = {step_reward}")
        # print(f"scans = {obs['scans']}, len scans = {len(obs['scans'][0])}")

        # obatin the next_state
        next_state = agent.compute_state(scans, linear_vels_x, ang_vels_z, collision)
        
        agent.remember(current_state, action_index, step_reward, next_state, done)
        agent.replay(batch_size=256)
        
        # env.render(mode='human')
        
        if iteration_count % save_interval == 0:
            agent.save(f'q_table_iter{iteration_count}.h5')  # Save the Q-table

    selected_points = np.linspace(0, len(scans)-1, num=11, dtype=int)
    selected_scans = scans[selected_points]
    logging.info(f"selected_scans = {selected_scans}")
    logging.info(f"min selected_scans = {np.min(selected_scans)}")
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
    logging.info(f'Sim elapsed time: {laptime}, Real elapsed time: {time.time()-start}')
    print(f"Total reward = {tt_reward}")
    logging.info(f"Total reward = {tt_reward}")
    
    agent.save('final_q_table.h5')

if __name__ == '__main__':
    # initiate logging file
    logging.basicConfig(filename=f'dqn_{time.ctime()}.log', level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f'Logging initiated {time.ctime()}')
    
    iterations = 1000
    for i in range(iterations):
        print(f"iteration = {i}")
        main(i)
        
    
    logging.info(f'Logging finished {time.ctime()}')