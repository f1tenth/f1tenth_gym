import gym
from stable_baselines3 import PPO
import warnings
from f110_gym.envs.base_classes import Integrator
from utils import *
import logging
import time
import threading
import keyboard

# Suppress the 'Box bound precision lowered by casting to float32' warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# instantiating the environment

env = create_env()

obs = env.reset()

done = False
# loops when env not done

# Initialize action array
action = np.array([[0.0, 0.0]])

# Initialize control variables
steering_angle = 0.0
velocity = 0.0
delta = 0.2


def update_action():
    global action, steering_angle, velocity, delta
    while True:
        # Increase steering angle (right)
        if keyboard.is_pressed('d'):
            steering_angle += delta
            steering_angle = min(0.4189, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Decrease steering angle (left)
        if keyboard.is_pressed('a'):
            steering_angle -= delta
            steering_angle = max(-0.4189, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Increase velocity
        if keyboard.is_pressed('w'):
            velocity += delta
            velocity = min(3.2, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        # Decrease velocity
        if keyboard.is_pressed('s'):
            velocity -= delta
            velocity = max(-3.2, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        time.sleep(0.1)

keyboard_thread = threading.Thread(target=update_action)
keyboard_thread.start()

while not done:
    
    
    
    
    # action = np.array([[1.0, 0.05]])
    # action = np.array([[0.0, 0.00]])
    # action, _state = model.predict(obs, deterministic=True)
    # action, _state = model.predict(obs)
    # stepping through the environment
    obs, reward, done, info = env.step(action)
    
    # print('vel s', obs['linear_vels_s'])
    # print('vel d', obs['linear_vels_d'])
    # print('angvel', obs['ang_vels_z'])
    # print('pose s', obs['poses_s'])
    # print('pose d', obs['poses_d'])
    # print('pose theta', obs['poses_theta'])
    # print('reward', reward)
    # print(min(obs['scans']))
    # print(max(obs['scans']))
    # print(obs['poses_theta'])
    # print()
    env.render()
    # time.sleep(0.5)
