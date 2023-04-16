from utils import create_env
import threading
import keyboard
import time
import numpy as np

# instantiating the environment
maps = list(range(1,150))
env = create_env(maps=maps)

obs = env.reset()

done = False

# Initialize action array
action = np.array([[0.0, 0.0]])

# Initialize control variables
steering_angle = 0.0
velocity = 0.0
delta = 0.05

def update_action():
    global action, steering_angle, velocity, delta
    while True:
        # Increase steering angle (right)
        if keyboard.is_pressed('a'):
            steering_angle += delta
            steering_angle = min(1.0, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Decrease steering angle (left)
        if keyboard.is_pressed('d'):
            steering_angle -= delta
            steering_angle = max(-1.0, steering_angle)
            action[0, 0] = steering_angle
            print("Action: ", action)

        # Increase velocity
        if keyboard.is_pressed('w'):
            velocity += delta
            velocity = min(1.0, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        # Decrease velocity
        if keyboard.is_pressed('s'):
            velocity -= delta
            velocity = max(-1.0, velocity)
            action[0, 1] = velocity
            print("Action: ", action)

        time.sleep(0.1)

keyboard_thread = threading.Thread(target=update_action)
keyboard_thread.start()

while not done:
    obs, reward, done, info = env.step(action)
    env.render()
