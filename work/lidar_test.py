import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from utils import create_env

maps = list(range(2, 3))
# maps=list(range(1,2))

env = create_env(maps=maps)
env.training = False
# exit()

# model = "models/ppo_model_950000"
model = "models/ppo_model/ent_1_200000"

model = PPO.load(path=model, env=env)

obs = env.reset()
done = False

# Set up the LiDAR data plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Get the LiDAR data from obs
    lidar_data = obs['scans']

    # Convert LiDAR data to polar coordinates
    num_angles = lidar_data.size
    full_lidar_angle = 270  # degrees
    angles = np.linspace(np.radians(full_lidar_angle / 2), -np.radians(full_lidar_angle / 2), num_angles)

    # Update the LiDAR data plot
    ax.clear()
    ax.plot(angles, lidar_data.flatten(), marker='o', markersize=2, linestyle='None')
    ax.set_title("Real-time LiDAR data")
    ax.set_ylim(0, np.max(lidar_data))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.draw()
    plt.pause(0.001)

    env.render(mode='human_fast')

# Close the LiDAR data plot
plt.ioff()
plt.close(fig)
