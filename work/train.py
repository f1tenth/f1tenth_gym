from stable_baselines3 import PPO
from utils import *
from reward import *

log_dir = "./metrics/"

env = create_env()
# model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4, max_grad_norm=0.5, clip_range=0.2, clip_range_vf=1.0, tensorboard_log="./metrics/")

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cuda')

save_interval = 50_000
save_path = "./models/ppo_model"

callback = SaveModelCallback(save_interval, save_path)

while True:
    model.learn(total_timesteps=10000_000, callback=callback, progress_bar=True)

env.env.close()
env.close()