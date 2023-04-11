from stable_baselines3 import PPO
from utils import *
from reward import *

save_interval = 50_000
save_path = "./models/ppo_model"
log_dir = "./metrics/"

env = create_env()

model_index = 'ppo_model_{}'.format(int(4 * 100000))
model_index = 'base/01_100423'

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, device='cuda')

# model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/{}.zip".format(model_index), env=env, tensorboard_log=log_dir)

combined_callback = TensorboardCallback(save_interval, save_path, log_dir=log_dir, verbose=1)

model.learn(total_timesteps=10000_000, callback=combined_callback)

env.env.close()
env.close()