from stable_baselines3 import PPO
from utils import *
from reward import *
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.logger import configure

save_interval = 50_000
save_path = "./models/ppo_model"
log_dir = "./metrics/"

env = create_env()
env = Monitor(env, log_dir)

# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)


model_index = int(4 * 100000)
model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_{}.zip".format(model_index), env=env)

combined_callback = TensorboardCallback(save_interval, save_path, verbose=1)



model.learn(total_timesteps=10000_000, callback=combined_callback, progress_bar=True)

env.env.close()
env.close()