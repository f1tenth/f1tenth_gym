from stable_baselines3 import PPO
from utils import *
from reward import *
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.logger import configure

from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat


save_interval = 50_000
save_path = "./models/ppo_model"
log_dir = "./metrics/"

env = create_env()
# env = Monitor(env, log_dir)

# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)


model_index = int(4 * 100000)
model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_{}.zip".format(model_index), env=env, tensorboard_log=log_dir)


# tb_output = None
# for output in model.logger.output_formats:
#     if isinstance(output, TensorBoardOutputFormat):
#         tb_output = output
#         break

combined_callback = TensorboardCallback(save_interval, save_path, log_dir=log_dir, verbose=1)


# combined_callback = TensorboardCallback(
#     save_interval, save_path, writer=tb_output.writer, log_dir=log_dir, verbose=1
# )



# model.learn(total_timesteps=10000_000, callback=combined_callback, progress_bar=True)
model.learn(total_timesteps=10000_000, callback=combined_callback)

env.env.close()
env.close()