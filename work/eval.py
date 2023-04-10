from stable_baselines3 import PPO
from utils import *
from reward import *

env = create_env()

model_index = int(4 * 100000)
model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_{}.zip".format(model_index), env=env)
# model = PPO.load("/Users/meraj/workspace/f1tenth_gym/work/models/base/01_100423.zip", env=env)

obs = env.reset()
done = False

while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode='human_fast')
