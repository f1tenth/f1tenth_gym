from stable_baselines3 import PPO
from utils import  create_env, TensorboardCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__=='__main__':
        
    save_interval = 50_000
    save_path = "./models/ppo_model"
    log_dir = "./metrics/"
    maps = list(range(2,60))
    
    num_cpu = 4  # Number of processes to use
    vec_env = SubprocVecEnv([(lambda i: lambda: create_env(maps=maps))(i)
                             for i in range(num_cpu)])

    # model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    model_path = "/Users/meraj/workspace/f1tenth_gym/work/models/ppo_model_400000.zip"
    model = PPO.load(model_path, env=vec_env, tensorboard_log=log_dir)

    combined_callback = TensorboardCallback(save_interval, save_path, verbose=1)
    model.learn(total_timesteps=10000_000, callback=combined_callback)

    vec_env.close()