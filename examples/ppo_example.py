import gymnasium as gym
from stable_baselines3 import PPO

# if using wandb (recommended):
from wandb.integration.sb3 import WandbCallback
import wandb

# toggle this to train or evaluate
train = False

if train:
    run = wandb.init(
        project="f1tenth_gym_ppo",
        sync_tensorboard=True,
        save_code=True,
    )

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "rl"},
            "reset_config": {"type": "rl_random_static"},
        },
    )

    # will be faster on cpu
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", device="cpu", seed=42
    )
    model.learn(
        total_timesteps=1_000_000,
        callback=WandbCallback(
            gradient_save_freq=0, model_save_path=f"models/{run.id}", verbose=2
        ),
    )
    run.finish()

else:
    model_path = "models/3wlusg06/model.zip"
    model = PPO.load(model_path, print_system_info=True, device="cpu")
    eval_env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "rl"},
            "reset_config": {"type": "rl_random_static"},
        },
        render_mode="human",
    )
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        eval_env.render()

        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()
    eval_env.close()
