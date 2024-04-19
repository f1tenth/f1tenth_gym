import gymnasium as gym

gym.register(
    id="f1tenth-v0",
    entry_point="f1tenth_gym.envs:F110Env",
)
