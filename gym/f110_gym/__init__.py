import gymnasium as gym

gym.register(
    id="f110-v0",
    entry_point="f110_gym.envs:F110Env",
)
