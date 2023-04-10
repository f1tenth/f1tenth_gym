import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

model = PPO("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/PPO/", verbose=1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


model.learn(50000, callback=TensorboardCallback())