from abc import abstractmethod

import numpy as np


class ResetFn:
    @abstractmethod
    def sample(self) -> np.ndarray:
        pass
