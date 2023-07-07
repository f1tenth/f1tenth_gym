from abc import abstractmethod
from enum import Enum
from typing import Any, Tuple, Dict

import numpy as np
from f110_gym.envs.dynamic_models import pid


class CarActionEnum(Enum):
    Accl = 1
    Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "accl":
            return AcclAction()
        elif action == "speed":
            return SpeedAction()
        else:
            raise ValueError(f"Unknown action type {action}")

class CarAction:
    def __init__(self) -> None:
        self._action_type = None

    @abstractmethod
    def act(self, action: Any, **kwargs) -> Tuple[float, float]:
        raise NotImplementedError("act method not implemented")

    @property
    def type(self) -> str:
        return self._action_type


class AcclAction(CarAction):
    def __init__(self) -> None:
        super().__init__()
        self._action_type = "accl"

    def act(
        self, action: Tuple[float, float], state, params
    ) -> Tuple[float, float]:
        return action


class SpeedAction(CarAction):
    def __init__(self) -> None:
        super().__init__()
        self._action_type = "speed"

    def act(self, action: Tuple[float, float], state: np.ndarray, params: Dict) -> Tuple[float, float]:
        accl, sv = pid(
            action[0],
            action[1],
            state[3],
            state[2],
            params["sv_max"],
            params["a_max"],
            params["v_max"],
            params["v_min"],
        )
        return accl, sv
