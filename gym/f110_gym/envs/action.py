from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from f110_gym.envs.dynamic_models import pid_steer, pid_accl


class LongitudinalActionEnum(Enum):
    Accl = 1
    Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "accl":
            return AcclAction
        elif action == "speed":
            return SpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")
        
class LongitudinalAction:
    def __init__(self) -> None:
        self._type = None

        self.lower_limit = None
        self.upper_limit = None

    @abstractmethod
    def act(self, longitudinal_action: Any, **kwargs) -> float:
        raise NotImplementedError("longitudinal act method not implemented")

    @property
    def type(self) -> str:
        return self._type

    @property
    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self.lower_limit, high=self.upper_limit, dtype=np.float32)

class AcclAction(LongitudinalAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "accl"
        self.lower_limit, self.upper_limit = -params["a_max"], params["a_max"]

    def act(self, action: Tuple[float, float], state, params) -> float:
        return action

class SpeedAction(LongitudinalAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "speed"
        self.lower_limit, self.upper_limit = params["v_min"], params["v_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> float:
        accl = pid_accl(
            action[0],
            state[3],
            params["a_max"],
            params["v_max"],
            params["v_min"],
        )

        return accl

class SteerAction:
    def __init__(self) -> None:
        self._type = None

        self.lower_limit = None
        self.upper_limit = None

    @abstractmethod
    def act(self, steer_action: Any, **kwargs) -> float:
        raise NotImplementedError("steer act method not implemented")

    @property
    def type(self) -> str:
        return self._steer_action_type
    
    @property
    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self.lower_limit, high=self.upper_limit, dtype=np.float32)

class SteeringSpeedAction(SteerAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "speed"
        self.lower_limit, self.upper_limit = params["sv_min"], params["sv_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> float: # pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v)
        sv = pid_steer(
            action[1],
            state[2],
            params["sv_max"],
        )
        return sv
    
class SteeringAngleAction(SteerAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "speed"
        self.lower_limit, self.upper_limit = params["s_min"], params["s_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> float: # pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v)
        return action[1]

class SteerActionEnum(Enum):
    Angle = 1
    Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "angle":
            return SteeringAngleAction
        elif action == "speed":
            return SteeringSpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")
class CarAction:
    def __init__(self, control_mode : list[str, str]) -> None:
        self._longitudinal_action : LongitudinalAction = LongitudinalActionEnum.from_string(control_mode[0])
        self._steer_action : SteerAction = SteerActionEnum.from_string(control_mode[1])

    @abstractmethod
    def act(self, action: Any, **kwargs) -> Tuple[float, float]:
        longitudinal_action = self._longitudinal_action.act(action[0], **kwargs)
        steer_action = self._steer_action.act(action[1], **kwargs)
        return steer_action, longitudinal_action

    @property
    def type(self) -> Tuple[str, str]:
        return (self._steer_action.type, self._longitudinal_action.type)

    @property
    def space(self) -> gym.Space:
        low = np.array([self._steer_action.lower_limit, self._longitudinal_action.lower_limit]).astype(np.float32)
        high = np.array([self._steer_action.upper_limit, self._longitudinal_action.upper_limit]).astype(np.float32)

        return gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)


def from_single_to_multi_action_space(
    single_agent_action_space: gym.spaces.Box, num_agents: int
) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=single_agent_action_space.low[None].repeat(num_agents, 0),
        high=single_agent_action_space.high[None].repeat(num_agents, 0),
        shape=(num_agents, single_agent_action_space.shape[0]),
        dtype=np.float32,
    )
