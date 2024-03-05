from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple

import warnings
import gymnasium as gym
import numpy as np
from .dynamic_models import pid_steer, pid_accl


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
            action,
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
        return self._type
    
    @property
    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self.lower_limit, high=self.upper_limit, dtype=np.float32)

class SteeringAngleAction(SteerAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "steering_angle"
        self.lower_limit, self.upper_limit = params["s_min"], params["s_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> float: 
        sv = pid_steer(
            action,
            state[2],
            params["sv_max"],
        )
        return sv
    
class SteeringSpeedAction(SteerAction):
    def __init__(self, params: Dict) -> None:
        super().__init__()
        self._type = "steering_speed"
        self.lower_limit, self.upper_limit = params["sv_min"], params["sv_max"]

    def act(
        self, action: Tuple[float, float], state: np.ndarray, params: Dict
    ) -> float: 
        return action

class SteerActionEnum(Enum):
    Steering_Angle = 1
    Steering_Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "steering_angle":
            return SteeringAngleAction
        elif action == "steering_speed":
            return SteeringSpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")

class CarAction:
    def __init__(self, control_mode : list[str, str], params: Dict) -> None:
        long_act_type_fn = None
        steer_act_type_fn = None
        if type(control_mode) == str: # only one control mode specified
            try:
                long_act_type_fn = LongitudinalActionEnum.from_string(control_mode)
            except ValueError:
                try:
                    steer_act_type_fn = SteerActionEnum.from_string(control_mode)
                except ValueError:
                    raise ValueError(f"Unknown control mode {control_mode}")
                if control_mode == "steering_speed":
                    warnings.warn(
                        f'Only one control mode specified, using {control_mode} for steering and defaulting to acceleration for longitudinal control'
                    )
                    long_act_type_fn = LongitudinalActionEnum.from_string("accl")
                else:
                    warnings.warn(
                        f'Only one control mode specified, using {control_mode} for steering and defaulting to speed for longitudinal control'
                    )
                    long_act_type_fn = LongitudinalActionEnum.from_string("speed")

            else:
                if control_mode == "accl":
                    warnings.warn(
                        f'Only one control mode specified, using {control_mode} for longitudinal control and defaulting to steering speed for steering'
                    )
                    steer_act_type_fn = SteerActionEnum.from_string("steering_speed")
                else:
                    warnings.warn(
                        f'Only one control mode specified, using {control_mode} for longitudinal control and defaulting to steering angle for steering'
                    )
                    steer_act_type_fn = SteerActionEnum.from_string("steering_angle")

        elif type(control_mode) == list:
            long_act_type_fn = LongitudinalActionEnum.from_string(control_mode[0])
            steer_act_type_fn = SteerActionEnum.from_string(control_mode[1])
        else:
            raise ValueError(f"Unknown control mode {control_mode}")
        
        self._longitudinal_action : LongitudinalAction = long_act_type_fn(params)
        self._steer_action : SteerAction = steer_act_type_fn(params)

    @abstractmethod
    def act(self, action: Any, **kwargs) -> Tuple[float, float]:
        longitudinal_action = self._longitudinal_action.act(action[0], **kwargs)
        steer_action = self._steer_action.act(action[1], **kwargs)
        return longitudinal_action, steer_action

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
