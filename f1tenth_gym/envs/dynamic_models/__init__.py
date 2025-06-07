"""
This module contains the dynamic models available in the F1Tenth Gym.
Each submodule contains a single model, and the equations or their source is documented alongside it. Many of the models are from the CommonRoad repository, available here: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
"""

import warnings
from enum import Enum
import numpy as np

from .kinematic import vehicle_dynamics_ks, get_standardized_state_ks
from .single_track import vehicle_dynamics_st, get_standardized_state_st
from .multi_body import init_mb, vehicle_dynamics_mb, get_standardized_state_mb
from .utils import pid_steer, pid_accl
from typing import Optional

class DynamicModel(Enum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3  # Multi-body Model

    @staticmethod
    def from_string(model: str):
        if model == "ks":
            warnings.warn(
                "Chosen model is KS. This is different from previous versions of the gym."
            )
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        elif model == "mb":
            return DynamicModel.MB
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None, params: Optional[dict] = None):
        # Assert that if self is MB, params is not None
        if self == DynamicModel.MB and params is None:
            raise ValueError("MultiBody model requires parameters to be provided.")
        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            self.state_dim = 5      
            self.control_dim = 2
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            self.state_dim = 7
            self.control_dim = 2
        elif self == DynamicModel.MB:
            # state is a 29D vector
            self.state_dim = 29
            self.control_dim = 2
        else:
            raise ValueError(f"Unknown model type {self}")
        state = np.zeros(self.state_dim)

        # set initial pose if provided
        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        # If state is MultiBody, we must inflate the state to 29D
        if self == DynamicModel.MB:
            state = init_mb(state, params)
        return state

    @property
    def f_dynamics(self):
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        elif self == DynamicModel.MB:
            return vehicle_dynamics_mb
        else:
            raise ValueError(f"Unknown model type {self}")

    def get_standardized_state_fn(self):
        """
        This function returns the standardized state information for the model.
        This needs to be a function, because the state information is different for each model.
        Slip is not directly available from the MB model.
        """
        if self == DynamicModel.KS:
            return get_standardized_state_ks
        elif self == DynamicModel.ST:
            return get_standardized_state_st
        elif self == DynamicModel.MB:
            return get_standardized_state_mb
        else:
            raise ValueError(f"Unknown model type {self}")
