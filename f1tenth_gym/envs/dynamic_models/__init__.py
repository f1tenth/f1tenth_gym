"""
This module contains the dynamic models available in the F1Tenth Gym.
Each submodule contains a single model, and the equations or their source is documented alongside it. Many of the models are from the CommonRoad repository, available here: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
"""
import warnings
from enum import Enum
import numpy as np

from .kinematic import vehicle_dynamics_ks
from .single_track import vehicle_dynamics_st
from .utils import pid_steer, pid_accl

class DynamicModel(Enum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3

    @staticmethod
    def from_string(model: str):
        if model == "ks":
            warnings.warn(
                "Chosen model is KS. This is different from previous versions of the gym."
            )
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None):
        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            state = np.zeros(5)
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            state = np.zeros(7)
        else:
            raise ValueError(f"Unknown model type {self}")

        # set initial pose if provided
        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        return state

    @property
    def f_dynamics(self):
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        else:
            raise ValueError(f"Unknown model type {self}")
