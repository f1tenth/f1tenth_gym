Laser Scan Simulator Models
======================================

This file contains all numba just-in-time compiled function for the 2D laser scan models. The core of the laser scan simulation is the Euclidean distance transform of the map image provided. See more details about the algorithm here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt. Then, ray tracing is used to create the beams of the 2D lasers scan.

.. currentmodule:: f110_gym.envs.laser_models

.. autosummary::
    :toctree: _generated/

    ScanSimulator2D
    get_dt
    xy_2_rc
    distance_transform
    trace_ray
    get_scan
    check_ttc_jit
    cross
    are_collinear
    get_range
    get_blocked_view_indices
    ray_cast
    