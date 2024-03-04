Collision Models
========================================

This file contains all the numba just-in-time compiled functions for collision checking between agents. The GJK algorithm (more detail here: https://cse442-17f.github.io/Gilbert-Johnson-Keerthi-Distance-Algorithm/) is used to check for overlap in polygons.

.. currentmodule:: f110_gym.envs.collision_models

.. autosummary::
    :toctree: _generated/

    perpendicular
    tripleProduct
    avgPoint
    indexOfFurthestPoint
    support
    collision
    collision_multiple
    get_trmtx
    get_vertices