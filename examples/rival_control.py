"""Shared, pure longitudinal command step for opt-in corner tracking."""
import math

from numba import njit


@njit(cache=True, fastmath=False)
def advance_speed_command(speed, acceleration, desired_speed, dt,
                          max_acceleration, max_braking, max_jerk):
    """Slew acceleration, reserving speed/headroom to release it at bounded jerk.

    Inputs use positive dt/limits and a nonnegative, reachable command state.
    A newly lowered goal need not be reachable: it changes the desired
    acceleration, never clips the existing speed or bypasses the jerk limit.
    """
    desired = min(max((desired_speed - speed) / dt, -max_braking), max_acceleration)
    release = max_jerk * dt
    if desired > 0.:
        # For the NEXT acceleration a, reserve a*dt + a**2/(2*J).
        # The full discrete step makes this conservative for sampled release,
        # including when the remaining positive acceleration is below J*dt.
        headroom = max(0., desired_speed - speed)
        positive_limit = max(0., math.sqrt(release ** 2 + 2 * max_jerk * headroom) - release)
        desired = min(desired, positive_limit)
    acceleration += min(max(desired - acceleration, -release), release)
    # Keep the existing braking-release budget, independent of a mutable goal.
    braking_limit = max(0., math.sqrt(release ** 2 + 2 * max_jerk * speed) - release)
    acceleration = max(acceleration, -braking_limit)
    return max(0., speed + acceleration * dt), acceleration
