"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import math

import numpy as np
import scipy.optimize as so
from scipy import interpolate
from typing import Union, Optional

from f1tenth_gym.envs.track.utils import nearest_point_on_trajectory


class CubicSpline2D:
    """
    Cubic CubicSpline2D class.

    Attributes
    ----------
    s : list
        cumulative distance along the data points.
    sx : CubicSpline1D
        cubic spline for x coordinates.
    sy : CubicSpline1D
        cubic spline for y coordinates.
    """

    def __init__(self, x, y):
        self.points = np.c_[x, y]
        if not np.all(self.points[-1] == self.points[0]):
            self.points = np.vstack(
                (self.points, self.points[0])
            )  # Ensure the path is closed
        self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necessary to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")

    def __calc_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calc cumulative distance.

        Parameters
        ----------
        x : list
            x coordinates for data points.
        y : list
            y coordinates for data points.

        Returns
        -------
        s : np.ndarray
            cumulative distance along the data points.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return np.array(s)

    def calc_position(self, s: float) -> np.ndarray:
        """
        Calc position at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float | None
            x position for given s.
        y : float | None
            y position for given s.
        """
        return self.spline(s)

    def calc_curvature(self, s: float) -> Optional[float]:
        """
        Calc curvature at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx, dy = self.spline(s, 1)
        ddx, ddy = self.spline(s, 2)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
        return k

    def calc_yaw(self, s: float) -> Optional[float]:
        """
        Calc yaw angle at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. If `s` is outside the data point's range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx, dy = self.spline(s, 1)
        yaw = math.atan2(dy, dx)
        # Convert yaw to [0, 2pi]
        yaw = yaw % (2 * math.pi)

        return yaw

    def calc_arclength(
        self, x: float, y: float, s_guess: float = 0.0
    ) -> tuple[float, float]:
        """
        Calculate arclength for a given point (x, y) on the trajectory.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.
        s_guess : float
            initial guess for s.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """

        def distance_to_spline(s):
            x_eval, y_eval = self.spline(s)[0]
            return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)

        output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
        closest_s = float(output[0][0])
        absolute_distance = output[1]
        return closest_s, absolute_distance

    def calc_arclength_inaccurate(self, x: float, y: float) -> tuple[float, float]:
        """
        Fast calculation of arclength for a given point (x, y) on the trajectory.
        Less accuarate and less smooth than calc_arclength but much faster.
        Suitable for lap counting.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """
        _, ey, t, min_dist_segment = nearest_point_on_trajectory(
            np.array([x, y]), self.points
        )
        # s = s at closest_point + t
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )

        return s, 0.0

    def _calc_tangent(self, s: float) -> np.ndarray:
        """
        Calculates the tangent to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        tangent : float
            tangent vector for given s.
        """
        dx, dy = self.spline(s, 1)
        tangent = np.array([dx, dy])
        return tangent

    def _calc_normal(self, s: float) -> np.ndarray:
        """
        Calculate the normal to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        normal : float
            normal vector for given s.
        """
        dx, dy = self.spline(s, 1)
        normal = np.array([-dy, dx])
        return normal
