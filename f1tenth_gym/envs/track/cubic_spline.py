"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""
import math

import numpy as np
import scipy.optimize as so
from scipy import interpolate
from numba import njit

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )

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
            self.points = np.vstack((self.points, self.points[0]))  # Ensure the path is closed
        self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necessary to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type='periodic')

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

    def calc_position(self, s: float) -> tuple[float | None, float | None]:
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

    def calc_curvature(self, s: float) -> float | None:
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

    def calc_yaw(self, s: float) -> float | None:
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
        return yaw

    def calc_arclength(self, x, y, s_guess=0.0):
        """
        calc arclength
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

        def distance_to_spline(s):
            x_eval, y_eval = self.spline(s)[0]
            return np.sqrt((x - x_eval)**2 + (y - y_eval)**2)
        
        output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
        closest_s = output[0][0]
        absolute_distance = output[1]
        return closest_s, absolute_distance
    
    def calc_arclength_inaccurate(self, x, y, s_guess=0.0):
        """
        calc arclength, use nearest_point_on_trajectory
        Less accuarate and less smooth than calc_arclength but
        much faster - suitable for lap counting
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
        _, ey, t, min_dist_segment = nearest_point_on_trajectory(np.array([x, y]), self.points)
        # s = s at closest_point + t
        s = self.s[min_dist_segment] + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])

        return s, 0

    def _calc_tangent(self, s):
        '''
        calculates the tangent to the curve at a given point
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        tangent : float
            tangent vector for given s.
        '''
        dx, dy = self.spline(s, 1)
        tangent = np.array([dx, dy])
        return tangent
    
    def _calc_normal(self, s):
        '''
        calculates the normal to the curve at a given point
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        normal : float
            normal vector for given s.
        '''
        dx, dy = self.spline(s, 1)
        normal = np.array([-dy, dx])
        return normal