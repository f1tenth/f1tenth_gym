"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import math

import numpy as np
from scipy import interpolate
from typing import Union, Optional
# import scipy.optimize as so
from f1tenth_gym.envs.track.utils import nearest_point_on_trajectory
from numba import njit

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

    def __init__(self, x, y,
        psis: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        vxs: Optional[np.ndarray] = None,
        axs: Optional[np.ndarray] = None,
        ss: Optional[np.ndarray] = None,
    ):        
        psis = psis if psis is not None else self._calc_yaw_from_xy(x, y)
        ks = ks if ks is not None else self._calc_kappa_from_xy(x, y)
        vxs = vxs if vxs is not None else np.ones_like(x)
        axs = axs if axs is not None else np.zeros_like(x)

        self.points = np.c_[x, y, 
                            np.cos(psis), np.sin(psis), 
                            ks, vxs, axs]
        
        if np.any(self.points[-1, :2] != self.points[0, :2]):
            self.points = np.vstack((self.points, self.points[0]))
        else:
            self.points[-1] = self.points[0]
        self.s = ss if ss is not None else self.__calc_s(self.points[:, 0], self.points[:, 1])
        self.ss, self.psis, self.ks = self.s, psis, ks
        self.s_interval = (self.s[-1] - self.s[0]) / len(self.s)
        self.s_frame_max = self.s[-1]

        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necessary to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")
        self.spline_x = np.array(self.spline.x) 
        self.spline_c = np.array(self.spline.c) 
        self.diffs = self.points[1:, :2] - self.points[:, :2][:-1, :2]
        self.l2s = self.diffs[:, 0] ** 2 + self.diffs[:, 1] ** 2

    def find_segment_for_s(self, x):
        # Find the segment of the spline that x is in
        # print(x, self.spline.x[-1], self.s_interval, len(self.spline_x))
        return (x / (self.spline.x[-1] + self.s_interval) * (len(self.spline_x) - 1)).astype(int)
        # return (x / (self.spline.x[-1]) * (len(self.spline_x) - 2)).astype(int)
    
    def predict_with_spline(self, point, segment, state_index=0):
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        # exp_x = (point - self.spline.x[[segment]])[None, :] ** np.arange(4)[::-1, None]
        exp_x = ((point - self.spline.x[segment % len(self.spline.x)]) ** np.arange(4)[::-1])[:, None]
        vec = self.spline_c[:, segment % self.spline_c.shape[1], state_index]
        # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
        point = vec.dot(exp_x)
        return np.asarray(point)

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
        return np.concatenate([np.array([0]), np.cumsum(self.ds)])
    
    def _calc_yaw_from_xy(self, x, y):
        dx_dt = np.gradient(x, edge_order=2)
        dy_dt = np.gradient(y, edge_order=2)
        heading = (np.arctan2(dy_dt, dx_dt) + 2 * math.pi) % (2 * math.pi)
        return heading

    def _calc_kappa_from_xy(self, x, y):
        # For more stable gradients, extend x and y by two (edge_order 2) elements on each side
        # The elements are taken from the other side of the array
        x_extended = np.concatenate((x[-2:], x, x[:2]))
        y_extended = np.concatenate((y[-2:], y, y[:2]))
        dx_dt = np.gradient(x_extended, edge_order=2)
        dy_dt = np.gradient(y_extended, edge_order=2)
        d2x_dt2 = np.gradient(dx_dt, edge_order=2)
        d2y_dt2 = np.gradient(dy_dt, edge_order=2)
        curvature = (dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature[2:-2]

    def calc_position(self, s: float, segment=None) -> np.ndarray:
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
        segment = segment or self.find_segment_for_s(s)
        x = self.predict_with_spline(s, segment, 0)[0]
        y = self.predict_with_spline(s, segment, 1)[0]
        return x,y

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
        segment = self.find_segment_for_s(s)
        k = self.predict_with_spline(s, segment, 4)[0]
        return k

    def find_curvature(self, s: float) -> Optional[float]:
        """
        Find curvature at the given s by the segment.

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
        segment = self.find_segment_for_s(s)
        k = self.points[segment, 4]
        return k
        
    def calc_yaw(self, s: float, segment=None) -> Optional[float]:
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
        segment = segment or self.find_segment_for_s(s)
        cos = self.predict_with_spline(s, segment, 2)[0]
        sin = self.predict_with_spline(s, segment, 3)[0]
        yaw = (math.atan2(sin, cos) + 2 * math.pi) % (2 * math.pi)
        return yaw

    # def calc_arclength(
    #     self, x: float, y: float, s_guess: float = 0.0
    # ) -> tuple[float, float]:
    #     """
    #     Calculate arclength for a given point (x, y) on the trajectory.

    #     Parameters
    #     ----------
    #     x : float
    #         x position.
    #     y : float
    #         y position.
    #     s_guess : float
    #         initial guess for s.
    #     Returns
    #     -------
    #     s : float
    #         distance from the start point for given x, y.
    #     ey : float
    #         lateral deviation for given x, y.
    #     """
    #     def distance_to_spline(s):
    #         x_eval, y_eval = self.spline(s)[0, :2]
    #         return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)
    #     output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
    #     closest_s = float(output[0][0])
    #     absolute_distance = output[1]
    #     return closest_s, absolute_distance

    def calc_arclength_inaccurate(self, x: float, y: float, s_inds=None) -> tuple[float, float]:
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
        if s_inds is None:
            ey, t, min_dist_segment = nearest_point_on_trajectory(
                np.asarray([x, y]).astype(np.float32), self.points[:, :2]
            )
        else:
            ey, t, min_dist_segment = nearest_point_on_trajectory(
                np.asarray([x, y]).astype(np.float32), self.points[s_inds, :2]
            )
            min_dist_segment = s_inds[min_dist_segment]
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )
        return s, ey

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
        dx, dy = self.spline(s, 1)[:2]
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
        dx, dy = self.spline(s, 1)[:2]
        normal = np.array([-dy, dx])
        return normal