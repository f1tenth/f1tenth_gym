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

    def __init__(self, x, y,
        psis: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        vxs: Optional[np.ndarray] = None,
        axs: Optional[np.ndarray] = None,
        ss: Optional[np.ndarray] = None,
    ):
        self.xs = x
        self.ys = y
        input_vals = [x, y, psis, ks, vxs, axs, ss]

        # Only close the path if for the input values from the user,
        # the first and last points are not the same => the path is not closed
        # Otherwise, the constructed values can mess up the s calculation and closure
        need_closure = False
        for input_val in input_vals:
            if input_val is not None:
                if not (input_val[-1] == input_val[0]):
                    need_closure = True
                    break

        def close_with_constructor(input_val, constructor, closed_path):
            '''
            If the input value is not None, return it.
            Otherwise, return the constructor, with closure if necessary.

            Parameters
            ----------
            input_val : np.ndarray | None
                The input value from the user.
            constructor : np.ndarray
                The constructor to use if the input value is None.
            closed_path : bool
                Indicator whether the orirignal path is closed.
            '''
            if input_val is not None:
                return input_val 
            else:
                temp_ret = constructor
                if closed_path:
                   temp_ret[-1] = temp_ret[0]
                return temp_ret
            
        self.psis = close_with_constructor(psis, self._calc_yaw_from_xy(x, y), not need_closure)
        self.ks = close_with_constructor(ks, self._calc_kappa_from_xy(x, y), not need_closure)
        self.vxs = close_with_constructor(vxs, np.ones_like(x), not need_closure)
        self.axs = close_with_constructor(axs, np.zeros_like(x), not need_closure)
        self.ss = close_with_constructor(ss, self.__calc_s(x, y), not need_closure)
        psis_spline = close_with_constructor(psis, self._calc_yaw_from_xy(x, y), not need_closure)

        # If yaw is provided, interpolate cosines and sines of yaw for continuity
        cosines_spline = np.cos(psis_spline)
        sines_spline = np.sin(psis_spline)
        
        ks_spline = close_with_constructor(ks, self._calc_kappa_from_xy(x, y), not need_closure)
        vxs_spline = close_with_constructor(vxs, np.zeros_like(x), not need_closure)
        axs_spline = close_with_constructor(axs, np.zeros_like(x), not need_closure)

        self.points = np.c_[self.xs, self.ys, 
                            cosines_spline, sines_spline, 
                            ks_spline, vxs_spline, axs_spline]
        
        if need_closure:
            self.points = np.vstack(
                (self.points, self.points[0])
            )  # Ensure the path is closed

        if ss is not None:
            self.s = ss
        else:
            self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        self.s_interval = (self.s[-1] - self.s[0]) / len(self.s)

        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necesaxsry to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")
        self.spline_x = np.array(self.spline.x) 
        self.spline_c = np.array(self.spline.c)


    def find_segment_for_s(self, x):
        # Find the segment of the spline that x is in
        return (x / (self.spline.x[-1] + self.s_interval) * (len(self.spline_x) - 1)).astype(int)
    
    def predict_with_spline(self, point, segment, state_index=0):
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        # exp_x = (point - self.spline.x[[segment]])[None, :] ** np.arange(4)[::-1, None]
        exp_x = ((point - self.spline.x[segment % len(self.spline.x)]) ** np.arange(4)[::-1])[:, None]
        vec = self.spline.c[:, segment % self.spline.c.shape[1], state_index]
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
        s = [0]
        s.extend(np.cumsum(self.ds))
        return np.array(s)
    
    def _calc_yaw_from_xy(self, x, y):
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        heading = np.arctan2(dy_dt, dx_dt)
        return heading

    def _calc_kappa_from_xy(self, x, y):
        dx_dt = np.gradient(x, 2)
        dy_dt = np.gradient(y, 2)
        d2x_dt2 = np.gradient(dx_dt, 2)
        d2y_dt2 = np.gradient(dy_dt, 2)
        curvature = -(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        return curvature

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
        segment = self.find_segment_for_s(s)
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
        segment = self.find_segment_for_s(s)
        cos = self.predict_with_spline(s, segment, 2)[0]
        sin = self.predict_with_spline(s, segment, 3)[0]
        # yaw = (math.atan2(sin, cos) + 2 * math.pi) % (2 * math.pi)
        yaw = np.arctan2(sin, cos)
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
            x_eval, y_eval = self.spline(s)[0, :2]
            return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)

        output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
        closest_s = float(output[0][0])
        absolute_distance = output[1]
        return closest_s, absolute_distance

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
            s_inds = np.arange(self.points.shape[0])
        _, ey, t, min_dist_segment = nearest_point_on_trajectory(
            np.array([x, y]).astype(np.float32), self.points[s_inds, :2]
        )
        min_dist_segment_s_ind = s_inds[min_dist_segment]
        s = float(
            self.s[min_dist_segment_s_ind]
            + t * (self.s[min_dist_segment_s_ind + 1] - self.s[min_dist_segment_s_ind])
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
