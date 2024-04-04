"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import math

import numpy as np
import scipy.optimize as so
from scipy import interpolate
class CubicSpline2D:
    """
    Cubic CubicSpline2D class
    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.
    """

    def __init__(self, x, y):
        points = np.c_[x, y]
        if not np.all(points[-1] == points[0]):
            points = np.vstack((points, points[0]))  # Ensure the path is closed
        self.s = self.__calc_s(points[:, 0], points[:, 1])
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necessary to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, points, bc_type='periodic')

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        return self.spline(s)

    def calc_curvature(self, s):
        """
        calc curvature
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

    def calc_yaw(self, s):
        """
        calc yaw
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
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