"""
Code from Cubic spline planner
Author: Atsushi Sakai(@Atsushi_twi)
"""
from __future__ import annotations
import bisect
import math

import numpy as np


class CubicSpline1D:
    """
    1D Cubic Spline class

    Attributes
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted in ascending order.
    y : list
        y coordinates for data points
    a : list
        coefficient a
    b : list
        coefficient b
    c : list
        coefficient c
    d : list
        coefficient d
    nx : int
        dimension of x
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Returns a 1D cubic spline object.

        Parameters
        ----------
        x : list
            x coordinates for data points. This x coordinates must be
            sorted in ascending order.
        y : list
            y coordinates for data points

        Raises
        ------
        ValueError
            if x is not sorted in ascending order
        """
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # Calc coefficient a
        self.a = [iy for iy in y]

        # Calc coefficient c
        A = self.__calc_A(h=h)
        B = self.__calc_B(h=h, a=self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (
                2.0 * self.c[i] + self.c[i + 1]
            )
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x: float) -> float | None:
        """
        Calc `y` position for given `x`.
        If `x` is outside the data point's `x` range, return None.

        Parameters
        ----------
        x : float
            position for which to calculate y.

        Returns
        -------
        y : float
            position along the spline for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = (
            self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        )

        return position

    def calc_first_derivative(self, x: float) -> float | None:
        """
        Calc first derivative at given x.
        If x is outside the input x, return None

        Parameters
        ----------
        x : float
            position for which to calculate the first derivative.

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x: float) -> float | None:
        """
        Calc second derivative at given x.
        If x is outside the input x, return None

        Parameters
        ----------
        x : float
            position for which to calculate the second derivative.

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x: float) -> int:
        """
        Search data segment index.

        Parameters
        ----------
        x : float
            position for which to find the segment index.

        Returns
        -------
        index : int
            index of the segment.
        """
        return bisect.bisect(self.x[:-1], x) - 1

    def __calc_A(self, h: np.ndarray) -> np.ndarray:
        """
        Calc matrix A for spline coefficient c.

        Parameters
        ----------
        h : np.ndarray
            difference of x coordinates.

        Returns
        -------
        A : np.ndarray
            matrix A.
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Calc matrix B for spline coefficient c.

        Parameters
        ----------
        h : np.ndarray
            difference of x coordinates.
        a : np.ndarray
            y coordinates for data points.

        Returns
        -------
        B : np.ndarray
            matrix B.
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = (
                3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
            )
        return B


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

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Returns a 2D cubic spline object.

        Parameters
        ----------
        x : list
            x coordinates for data points.
        y : list
            y coordinates for data points.
        """
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(x=self.s, y=x)
        self.sy = CubicSpline1D(x=self.s, y=y)

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
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

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
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
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
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw
