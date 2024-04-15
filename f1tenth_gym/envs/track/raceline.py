from __future__ import annotations
import pathlib
from typing import Optional

import numpy as np

from ..rendering import EnvRenderer
from .cubic_spline import CubicSpline2D


class Raceline:
    """
    Raceline object.

    Attributes
    ----------
    n : int
        number of waypoints
    ss : np.ndarray
        arclength along the raceline
    xs : np.ndarray
        x-coordinates of the waypoints
    ys : np.ndarray
        y-coordinates of the waypoints
    yaws : np.ndarray
        yaw angles of the waypoints
    ks : np.ndarray
        curvature of the waypoints
    vxs : np.ndarray
        velocity along the raceline
    axs : np.ndarray
        acceleration along the raceline
    length : float
        length of the raceline
    spline : CubicSpline2D
        spline object through the waypoints
    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
        spline: Optional[CubicSpline2D] = None,
    ):
        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs

        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))

        # compute spline through waypoints if not provided
        self.spline = spline or CubicSpline2D(x=xs, y=ys)

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path,
        delimiter: Optional[str] = ",",
        fixed_speed: Optional[float] = 1.0,
    ):
        """
        Load raceline from a centerline file.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the centerline file
        delimiter : str, optional
            delimiter used in the file, by default ","
        fixed_speed : float, optional
            fixed speed along the raceline, by default 1.0

        Returns
        -------
        Raceline
            raceline object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert waypoints.shape[1] == 4, "expected waypoints as [x, y, w_left, w_right]"

        # fit cubic spline to waypoints
        xx, yy = waypoints[:, 0], waypoints[:, 1]
        # close loop
        xx = np.append(xx, xx[0])
        yy = np.append(yy, yy[0])
        spline = CubicSpline2D(x=xx, y=yy)
        ds = 0.1

        ss, xs, ys, yaws, ks = [], [], [], [], []

        for i_s in np.arange(0, spline.s[-1], ds):
            x, y = spline.calc_position(i_s)
            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)

        return Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.ones_like(ss).astype(np.float32) * fixed_speed,  # constant speed
            accxs=np.zeros_like(ss).astype(np.float32),  # constant acceleration
            spline=spline,
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";"):
        """
        Load raceline from a raceline file.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"

        Returns
        -------
        Raceline
            raceline object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter).astype(np.float32)
        assert (
            waypoints.shape[1] == 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        return Raceline(
            ss=waypoints[:, 0],
            xs=waypoints[:, 1],
            ys=waypoints[:, 2],
            psis=waypoints[:, 3],
            kappas=waypoints[:, 4],
            velxs=waypoints[:, 5],
            accxs=waypoints[:, 6],
        )

    def render_waypoints(self, e: EnvRenderer) -> None:
        """
        Callback to render waypoints.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object.
        """
        points = np.stack([self.xs, self.ys], axis=1)
        e.render_closed_lines(points, color=(0, 128, 0), size=1)
