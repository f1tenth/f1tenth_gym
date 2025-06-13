from __future__ import annotations
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from ..rendering import EnvRenderer
from .cubic_spline import CubicSplineND


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
    spline : CubicSplineND
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
        spline: Optional[CubicSplineND] = None,
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
        self.spline = spline or CubicSplineND(x=xs, y=ys,
                                            psis=psis,
                                            ks=kappas,
                                            vxs=velxs,
                                            axs=accxs,
                                            ss=ss)

        self.waypoint_render = None

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path,
        delimiter: Optional[str] = ",",
        fixed_speed: Optional[float] = 1.0,
        track_scale: Optional[float] = 1.0,
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
        track_scale : float, optional
            scaling factor for the track, by default 1.0

        Returns
        -------
        Raceline
            raceline object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        df = pd.read_csv(filepath, delimiter=delimiter, header=0).astype(np.float32)

        # Clean column names: remove '#' and strip whitespace
        df.columns = df.columns.str.replace('#', '').str.strip()

        # Required columns
        required_cols = ["x_m", "y_m"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in raceline file: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # fit cubic spline to waypoints
        xx, yy = df["x_m"].values, df["y_m"].values
        # scale waypoints
        xx, yy = xx * track_scale, yy * track_scale
        
        # Velocity is constant along the raceline
        vx = np.ones_like(xx) * fixed_speed
        # close loop
        spline = CubicSplineND(x=xx, y=yy, vxs=vx)

        return Raceline(
            ss=spline.ss,
            xs=spline.xs,
            ys=spline.ys,
            psis=spline.psis,
            kappas=spline.ks,
            velxs=spline.vxs,
            accxs=spline.axs,
            spline=spline,
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", skip_rows: int = 3, track_scale: Optional[float] = 1.0) -> Raceline:
        """
        Load raceline from a raceline file of the format [s, x, y, psi, k, vx, ax].

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        track_scale : float, optional
            scaling factor for the track, by default 1.0

        Returns
        -------
        Raceline
            raceline object
        """
        if type(filepath) is str:
            filepath = pathlib.Path(filepath)

        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_rows - 1, header=0).astype(np.float32)
        # Clean column names: remove '#' and strip whitespace
        df.columns = df.columns.str.replace('#', '').str.strip()

        # Required columns
        required_cols = ["x_m", "y_m", "psi_rad", "vx_mps"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in raceline file: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        if track_scale != 1.0:
            # scale x-y waypoints and recalculate s, psi, and k
            df["x_m"] *= track_scale
            df["y_m"] *= track_scale
            spline = CubicSplineND(x=df["x_m"].values, y=df["y_m"].values)    
            ss, yaws, ks = spline.ss, spline.psis, spline.ks
            df["psi_rad"] = yaws
            if "kappa_radpm" in df.columns:
                df["kappa_radpm"] = ks
            if "s_m" in df.columns:
                df["s_m"] = ss
        
        return Raceline(
            ss=df["s_m"].values,
            xs=df["x_m"].values,
            ys=df["y_m"].values,
            psis=df["psi_rad"].values if "psi_rad" in df.columns else None,
            kappas=df["kappa_radpm"].values if "kappa_radpm" in df.columns else None,
            velxs=df["vx_mps"].values if "vx_mps" in df.columns else None,
            accxs=df["ax_mps2"].values if "ax_mps2" in df.columns else None,
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
        if self.waypoint_render is None:
            self.waypoint_render = e.render_closed_lines(points, color=(0, 128, 0), size=1)
        else:
            self.waypoint_render.updateItems(points)
