from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig

from . import Raceline
from .cubic_spline import CubicSpline2D
from .utils import find_track_dir


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: str
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class Track:
    spec: TrackSpec
    filepath: str
    ext: str
    occupancy_map: np.ndarray
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        filepath: str,
        ext: str,
        occupancy_map: np.ndarray,
        centerline: Optional[Raceline] = None,
        raceline: Optional[Raceline] = None,
    ):
        """
        Initialize track object.

        Parameters
        ----------
        spec : TrackSpec
            track specification
        filepath : str
            path to the track image
        ext : str
            file extension of the track image file
        occupancy_map : np.ndarray
            occupancy grid map
        centerline : Raceline, optional
            centerline of the track, by default None
        raceline : Raceline, optional
            raceline of the track, by default None
        """
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.centerline = centerline
        self.raceline = raceline

    @staticmethod
    def load_spec(track: str, filespec: str) -> TrackSpec:
        """
        Load track specification from yaml file.

        Parameters
        ----------
        track : str
            name of the track
        filespec : str
            path to the yaml file

        Returns
        -------
        TrackSpec
            track specification
        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            track_spec = TrackSpec(name=track, **map_metadata)
        return track_spec

    @staticmethod
    def from_track_name(track: str):
        """
        Load track from track name.

        Parameters
        ----------
        track : str
            name of the track

        Returns
        -------
        Track
            track object

        Raises
        ------
        FileNotFoundError
            if the track cannot be loaded
        """
        try:
            track_dir = find_track_dir(track)
            track_spec = Track.load_spec(
                track=track, filespec=str(track_dir / f"{track_dir.stem}_map.yaml")
            )

            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            image = Image.open(track_dir / str(map_filename)).transpose(
                Transpose.FLIP_TOP_BOTTOM
            )
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # if exists, load centerline
            if (track_dir / f"{track}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(
                    track_dir / f"{track}_centerline.csv"
                )
            else:
                centerline = None

            # if exists, load raceline
            if (track_dir / f"{track}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(
                    track_dir / f"{track}_raceline.csv"
                )
            else:
                raceline = centerline

            return Track(
                spec=track_spec,
                filepath=str((track_dir / map_filename.stem).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )
        except Exception as ex:
            print(ex)
            raise FileNotFoundError(f"It could not load track {track}") from ex

    @staticmethod
    def from_refline(x: np.ndarray, y: np.ndarray, velx: np.ndarray,):
        """
        Create an empty track reference line.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the waypoints
        y : np.ndarray
            y-coordinates of the waypoints
        velx : np.ndarray
            velocities at the waypoints

        Returns
        -------
        Track
            track object
        """
        ds = 0.1
        resolution = 0.05
        margin_perc = 0.1

        spline = CubicSpline2D(x=x, y=y)
        ss, xs, ys, yaws, ks, vxs = [], [], [], [], [], []
        for i_s in np.arange(0, spline.s[-1], ds):
            xi, yi = spline.calc_position(i_s)
            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            # find closest waypoint
            closest = np.argmin(np.hypot(x - xi, y - yi))
            v = velx[closest]

            xs.append(xi)
            ys.append(yi)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)
            vxs.append(v)

        refline = Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.array(vxs).astype(np.float32),
            accxs=np.zeros_like(ss).astype(np.float32),
            spline=spline,
        )

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        x_range = max_x - min_x
        y_range = max_y - min_y
        occupancy_map = 255.0 * np.ones(
            (
                int((1 + 2 * margin_perc) * x_range / resolution),
                int((1 + 2 * margin_perc) * y_range / resolution),
            ),
            dtype=np.float32,
        )
        # origin is the bottom left corner
        origin = (min_x - margin_perc * x_range, min_y - margin_perc * y_range, 0.0)

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=resolution,
            origin=origin,
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )

        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=refline,
            centerline=refline,
        )
