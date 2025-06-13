from __future__ import annotations
import time
import uuid
import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig

from . import Raceline
from .cubic_spline import CubicSplineND
from .utils import find_track_dir


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: Optional[str]
    image: Optional[str]
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class Track:
    spec: TrackSpec
    filepath: Optional[str]
    ext: Optional[str]
    occupancy_map: np.ndarray
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        occupancy_map: np.ndarray,
        filepath: Optional[str] = None,
        ext: Optional[str] = None,
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
    def from_track_name(track: str, track_scale: float = 1.0) -> Track:
        """
        Load track from track name.

        Parameters
        ----------
        track : str
            name of the track
        track_scale : float, optional
            scale of the track, by default 1.0

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
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
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
                    track_dir / f"{track}_centerline.csv",
                    track_scale=track_scale,
                )
            else:
                centerline = None

            # if exists, load raceline
            if (track_dir / f"{track}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(
                    track_dir / f"{track}_raceline.csv",
                    track_scale=track_scale,
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
    def from_track_path(path: pathlib.Path, track_scale: float = 1.0) -> Track:
        """
        Load track from track path.

        Parameters
        ----------
        path : pathlib.Path
            path to the track yaml file

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
            if type(path) is str:
                path = pathlib.Path(path)
        
            track_spec = Track.load_spec(
                track=path.stem, filespec=path
            )
            track_spec.resolution = track_spec.resolution * track_scale
            track_spec.origin = (
                track_spec.origin[0] * track_scale,
                track_spec.origin[1] * track_scale,
                track_spec.origin[2],
            )

            # load occupancy grid
            # Image path is from path + image name from track_spec
            image_path = path.parent / track_spec.image  
            image = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # if exists, load centerline
            if (path / f"{path.stem}_centerline.csv").exists():
                centerline = Raceline.from_centerline_file(path / f"{path.stem}_centerline.csv")
            else:
                centerline = None

            # if exists, load raceline
            if (path / f"{path.stem}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(path / f"{path.stem}_raceline.csv")
            else:
                raceline = centerline

            return Track(
                spec=track_spec,
                filepath=str(path.absolute()),
                ext=image_path.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )
        except Exception as ex:
            print(ex)
            raise FileNotFoundError(f"It could not load track {path}") from ex

    @staticmethod
    def from_refline(x: np.ndarray, y: np.ndarray, velx: np.ndarray):
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
        track: Track
            track object
        """
        ds = 0.1
        resolution = 0.05
        margin_perc = 0.1

        spline = CubicSplineND(x=x, y=y)
        ss, xs, ys, yaws, ks, vxs = spline.ss, spline.xs, spline.ys, spline.psis, spline.ks, velx

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
    
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", skip_rows: int = 3, track_scale: float = 1.0) -> Track:
        """
        Creates a Track object from a raceline file of the format [s, x, y, psi, k, vx, ax].
        
        Args:
            filepath (pathlib.Path): path to the raceline file
            delimiter (str, optional): delimiter used in the file. Defaults to ";".
            skip_rows (int, optional): number of rows to skip. Defaults to 3.
            track_scale (float, optional): scale of the track. Defaults to 1.0.
        
        Returns:
            Track: track object
        """
        raceline = Raceline.from_raceline_file(filepath, delimiter, skip_rows, track_scale)
        xs = raceline.xs
        ys = raceline.ys
        resolution = 0.05
        margin_perc = 0.1

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

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=0.05,
            origin=(0.0, 0.0, 0.0),
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )
        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=raceline,
            centerline=raceline,
        )

    def save_raceline(self, outdir: pathlib.Path):
        """
        Save track raceline.

        Parameters
        ----------
        outdir : pathlib.Path
            output directory
        """
        raceline_filepath = outdir / f"{self.spec.name}_raceline.csv"
        with open(raceline_filepath, "w") as raceline_csv:
            raceline_csv.write("# " + str(uuid.uuid4()) + "\n") # same as TUM opt
            raceline_csv.write('# {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'))) # TUM opt uses ggv hash, but no ggv here
            raceline_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
            for i in range(len(self.raceline.ss)):
                raceline_csv.write(
                    f"{self.raceline.ss[i]}; {self.raceline.xs[i]}; {self.raceline.ys[i]}; {self.raceline.yaws[i]}; {self.raceline.ks[i]}; {self.raceline.vxs[i]}; {self.raceline.axs[i]}\n"
                )

    def save_centerline(self, outdir: pathlib.Path, half_width: float):
        """
        Save track raceline.

        Parameters
        ----------
        outdir : pathlib.Path
            output directory
        half_width : float
            half width of the track
        """
        raceline_filepath = outdir / f"{self.spec.name}_raceline.csv"
        with open(raceline_filepath, "w") as raceline_csv:
            raceline_csv.write("# " + str(uuid.uuid4()) + "\n") # same as TUM opt
            raceline_csv.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
            for i in range(len(self.centerline.ss)):
                raceline_csv.write(
                    f"{self.centerline.xs[i]}, {self.centerline.ys[i]}, {half_width}, {half_width}\n"
                )

    def frenet_to_cartesian(self, s, ey, ephi, use_raceline=False):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        line = self.raceline if use_raceline else self.centerline
        x, y = line.spline.calc_position(s)
        psi = line.spline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, psi

    def cartesian_to_frenet(self, x, y, phi, use_raceline=False, s_guess=0):
        """
        Convert Cartesian coordinates to Frenet coordinates.

        x: x-coordinate
        y: y-coordinate
        phi: yaw angle

        returns:
            s: distance along the centerline
            ey: lateral deviation
            ephi: heading deviation
        """
        line = self.raceline if use_raceline else self.centerline
        # s, ey = line.spline.calc_arclength_inaccurate(x, y) # inaccurate, but much faster
        s, ey = line.spline.calc_arclength(x, y, s_guess)
        # Wrap around
        s = s % line.spline.s[-1]

        # Use the normal to calculate the signed lateral deviation
        yaw = line.spline.calc_yaw(s)
        normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
        x_eval, y_eval = line.spline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        phi = phi - yaw
        return s, ey, np.arctan2(np.sin(phi), np.cos(phi))
