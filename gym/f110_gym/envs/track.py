import pathlib
import tarfile
from dataclasses import dataclass
from typing import Tuple
import tempfile

import numpy as np
import requests
import yaml
from f110_gym.envs.cubic_spline import CubicSpline2D
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig


class Raceline:
    n: int

    ss: np.ndarray  # cumulative distance along the raceline
    xs: np.ndarray  # x-coordinates of the raceline
    ys: np.ndarray  # y-coordinates of the raceline
    yaws: np.ndarray  # yaw angle of the raceline
    ks: np.ndarray  # curvature of the raceline
    vxs: np.ndarray  # velocity along the raceline
    axs: np.ndarray  # acceleration along the raceline

    length: float

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        ss: np.ndarray = None,
        psis: np.ndarray = None,
        kappas: np.ndarray = None,
        accxs: np.ndarray = None,
        spline: CubicSpline2D = None,
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

        # compute spline
        self.spline = spline if spline is not None else CubicSpline2D(xs, ys)

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path, delimiter: str = ",", fixed_speed: float = 1.0
    ):
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert waypoints.shape[1] == 4, "expected waypoints as [x, y, w_left, w_right]"

        # fit cubic spline to waypoints
        xx, yy = waypoints[:, 0], waypoints[:, 1]
        # close loop
        xx = np.append(xx, xx[0])
        yy = np.append(yy, yy[0])
        spline = CubicSpline2D(xx, yy)
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
            velxs=np.ones_like(ss).astype(
                np.float32
            ),  # centerline does not have a speed profile, keep it constant at 1.0 m/s
            accxs=np.zeros_like(ss).astype(np.float32),  # constant acceleration
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";"):
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


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: str
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


def find_track_dir(track_name):
    # we assume there are no blank space in the track name. however, to take into account eventual blank spaces in
    # the map dirpath, we loop over all possible maps and check if there is a matching with the current track
    map_dir = pathlib.Path(__file__).parent.parent.parent.parent / "maps"

    if not (map_dir / track_name).exists():
        print("Downloading Files for: " + track_name)
        tracks_url = "http://api.f1tenth.org/" + track_name + ".tar.xz"
        tracks_r = requests.get(url=tracks_url, allow_redirects=True)
        if tracks_r.status_code == 404:
            raise FileNotFoundError(f"No maps exists for {track_name}.")

        tempdir = tempfile.gettempdir() + "/"

        with open(tempdir + track_name + ".tar.xz", "wb") as f:
            f.write(tracks_r.content)

        # extract
        print("Extracting Files for: " + track_name)
        tracks_file = tarfile.open(tempdir + track_name + ".tar.xz")
        tracks_file.extractall(map_dir)
        tracks_file.close()

    for base_dir in [map_dir]:
        if not base_dir.exists():
            continue

        for dir in base_dir.iterdir():
            if track_name == str(dir.stem).replace(" ", ""):
                return dir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {[map_dir]}")


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
        centerline: Raceline = None,
        raceline: Raceline = None,
    ):
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.centerline = centerline
        self.raceline = raceline

    @staticmethod
    def load_spec(track: str, filespec: str):
        """
        Load track specification from yaml file.

        Args:

        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            track_spec = TrackSpec(name=track, **map_metadata)
        return track_spec

    @staticmethod
    def from_track_name(track: str):
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
            raise FileNotFoundError(f"could not load track {track}") from ex

    def frenet_to_cartesian(self, s, ey, ephi):
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
        x, y = self.centerline.spline.calc_position(s)
        psi = self.centerline.spline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, psi
    
    def cartesian_to_frenet(self, x, y, phi, s_guess=0):
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
        s, ey = self.centerline.spline.calc_arclength(x, y, s_guess)
        if s > self.centerline.spline.s[-1]:
            # Wrap around
            s = s - self.centerline.spline.s[-1]
        if s < 0:
            # Negative s means we are behind the start point
            s = s + self.centerline.spline.s[-1]

        # Use the normal to calculate the signed lateral deviation
        normal = self.centerline.spline._calc_normal(s)
        x_eval, y_eval = self.centerline.spline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        phi = phi - self.centerline.spline.calc_yaw(s)

        return s, ey, phi

    
if __name__ == "__main__":
    track = Track.from_track_name("Example")
    print("[Result] map loaded successfully")
