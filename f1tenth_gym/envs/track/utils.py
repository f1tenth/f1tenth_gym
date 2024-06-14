import pathlib
import tarfile
import tempfile

import numpy as np
import requests
from numba import njit


def find_track_dir(track_name: str) -> pathlib.Path:
    """
    Find the directory of the track map corresponding to the given track name.

    Parameters
    ----------
    track_name : str
        name of the track

    Returns
    -------
    pathlib.Path
        path to the track map directory

    Raises
    ------
    FileNotFoundError
        if no map directory matching the track name is found
    """
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

        print("Extracting Files for: " + track_name)
        tracks_file = tarfile.open(tempdir + track_name + ".tar.xz")
        tracks_file.extractall(map_dir)
        tracks_file.close()

    # search for map in the map directory
    for subdir in map_dir.iterdir():
        if track_name == str(subdir.stem).replace(" ", ""):
            return subdir

    raise FileNotFoundError(f"no mapdir matching {track_name} in {[map_dir]}")


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> tuple:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    Parameters
    ----------
    point: np.ndarray
        The 2d point to project onto the trajectory
    trajectory: np.ndarray
        The trajectory to project the point onto, shape (N, 2)
        The points must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns
    -------
    nearest_point: np.ndarray
        The nearest point on the trajectory
    distance: float
        The distance from the point to the nearest point on the trajectory
    t: float
    min_dist_segment: int
        The index of the nearest point on the trajectory
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
