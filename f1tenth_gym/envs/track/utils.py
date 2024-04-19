import pathlib
import tarfile
import tempfile

import requests


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
