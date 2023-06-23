import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig


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
    map_dir = pathlib.Path(__file__).parent.parent / "maps"

    for base_dir in [map_dir]:
        if not base_dir.exists():
            continue

        for dir in base_dir.iterdir():
            if track_name == str(dir.stem).replace(" ", ""):
                return dir

    raise FileNotFoundError(f'no mapdir matching {track_name} in {[map_dir]}')


@dataclass
class Track:
    spec: TrackSpec
    filepath: str
    ext: str
    occupancy_map: np.ndarray

    def __init__(
            self,
            spec: TrackSpec,
            filepath: str,
            ext: str,
            occupancy_map: np.ndarray,
    ):
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map

    @staticmethod
    def from_track_name(track: str):
        try:
            track_dir = find_track_dir(track)
            # load track spec
            with open(track_dir / f"{track}_map.yaml", 'r') as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
                track_spec = TrackSpec(name=track, **map_metadata)

            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            image = Image.open(track_dir / str(map_filename)).transpose(Transpose.FLIP_TOP_BOTTOM)
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            return Track(
                spec=track_spec,
                filepath=str((track_dir / map_filename.stem).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
            )

        except Exception as ex:
            raise FileNotFoundError(f"could not load track {track}") from ex


if __name__ == "__main__":
    track = Track.from_track_name("Example")
    print("[Result] map loaded successfully")
