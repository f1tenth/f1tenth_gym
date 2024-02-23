from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig

from f110_gym.envs.track import Raceline
from f110_gym.envs.track.utils import find_track_dir


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
        centerline: Raceline = None,
        raceline: Raceline = None,
    ):
        """
        Initialize track object.

        Args:
            spec: track specification containing metadata
            filepath: path to the track map image
            ext: extension of the track map image
            occupancy_map: binary occupancy map
            centerline: centerline of the track
            raceline: raceline of the track
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

        Args:
            track: name of the track
            filespec: path to the yaml file

        Returns:
            TrackSpec: track specification object
        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            track_spec = TrackSpec(name=track, **map_metadata)
        return track_spec

    @staticmethod
    def from_track_name(track: str) -> Track:
        """
        Load track from track name.

        Args:
            track: name of the track

        Returns:
            Track: track object

        Raises:
            FileNotFoundError: if the track cannot be loaded
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
