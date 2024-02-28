from f110_gym.envs.track import Track
from scipy.ndimage import distance_transform_edt as edt
import numpy as np

DEFAULT_MAP_NAMES = [
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "YasMarina",
    "Zandvoort",
]

for track_name in DEFAULT_MAP_NAMES:
    print("Loading a map without edt, a warning should appear")
    track = Track.from_track_name(track_name)
    occupancy_map = track.occupancy_map
    resolution = track.spec.resolution

    dt = resolution * edt(occupancy_map)

    # saving
    np.save(track.filepath, dt)

    print("Loading a map with edt, warning should no longer appear")
    track_wedt = Track.from_track_name(track_name)
