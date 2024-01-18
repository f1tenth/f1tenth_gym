from f110_gym.envs.track import Track
from scipy.ndimage import distance_transform_edt as edt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--track_name", type=str, required=True)
args = parser.parse_args()

print("Loading a map without edt, a warning should appear")
track = Track.from_track_name(args.track_name)
occupancy_map = track.occupancy_map
resolution = track.spec.resolution

dt = resolution * edt(occupancy_map)

# saving
np.save(track.filepath, dt)

print("Loading a map with edt, warning should no longer appear")
track_wedt = Track.from_track_name(args.track_name)