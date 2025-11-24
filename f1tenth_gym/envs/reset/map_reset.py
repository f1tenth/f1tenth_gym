from abc import abstractmethod

import cv2
import numpy as np

from .reset_fn import ResetFn
from .utils import sample_around_pose
from ..track import Track


class MapResetFn(ResetFn):
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        pass

    def __init__(
        self,
        track: Track,
        num_agents: int,
        move_laterally: bool,
        min_dist: float,
        max_dist: float,
    ):
        self.track = track
        self.n_agents = num_agents
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.move_laterally = move_laterally
        # Mask is a 2D array of booleans of where the agents can be placed
        # Should acount for max_dist from obstacles
        self.mask = self.get_mask() 


    def sample(self) -> np.ndarray:
        # Random ample an x-y position from the mask
        valid_x, valid_y = np.where(self.mask)
        idx = np.random.choice(len(valid_x))
        pose_x = valid_x[idx] * self.track.spec.resolution + self.track.spec.origin[0]
        pose_y = valid_y[idx] * self.track.spec.resolution + self.track.spec.origin[1]
        pose_theta = np.random.uniform(-np.pi, np.pi)
        pose = np.array([pose_x, pose_y, pose_theta])
        
        poses = sample_around_pose(
            pose=pose,
            n_agents=self.n_agents,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
        )
        return poses

class AllMapResetFn(MapResetFn):
    def __init__(
        self,
        track: Track,
        num_agents: int,
        move_laterally: bool = True,
        shuffle: bool = True,
        min_dist: float = 0.5,
        max_dist: float = 1.0,
    ):
        super().__init__(
            track=track,
            num_agents=num_agents,
            move_laterally=move_laterally,
            min_dist=min_dist,
            max_dist=max_dist,
        )
        self.shuffle = shuffle

    def get_mask(self) -> np.ndarray:
        # Create mask from occupancy grid enlarged by max_dist
        dilation_size = int(self.max_dist / self.track.spec.resolution)
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        inverted_occ_map = (255 - self.track.occupancy_map)
        dilated = cv2.dilate(inverted_occ_map, kernel, iterations=1)
        dilated_inverted = (255 - dilated)
        return dilated_inverted == 255

    def sample(self) -> np.ndarray:
        poses = super().sample()

        if self.shuffle:
            np.random.shuffle(poses)

        return poses
