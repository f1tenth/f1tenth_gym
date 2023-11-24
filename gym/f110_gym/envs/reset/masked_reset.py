from abc import abstractmethod

import numpy as np

from f110_gym.envs.reset.reset_fn import ResetFn
from f110_gym.envs.reset.utils import sample_around_waypoint
from f110_gym.envs.track import Track


class MaskedResetFn(ResetFn):
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
        self.mask = self.get_mask()

    def sample(self) -> np.ndarray:
        waypoint_id = np.random.choice(np.where(self.mask)[0])
        poses = sample_around_waypoint(
            track=self.track,
            waypoint_id=waypoint_id,
            n_agents=self.n_agents,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            move_laterally=self.move_laterally,
        )
        return poses


class GridResetFn(MaskedResetFn):
    def __init__(
        self,
        track: Track,
        num_agents: int,
        move_laterally: bool = True,
        shuffle: bool = True,
        start_width: float = 1.0,
        min_dist: float = 1.5,
        max_dist: float = 2.5,
    ):
        self.start_width = start_width
        self.shuffle = shuffle

        super().__init__(
            track=track,
            num_agents=num_agents,
            move_laterally=move_laterally,
            min_dist=min_dist,
            max_dist=max_dist,
        )



    def get_mask(self) -> np.ndarray:
        # approximate the nr waypoints in the starting line
        step_size = self.track.centerline.length / self.track.centerline.n
        n_wps = int(self.start_width / step_size)

        mask = np.zeros(self.track.centerline.n)
        mask[: n_wps] = 1
        return mask.astype(bool)

    def sample(self) -> np.ndarray:
        poses = super().sample()

        if self.shuffle:
            np.random.shuffle(poses)

        return poses


class AllTrackResetFn(MaskedResetFn):
    def __init__(
        self,
        track: Track,
        num_agents: int,
        move_laterally: bool = True,
        shuffle: bool = True,
        min_dist: float = 1.5,
        max_dist: float = 2.5,
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
        return np.ones(self.track.centerline.n).astype(bool)

    def sample(self) -> np.ndarray:
        poses = super().sample()

        if self.shuffle:
            np.random.shuffle(poses)

        return poses
