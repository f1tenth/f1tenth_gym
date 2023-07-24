from typing import Dict

import numpy as np

from f110_gym.envs import Track


class PygameEnvRenderer:
    def __init__(
        self,
        render_mode: str,
        render_fps: int,
        width: int,
        height: int,
        zoom_level: float,
        car_length: float,
        car_width: float,
    ):
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.event.set_allowed([])
            flags = DOUBLEBUF
            self.window = pygame.display.set_mode((width, height), flags, 16)
            self.clock = pygame.time.Clock()

        self.poses = None
        self.colors = None
        self.track_map = None


    def update_map(self, track: Track) -> None:
        if self.track_map is None:
            track_map = self.map_img  # shape (W, H)
            track_map = np.stack([track_map, track_map, track_map], axis=-1)  # shape (W, H, 3)
            track_map = np.rot90(track_map, k=1)  # rotate clockwise
            track_map = np.flip(track_map, axis=0)  # flip vertically
            self.track_map = track_map
        self.canvas = pygame.surfarray.make_surface(self.track_map)

    def update_actors(self, poses: Dict[str, np.ndarray]) -> None:
        pass

    def update_labels(self) -> None:
        pass

    def render(self):
        self.canvas.fill((0, 0, 0))  # fill canvas with black
        self.render_map()

        origin = self.map_metadata['origin']
        resolution = self.map_metadata['resolution']
        for i in range(len(self.poses)):
            color, pose = self.colors[i], self.poses[i]

            vertices = get_vertices(pose, self.car_length, self.car_width)
            vertices[:, 0] = (vertices[:, 0] - origin[0]) / resolution
            vertices[:, 1] = (vertices[:, 1] - origin[1]) / resolution

            pygame.draw.lines(self.canvas, color, True, vertices, 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))
