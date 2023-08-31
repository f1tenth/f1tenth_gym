import pathlib
import warnings

import cv2
import numpy as np
import pygame
import yaml
from PIL import Image
from pygame.locals import *

from f110_gym.envs import get_vertices
from f110_gym.envs.track import Track
from f110_gym.envs.rendering.renderer import EnvRenderer, RenderSpec


class CarSprite(pygame.sprite.Sprite):
    def __init__(self, color, height, width):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill((0, 0, 0))
        self.image.set_colorkey(color)

        pygame.draw.rect(self.image, color, pygame.Rect(0, 0, width, height))

        self.rect = self.image.get_rect()


class PygameEnvRenderer(EnvRenderer):
    def __init__(self, track: Track, render_spec: RenderSpec):
        super().__init__()

        self.window = None
        self.canvas = None
        self.clock = None
        self.render_fps = render_spec.render_fps
        self.render_mode = render_spec.render_mode

        width, height = (
            1600,
            1600,
        )  # render_spec.window_width, render_spec.window_height
        # self.zoom_level = render_spec.zoom_in_factor

        self.car_length = render_spec.car_length
        self.car_width = render_spec.car_width
        self.cars = None

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
        self.track = track

        # load map metadata
        map_filepath = pathlib.Path(track.filepath)
        map_yaml = map_filepath.with_suffix(".yaml")
        with open(map_yaml, "r") as yaml_stream:
            try:
                self.map_metadata = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        original_img = map_filepath.parent / self.map_metadata["image"]
        original_img = np.array(
            Image.open(original_img).transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)
        self.map_img = cv2.resize(
            original_img, dsize=(width, height), interpolation=cv2.INTER_AREA
        )
        self.ppu = original_img.shape[0] / self.map_img.shape[0]  # pixels per unit

        self.render_map()

    def update(self, state):
        self.colors = [
            (255, 0, 0) if state["collisions"][i] else (0, 125, 0)
            for i in range(len(state["poses_x"]))
        ]
        self.poses = np.stack(
            (state["poses_x"], state["poses_y"], state["poses_theta"])
        ).T

    def add_renderer_callback(self, callback_fn: callable):
        warnings.warn("add_render_callback is not implemented for PygameEnvRenderer")

    def render_map(self):
        if self.track_map is None:
            track_map = self.map_img  # shape (W, H)
            track_map = np.stack(
                [track_map, track_map, track_map], axis=-1
            )  # shape (W, H, 3)
            track_map = np.rot90(track_map, k=1)  # rotate clockwise
            track_map = np.flip(track_map, axis=0)  # flip vertically
            self.track_map = track_map
        self.canvas = pygame.surfarray.make_surface(self.track_map)

    def render(self):
        origin = self.map_metadata["origin"]
        resolution = self.map_metadata["resolution"] * self.ppu
        car_length = self.car_length / self.ppu
        car_width = self.car_width / self.ppu

        if self.cars is None and len(self.poses) > 0:
            self.cars = pygame.sprite.Group()

            for i in range(len(self.poses)):
                color, pose = self.colors[i], self.poses[i]
                car = CarSprite(color, car_width, car_length)
                car.rect.x = (pose[0] - origin[0]) / resolution
                car.rect.y = (pose[1] - origin[1]) / resolution
                self.cars.add(car)

        self.cars.update()
        self.canvas.fill((0, 0, 0))  # fill canvas with black
        self.render_map()
        self.cars.draw(self.window)

        for i in range(len(self.poses)):
            color, pose = self.colors[i], self.poses[i]

            vertices = get_vertices(pose, car_length, car_width)
            vertices[:, 0] = ((vertices[:, 0] - origin[0]) / resolution)
            vertices[:, 1] = ((vertices[:, 1] - origin[1]) / resolution)
            pygame.draw.lines(self.canvas, color, True, vertices, 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick()
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )