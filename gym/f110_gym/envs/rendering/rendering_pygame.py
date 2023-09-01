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


class FPS():
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, display):
        txt = f"FPS: {self.clock.get_fps():.2f}"
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom left corner of display
        display_height = display.get_height()
        text_height = self.text.get_height()
        bottom_left = (0, display_height - text_height)

        display.blit(self.text, bottom_left)

class Map:
    def __init__(self, map_img: np.ndarray):
        # from shape (W, H) to (W, H, 3)
        track_map = np.stack(
            [map_img, map_img, map_img], axis=-1
        )

        track_map = np.rot90(track_map, k=1)  # rotate clockwise
        track_map = np.flip(track_map, axis=0)  # flip vertically

        self.track_map = track_map
        self.map_surface = pygame.surfarray.make_surface(self.track_map)

    def render(self, display):
        display.blit(self.map_surface, (0, 0))


class PygameEnvRenderer(EnvRenderer):
    def __init__(self, track: Track, render_spec: RenderSpec, render_mode: str):
        super().__init__()

        self.window = None
        self.canvas = None

        self.clock = None
        self.render_fps = render_spec.render_fps
        self.render_mode = render_mode

        width, height = render_spec.window_size, render_spec.window_size
        self.zoom_level = render_spec.zoom_in_factor

        self.car_length = render_spec.car_length
        self.car_width = render_spec.car_width
        self.car_tickness = render_spec.car_tickness
        self.cars = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.event.set_allowed([])
            self.window = pygame.display.set_mode((width, height))
            self.window.fill((255, 255, 255))  # white background
            self.clock = pygame.time.Clock()

            self.fps = FPS()

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

        dwidth = int(width * self.zoom_level)
        dheight = int(height * self.zoom_level)
        self.map_img = cv2.resize(
            original_img, dsize=(dwidth, dheight), interpolation=cv2.INTER_AREA
        )
        self.ppu = original_img.shape[0] / self.map_img.shape[0]  # pixels per unit

        self.canvas = pygame.Surface((dwidth, dheight))
        self.map_renderer = Map(self.map_img)

    def update(self, state):
        self.colors = [
            (255, 0, 0) if state["collisions"][i] else (0, 125, 0)
            for i in range(len(state["poses_x"]))
        ]
        self.poses = np.stack(
            (state["poses_x"], state["poses_y"], state["poses_theta"])
        ).T
        self.steering_angles = state["steering_angles"]

    def add_renderer_callback(self, callback_fn: callable):
        warnings.warn("add_render_callback is not implemented for PygameEnvRenderer")

    def render_map(self):
        """
        if self.track_map is None:
            track_map = self.map_img  # shape (W, H)
            track_map = np.stack(
                [track_map, track_map, track_map], axis=-1
            )  # shape (W, H, 3)
            track_map = np.rot90(track_map, k=1)  # rotate clockwise
            track_map = np.flip(track_map, axis=0)  # flip vertically
            self.track_map = track_map
        self.canvas = pygame.surfarray.make_surface(self.track_map)
        """
        self.map_renderer.render(self.canvas)

    def render(self):
        origin = self.map_metadata["origin"]
        resolution = self.map_metadata["resolution"] * self.ppu
        car_length = self.car_length
        car_width = self.car_width
        car_tickness = self.car_tickness

        print("ppu: ", self.ppu)
        print("res: ", resolution)
        print("car_length: ", car_length)
        print("car_width: ", car_width)
        print()

        self.window.fill((255, 255, 255))  # white background
        self.canvas.fill((255, 255, 255))  # white background
        self.map_renderer.render(self.canvas)

        # draw cars
        for i in range(len(self.poses)):
            color, pose = self.colors[i], self.poses[i]

            vertices = get_vertices(pose, car_length, car_width)
            vertices[:, 0] = (vertices[:, 0] - origin[0]) / resolution
            vertices[:, 1] = (vertices[:, 1] - origin[1]) / resolution
            pygame.draw.lines(self.canvas, color, True, vertices, 3)

            # draw car steering angle from front center
            arrow_length = 0.2 / resolution

            #center = np.array([pose[0], pose[1]])
            #center[0] = (center[0] - origin[0]) / resolution
            #center[1] = (center[1] - origin[1]) / resolution

            front_center = (vertices[2] + vertices[3]) / 2

            steering_angle = pose[2] + self.steering_angles[i]
            end_point = front_center + arrow_length * np.array([np.cos(steering_angle), np.sin(steering_angle)])
            pygame.draw.line(self.canvas, color, front_center.astype(int), end_point.astype(int), 3)

        # follow the first car
        surface_mod_rect = self.canvas.get_rect()
        screen_rect = self.window.get_rect()

        # agent to follow
        ego_x, ego_y = self.poses[0, 0], self.poses[0, 1]
        ego_x = (ego_x - origin[0]) / resolution
        ego_y = (ego_y - origin[1]) / resolution

        surface_mod_rect.x = (screen_rect.centerx - ego_x)
        surface_mod_rect.y = (screen_rect.centery - ego_y)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, surface_mod_rect)

            # fps
            if self.window is not None:
                self.fps.render(self.window)

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.fps.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
