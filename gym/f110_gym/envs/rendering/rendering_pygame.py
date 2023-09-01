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


class FPS:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32)

        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, display):
        txt = f"FPS: {self.clock.get_fps():.2f}"
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom left corner of display
        display_height = display.get_height()
        text_height = self.text.get_height()
        bottom_left = (0, display_height - text_height)

        display.blit(self.text, bottom_left)

class Timer:
    def __init__(self):
        self.font = pygame.font.SysFont("Arial", 32)
        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, time: float, display: pygame.Surface):
        txt = f"Time: {time:.2f}"
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom right corner of display
        display_width, display_height = display.get_width(), display.get_height()
        text_width, text_height = self.text.get_width(), self.text.get_height()
        bottom_right = (display_width - text_width, display_height - text_height)

        display.blit(self.text, bottom_right)


class Map:
    def __init__(self, map_img: np.ndarray, render_spec: RenderSpec):
        width, height = map_img.shape
        dwidth = int(width * render_spec.zoom_in_factor)
        dheight = int(height * render_spec.zoom_in_factor)
        map_img = cv2.resize(
            map_img, dsize=(dwidth, dheight), interpolation=cv2.INTER_AREA
        )

        # from shape (W, H) to (W, H, 3)
        track_map = np.stack([map_img, map_img, map_img], axis=-1)

        track_map = np.rot90(track_map, k=1)  # rotate clockwise
        track_map = np.flip(track_map, axis=0)  # flip vertically

        self.track_map = track_map
        self.map_surface = pygame.surfarray.make_surface(self.track_map)

    def render(self, display):
        display.blit(self.map_surface, (0, 0))


class Car:
    def __init__(self, render_spec, map_origin, resolution, ppu):
        self.car_length = render_spec.car_length
        self.car_width = render_spec.car_width
        self.steering_arrow_len = 0.2
        self.car_tickness = render_spec.car_tickness

        self.origin = map_origin
        self.resolution = resolution * ppu

        self.color = None
        self.pose = None

    def render(self, pose, steering, color, display):
        vertices = get_vertices(pose, self.car_length, self.car_width)
        vertices[:, 0] = (vertices[:, 0] - self.origin[0]) / self.resolution
        vertices[:, 1] = (vertices[:, 1] - self.origin[1]) / self.resolution
        pygame.draw.lines(display, color, True, vertices, self.car_tickness)

        # draw two lines in proximity of the front wheels
        # to indicate the steering angle
        lam = 0.15
        # find point at perc between front and back vertices
        front_left = (vertices[0] * lam + vertices[3] * (1 - lam)).astype(int)
        front_right = (vertices[1] * lam + vertices[2] * (1 - lam)).astype(int)
        arrow_length = self.steering_arrow_len / self.resolution

        steering_angle = pose[2] + steering
        for mid_point in [front_left, front_right]:
            end_point = mid_point + 0.5 * arrow_length * np.array(
                [np.cos(steering_angle), np.sin(steering_angle)]
            )
            base_point = mid_point - 0.5 * arrow_length * np.array(
                [np.cos(steering_angle), np.sin(steering_angle)]
            )

            pygame.draw.line(
                display,
                (0, 0, 0),
                base_point.astype(int),
                end_point.astype(int),
                self.car_tickness + 1,
            )


class PygameEnvRenderer(EnvRenderer):
    def __init__(self, track: Track, render_spec: RenderSpec, render_mode: str):
        super().__init__()

        self.window = None
        self.canvas = None
        self.map_canvas = None

        self.time_renderer = None

        self.render_spec = render_spec
        self.render_fps = render_spec.render_fps
        self.render_mode = render_mode
        self.zoom_level = render_spec.zoom_in_factor

        self.car_length = render_spec.car_length
        self.car_width = render_spec.car_width
        self.car_tickness = render_spec.car_tickness
        self.poses = None
        self.colors = None
        self.cars = None

        width, height = render_spec.window_size, render_spec.window_size

        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            pygame.event.set_allowed([])
            self.window = pygame.display.set_mode((width, height))
            self.window.fill((255, 255, 255))  # white background

        self.canvas = pygame.Surface((width, height))

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

        self.map_renderer = Map(map_img=original_img, render_spec=render_spec)
        mapwidth, mapheight, _ = self.map_renderer.track_map.shape
        self.ppu = (
            original_img.shape[0] / self.map_renderer.track_map.shape[0]
        )  # pixels per unit

        self.map_canvas = pygame.Surface((mapwidth, mapheight))

        # fps and time renderer
        self.fps = FPS()
        self.time_renderer = Timer()

    def update(self, state):
        if self.cars is None:
            self.cars = [
                Car(
                    render_spec=self.render_spec,
                    map_origin=self.map_metadata["origin"],
                    resolution=self.map_metadata["resolution"],
                    ppu=self.ppu,
                )
                for _ in range(len(state["poses_x"]))
            ]

        self.colors = [
            (255, 0, 0) if state["collisions"][i] else (0, 125, 0)
            for i in range(len(state["poses_x"]))
        ]
        self.poses = np.stack(
            (state["poses_x"], state["poses_y"], state["poses_theta"])
        ).T
        self.steering_angles = state["steering_angles"]
        self.sim_time = state["sim_time"]

    def add_renderer_callback(self, callback_fn: callable):
        warnings.warn("add_render_callback is not implemented for PygameEnvRenderer")

    def render(self):
        origin = self.map_metadata["origin"]
        resolution = self.map_metadata["resolution"] * self.ppu
        car_length = self.car_length
        car_width = self.car_width
        car_tickness = self.car_tickness

        self.canvas.fill((255, 255, 255))  # white background
        self.map_canvas.fill((255, 255, 255))  # white background
        self.map_renderer.render(self.map_canvas)

        # draw cars
        for i in range(len(self.poses)):
            color, pose, steering = (
                self.colors[i],
                self.poses[i],
                self.steering_angles[i],
            )
            car = self.cars[i]

            car.render(pose, steering, color, self.map_canvas)  # directly give state

        # follow the first car
        surface_mod_rect = self.map_canvas.get_rect()
        screen_rect = self.canvas.get_rect()

        # agent to follow
        ego_x, ego_y = self.poses[0, 0], self.poses[0, 1]
        ego_x = (ego_x - origin[0]) / resolution
        ego_y = (ego_y - origin[1]) / resolution

        surface_mod_rect.x = screen_rect.centerx - ego_x
        surface_mod_rect.y = screen_rect.centery - ego_y
        self.canvas.blit(self.map_canvas, surface_mod_rect)

        self.time_renderer.render(time=self.sim_time, display=self.canvas)

        if self.render_mode == "human":
            assert self.window is not None
            self.fps.render(self.canvas)

            self.window.blit(self.canvas, self.canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.fps.clock.tick(self.render_fps)
        else:  # rgb_array
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
            if frame.shape[0] > 2000:
                frame = cv2.resize(
                    frame, dsize=(2000, 2000), interpolation=cv2.INTER_AREA
                )
            return frame
