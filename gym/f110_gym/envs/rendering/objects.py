from __future__ import annotations
import cv2
import numpy as np
import pygame

from f110_gym.envs.collision_models import get_vertices
from f110_gym.envs.rendering import RenderSpec


class FPS:
    """
    Class to display the current FPS on the screen.
    """

    def __init__(self, window_shape: tuple[int, int] = (1000, 1000)):
        self.clock = pygame.time.Clock()
        font_size = int(32 * window_shape[0] / 1000)
        self.font = pygame.font.SysFont("Arial", font_size)

        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, display: pygame.Surface):
        txt = f"FPS: {self.clock.get_fps():.2f}"
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom left corner of display
        display_height = display.get_height()
        text_height = self.text.get_height()
        bottom_left = (0, display_height - text_height)

        display.blit(self.text, bottom_left)


class Timer:
    """
    Class to display the current time on the screen.
    """

    def __init__(self, window_shape: tuple[int, int] = (1000, 1000)):
        self.font = pygame.font.SysFont("Arial", 32)
        font_size = int(32 * window_shape[0] / 1000)
        self.font = pygame.font.SysFont("Arial", font_size)

    def render(self, time: float, display: pygame.Surface):
        txt = f"Time: {time:.2f}"
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom right corner of display
        display_width, display_height = display.get_width(), display.get_height()
        text_width, text_height = self.text.get_width(), self.text.get_height()
        bottom_right = (display_width - text_width, display_height - text_height)

        display.blit(self.text, bottom_right)


class BottomInfo:
    """
    Class to display text on the bottom of the screen.
    """

    def __init__(self, window_shape: tuple[int, int] = (1000, 1000)):
        font_size = int(32 * window_shape[0] / 1000)
        self.font = pygame.font.SysFont("Arial", font_size)
        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, txt: str, display: pygame.Surface):
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find bottom center of display
        display_width, display_height = display.get_width(), display.get_height()
        text_width, text_height = self.text.get_width(), self.text.get_height()
        bottom_center = ((display_width - text_width) / 2, display_height - text_height)

        display.blit(self.text, bottom_center)


class TopInfo:
    """
    Class to display text on the top of the screen.
    """

    def __init__(self, window_shape: tuple[int, int] = (1000, 1000)):
        font_size = int(32 * window_shape[0] / 1000)
        self.font = pygame.font.SysFont("Arial", font_size)
        self.text = self.font.render("", True, (125, 125, 125))

    def render(self, txt: str, display: pygame.Surface):
        self.text = self.font.render(txt, True, (125, 125, 125))

        # find top center of display
        display_width, display_height = display.get_width(), display.get_height()
        text_width, text_height = self.text.get_width(), self.text.get_height()
        top_center = ((display_width - text_width) / 2, 0)

        display.blit(self.text, top_center)


class Map:
    """
    Class to display the track map according to the desired zoom level.
    """

    def __init__(self, map_img: np.ndarray, zoom_level: float):
        orig_width, orig_height = map_img.shape
        scaled_width = int(orig_width * zoom_level)
        scaled_height = int(orig_height * zoom_level)
        map_img = cv2.resize(
            map_img, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )

        # convert shape from (W, H) to (W, H, 3)
        track_map = np.stack([map_img, map_img, map_img], axis=-1)

        # rotate and flip to match the track orientation
        track_map = np.rot90(track_map, k=1)  # rotate clockwise
        track_map = np.flip(track_map, axis=0)  # flip vertically

        self.track_map = track_map
        self.map_surface = pygame.surfarray.make_surface(self.track_map)

    def render(self, display: pygame.Surface):
        display.blit(self.map_surface, (0, 0))


class Car:
    """
    Class to display the car.
    """

    def __init__(
        self,
        render_spec: RenderSpec,
        map_origin: tuple[float, float],
        resolution: float,
        ppu: float,
        car_length: float,
        car_width: float,
        color: list[int] | None = None,
        wheel_size: float = 0.2,
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.wheel_size = wheel_size
        self.car_tickness = render_spec.car_tickness
        self.show_wheels = render_spec.show_wheels

        self.origin = map_origin
        self.resolution = resolution
        self.ppu = ppu

        self.color = color or (0, 0, 0)
        self.pose = None
        self.steering = None
        self.rect = None

    def update(self, state: dict[str, np.ndarray], idx: int):
        self.pose = (
            state["poses_x"][idx],
            state["poses_y"][idx],
            state["poses_theta"][idx],
        )
        self.color = (255, 0, 0) if state["collisions"][idx] > 0 else self.color
        self.steering = self.pose[2] + state["steering_angles"][idx]

    def render(self, display: pygame.Surface):
        vertices = get_vertices(self.pose, self.car_length, self.car_width)
        vertices[:, 0] = (vertices[:, 0] - self.origin[0]) / (
            self.resolution * self.ppu
        )
        vertices[:, 1] = (vertices[:, 1] - self.origin[1]) / (
            self.resolution * self.ppu
        )

        self.rect = pygame.draw.polygon(display, self.color, vertices)

        pygame.draw.lines(display, (0, 0, 0), True, vertices, self.car_tickness)

        # draw two lines in proximity of the front wheels
        # to indicate the steering angle
        if self.show_wheels:
            # percentage along the car length to draw the wheels segments
            lam = 0.15

            # find point at perc between front and back vertices
            front_left = (vertices[0] * lam + vertices[3] * (1 - lam)).astype(int)
            front_right = (vertices[1] * lam + vertices[2] * (1 - lam)).astype(int)
            arrow_length = self.wheel_size / self.resolution

            for mid_point in [front_left, front_right]:
                end_point = mid_point + 0.5 * arrow_length * np.array(
                    [np.cos(self.steering), np.sin(self.steering)]
                )
                base_point = mid_point - 0.5 * arrow_length * np.array(
                    [np.cos(self.steering), np.sin(self.steering)]
                )

                pygame.draw.line(
                    display,
                    (0, 0, 0),
                    base_point.astype(int),
                    end_point.astype(int),
                    self.car_tickness + 1,
                )
