import cv2
import numpy as np
import pygame

from f110_gym.envs.collision_models import get_vertices


class FPS:
    def __init__(self, window_shape=(1000, 1000)):
        self.clock = pygame.time.Clock()
        font_size = int(32 * window_shape[0] / 1000)
        self.font = pygame.font.SysFont("Arial", font_size)

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
    def __init__(self, window_shape=(1000, 1000)):
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
    def __init__(self, window_shape=(1000, 1000)):
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
    def __init__(self, window_shape=(1000, 1000)):
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
    def __init__(self, map_img: np.ndarray, zoom_level: float):
        width, height = map_img.shape
        dwidth = int(width * zoom_level)
        dheight = int(height * zoom_level)
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
    def __init__(
        self,
        render_spec,
        map_origin,
        resolution,
        ppu,
        car_length,
        car_width,
        color=None,
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.steering_arrow_len = 0.2
        self.car_tickness = render_spec.car_tickness
        self.show_wheels = render_spec.show_wheels

        self.origin = map_origin
        self.resolution = resolution
        self.ppu = ppu

        self.color = color or (0, 0, 0)
        self.pose = None
        self.rect = None

    def update(self, state, idx: int):
        self.pose = (
            state["poses_x"][idx],
            state["poses_y"][idx],
            state["poses_theta"][idx],
        )
        self.color = (255, 0, 0) if state["collisions"][idx] > 0 else self.color
        self.steering_angle = self.pose[2] + state["steering_angles"][idx]

    def render(self, display):
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
            lam = 0.15
            # find point at perc between front and back vertices
            front_left = (vertices[0] * lam + vertices[3] * (1 - lam)).astype(int)
            front_right = (vertices[1] * lam + vertices[2] * (1 - lam)).astype(int)
            arrow_length = self.steering_arrow_len / self.resolution

            for mid_point in [front_left, front_right]:
                end_point = mid_point + 0.5 * arrow_length * np.array(
                    [np.cos(self.steering_angle), np.sin(self.steering_angle)]
                )
                base_point = mid_point - 0.5 * arrow_length * np.array(
                    [np.cos(self.steering_angle), np.sin(self.steering_angle)]
                )

                pygame.draw.line(
                    display,
                    (0, 0, 0),
                    base_point.astype(int),
                    end_point.astype(int),
                    self.car_tickness + 1,
                )
