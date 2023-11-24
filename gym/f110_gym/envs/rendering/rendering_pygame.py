from __future__ import annotations

import logging
import math
import pathlib
from typing import Union, List, Tuple, Any

import cv2
import matplotlib
import numpy as np
import pygame
import yaml
from PIL import Image

from f110_gym.envs.rendering.objects import FPS, Timer, BottomInfo, Map, Car, TopInfo
from f110_gym.envs.track import Track
from f110_gym.envs.rendering.renderer import EnvRenderer, RenderSpec

INSTRUCTION_TEXT = "Mouse click (L/M/R): Change POV - 'S' key: On/Off"


class PygameEnvRenderer(EnvRenderer):
    def __init__(
        self,
        params: dict[str, Any],
        track: Track,
        agent_ids: list[str],
        render_spec: RenderSpec,
        render_mode: str,
        render_fps: int,
    ):
        super().__init__()

        self.params = params

        self.callbacks = []
        self.window = None
        self.canvas = None

        self.time_renderer = None

        self.render_spec = render_spec
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.zoom_level = render_spec.zoom_in_factor

        self.cars = None
        self.agent_ids = agent_ids

        colors_rgb = [
            [255 * rgb for rgb in matplotlib.colors.hex2color(c)]
            for c in render_spec.vehicle_palette
        ]
        self.car_colors = [
            colors_rgb[i % len(colors_rgb)] for i in range(len(self.agent_ids))
        ]

        width, height = render_spec.window_size, render_spec.window_size

        pygame.init()
        if self.render_mode in ["human", "human_fast"]:
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

        # fps and time renderer
        self.fps_renderer = FPS(window_shape=(width, height))
        self.time_renderer = Timer(window_shape=(width, height))
        self.bottom_info_renderer = BottomInfo(window_shape=(width, height))
        self.top_info_renderer = TopInfo(window_shape=(width, height))

        # load map image
        original_img = map_filepath.parent / self.map_metadata["image"]
        original_img = np.array(
            Image.open(original_img).transpose(Image.FLIP_TOP_BOTTOM)
        ).astype(np.float64)

        self.map_renderers = {
            "map": Map(map_img=original_img, zoom_level=0.4),
            "car": Map(map_img=original_img, zoom_level=render_spec.zoom_in_factor),
        }
        self.map_canvases = {
            k: pygame.Surface((map_r.track_map.shape[0], map_r.track_map.shape[1]))
            for k, map_r in self.map_renderers.items()
        }
        self.ppus = {
            k: original_img.shape[0] / map_r.track_map.shape[0]
            for k, map_r in self.map_renderers.items()
        }

        # event handling flags
        self.draw_flag: bool = True
        if render_spec.focus_on:
            self.active_map_renderer = "car"
            self.follow_agent_flag: bool = True
            self.agent_to_follow: int = self.agent_ids.index(render_spec.focus_on)
        else:
            self.active_map_renderer = "map"
            self.follow_agent_flag: bool = False
            self.agent_to_follow: int = None

    def update(self, state):
        if self.cars is None:
            self.cars = [
                Car(
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_metadata["origin"],
                    resolution=self.map_metadata["resolution"],
                    ppu=self.ppus[self.active_map_renderer],
                )
                for ic in range(len(self.agent_ids))
            ]

        for i in range(len(self.agent_ids)):
            self.cars[i].update(state, i)
            self.cars[i].ppu = self.ppus[self.active_map_renderer]

        self.sim_time = state["sim_time"]

    def add_renderer_callback(self, callback_fn: callable):
        self.callbacks.append(callback_fn)

    def render(self):
        self.event_handling()

        self.canvas.fill((255, 255, 255))  # white background
        self.map_canvas = self.map_canvases[self.active_map_renderer]
        self.map_canvas.fill((255, 255, 255))  # white background

        if self.draw_flag:
            self.map_renderers[self.active_map_renderer].render(self.map_canvas)

            # draw cars
            for i in range(len(self.agent_ids)):
                self.cars[i].render(self.map_canvas)

            # call callbacks
            for callback_fn in self.callbacks:
                callback_fn(self)

            surface_mod_rect = self.map_canvas.get_rect()
            screen_rect = self.canvas.get_rect()

            if self.follow_agent_flag:
                origin = self.map_metadata["origin"]
                resolution = (
                    self.map_metadata["resolution"]
                    * self.ppus[self.active_map_renderer]
                )
                ego_x, ego_y = self.cars[self.agent_to_follow].pose[:2]
                cx = (ego_x - origin[0]) / resolution
                cy = (ego_y - origin[1]) / resolution
            else:
                cx = self.map_canvas.get_width() / 2
                cy = self.map_canvas.get_height() / 2

            surface_mod_rect.x = screen_rect.centerx - cx
            surface_mod_rect.y = screen_rect.centery - cy

            self.canvas.blit(self.map_canvas, surface_mod_rect)

            agent_to_follow_id = (
                self.agent_ids[self.agent_to_follow]
                if self.agent_to_follow is not None
                else None
            )
            self.bottom_info_renderer.render(
                txt=f"Focus on: {agent_to_follow_id}", display=self.canvas
            )

        if self.render_spec.show_info:
            self.top_info_renderer.render(txt=INSTRUCTION_TEXT, display=self.canvas)
        self.time_renderer.render(time=self.sim_time, display=self.canvas)

        if self.render_mode in ["human", "human_fast"]:
            assert self.window is not None

            self.fps_renderer.render(self.canvas)

            self.window.blit(self.canvas, self.canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.fps_renderer.clock.tick(self.render_fps)
        else:  # rgb_array
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
            if frame.shape[0] > 2000:
                frame = cv2.resize(
                    frame, dsize=(2000, 2000), interpolation=cv2.INTER_AREA
                )
            return frame

    def event_handling(self):

        for event in pygame.event.get():

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                logging.debug("Pressed left button -> Follow Next agent")
                self.follow_agent_flag = True
                if self.agent_to_follow is None:
                    self.agent_to_follow = 0
                else:
                    self.agent_to_follow = (self.agent_to_follow + 1) % len(
                        self.agent_ids
                    )
                self.active_map_renderer = "car"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                logging.debug("Pressed right button -> Follow Previous agent")
                self.follow_agent_flag = True
                if self.agent_to_follow is None:
                    self.agent_to_follow = 0
                else:
                    self.agent_to_follow = (self.agent_to_follow - 1) % len(
                        self.agent_ids
                    )
                self.active_map_renderer = "car"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                logging.debug("Pressed middle button -> Change to Map View")
                self.follow_agent_flag = False
                self.agent_to_follow = None
                self.active_map_renderer = "map"

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                logging.debug("Pressed S key -> Enable/disable rendering")
                self.draw_flag = not (self.draw_flag)

    def render_points(
        self,
        points: Union[List, np.ndarray],
        color: Tuple[int, int, int] = (0, 0, 255),
        size: int = 1,
    ):
        """
        Render a sequence of xy points on screen.

        Args:
            points: sequence of xy points (N, 2)
            color: rgb color of the points
            size: size of the points in pixels
        """
        origin = self.map_metadata["origin"]
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_metadata["resolution"] * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        for point in points:
            pygame.draw.circle(self.map_canvas, color, point, size)

    def render_lines(
        self,
        points: Union[List, np.ndarray],
        color: Tuple[int, int, int] = (0, 0, 255),
        size: int = 1,
    ):
        """
        Render a sequence of lines segments.

        Args:
            points: sequence of xy points (N, 2)
            color: rgb color of the points
            size: size of the points in pixels
        """
        origin = self.map_metadata["origin"]
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_metadata["resolution"] * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        pygame.draw.lines(
            self.map_canvas, color, closed=False, points=points, width=size
        )

    def render_closed_lines(
        self,
        points: Union[List, np.ndarray],
        color: Tuple[int, int, int] = (0, 0, 255),
        size: int = 1,
    ):
        """
        Render a sequence of lines segments.

        Args:
            points: sequence of xy points (N, 2)
            color: rgb color of the points
            size: size of the points in pixels
        """
        origin = self.map_metadata["origin"]
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_metadata["resolution"] * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        pygame.draw.lines(
            self.map_canvas, color, closed=True, points=points, width=size
        )

    def close(self):
        if self.render_mode == "human" or self.render_mode == "human_fast":
            pygame.quit()
