from __future__ import annotations
import logging
import math
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pygame
from PIL import ImageColor

from .objects import (
    Map,
    Car,
    TextObject,
)
from ..track import Track
from .renderer import EnvRenderer, RenderSpec

# one-line instructions visualized at the top of the screen (if show_info=True)
INSTRUCTION_TEXT = "Mouse click (L/M/R): Change POV - 'S' key: On/Off"


class PygameEnvRenderer(EnvRenderer):
    """
    Renderer of the environment using Pygame.
    """

    def __init__(
        self,
        params: dict[str, Any],
        track: Track,
        agent_ids: list[str],
        render_spec: RenderSpec,
        render_mode: str,
        render_fps: int,
    ):
        """
        Initialize the Pygame renderer.

        Parameters
        ----------
        params : dict
            dictionary of simulation parameters (including vehicle dimensions, etc.)
        track : Track
            track object
        agent_ids : list
            list of agent ids to render
        render_spec : RenderSpec
            rendering specification
        render_mode : str
            rendering mode in ["human", "human_fast", "rgb_array"]
        render_fps : int
            number of frames per second            
        """
        super().__init__()
        self.params = params
        self.agent_ids = agent_ids

        self.cars = None
        self.sim_time = None
        self.window = None
        self.canvas = None

        self.render_spec = render_spec
        self.render_mode = render_mode
        self.render_fps = render_fps

        colors_rgb = [
            [rgb for rgb in ImageColor.getcolor(c, "RGB")]
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
        self.map_canvas = None

        # map metadata
        self.map_origin = track.spec.origin
        self.map_resolution = track.spec.resolution

        # fps and time renderer
        self.clock = pygame.time.Clock()
        self.fps_renderer = TextObject(
            window_shape=(width, height), position="bottom_left"
        )
        self.time_renderer = TextObject(
            window_shape=(width, height), position="bottom_right"
        )
        self.bottom_info_renderer = TextObject(
            window_shape=(width, height), position="bottom_center"
        )
        self.top_info_renderer = TextObject(
            window_shape=(width, height), position="top_center"
        )

        # load map image
        original_img = track.occupancy_map

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

        # callbacks for custom visualization, called at every rendering step
        self.callbacks = []

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

    def update(self, state: dict) -> None:
        """
        Update the simulation state to be rendered.

        Parameters
        ----------
            state: simulation state as dictionary
        """
        if self.cars is None:
            self.cars = [
                Car(
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                    ppu=self.ppus[self.active_map_renderer],
                )
                for ic in range(len(self.agent_ids))
            ]

        # update cars state and zoom level (updating points-per-unit)
        for i in range(len(self.agent_ids)):
            self.cars[i].update(state, i)
            self.cars[i].ppu = self.ppus[self.active_map_renderer]

        # update time
        self.sim_time = state["sim_time"]

    def add_renderer_callback(self, callback_fn: Callable[[EnvRenderer], None]) -> None:
        """
        Add a custom callback for visualization.
        All the callbacks are called at every rendering step, after having rendered the map and the cars.

        Parameters
        ----------
        callback_fn : Callable[[EnvRenderer], None]
            callback function to be called at every rendering step
        """
        self.callbacks.append(callback_fn)

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state in a frame.
        It renders in the order: map, cars, callbacks, info text.

        Returns
        -------
        Optional[np.ndarray]
            if render_mode is "rgb_array", returns the rendered frame as an array
        """
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
                origin = self.map_origin
                resolution = self.map_resolution * self.ppus[self.active_map_renderer]
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
                text=f"Focus on: {agent_to_follow_id}", display=self.canvas
            )

        if self.render_spec.show_info:
            self.top_info_renderer.render(text=INSTRUCTION_TEXT, display=self.canvas)
        self.time_renderer.render(text=f"{self.sim_time:.2f}", display=self.canvas)

        if self.render_mode in ["human", "human_fast"]:
            assert self.window is not None

            self.fps_renderer.render(
                text=f"FPS: {self.clock.get_fps():.2f}", display=self.canvas
            )

            self.window.blit(self.canvas, self.canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
            if frame.shape[0] > 2000:
                frame = cv2.resize(
                    frame, dsize=(2000, 2000), interpolation=cv2.INTER_AREA
                )
            return frame

    def event_handling(self) -> None:
        """
        Handle interaction events to change point-of-view.

        Events:
            - Left mouse button: follow next agent (according to agent_ids order)
            - Right mouse button: follow previous agent
            - Middle mouse button: change to map view
            - S key: enable/disable rendering
        """
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
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> None:
        """
        Render a sequence of xy points on screen.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the points in pixels, by default 1
        """
        origin = self.map_origin
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_resolution * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        for point in points:
            pygame.draw.circle(self.map_canvas, color, point, size)

    def render_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> None:
        """
        Render a sequence of lines segments.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the line, by default 1
        """
        origin = self.map_origin
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_resolution * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        pygame.draw.lines(
            self.map_canvas, color, closed=False, points=points, width=size
        )

    def render_closed_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> None:
        """
        Render a sequence of lines segments forming a closed loop (draw a line between the last and the first point).

        Parameters
        ----------
        points : list | np.ndarray
            list of 2d points to render
        color : Optional[tuple[int, int, int]], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : Optional[int], optional
            size of the line, by default 1
        """
        origin = self.map_origin
        ppu = self.ppus[self.active_map_renderer]
        resolution = self.map_resolution * ppu
        points = ((points - origin[:2]) / resolution).astype(int)
        size = math.ceil(size / ppu)

        pygame.draw.lines(
            self.map_canvas, color, closed=True, points=points, width=size
        )

    def close(self) -> None:
        """
        Close the rendering environment.
        """
        if self.render_mode in ["human", "human_fast"]:
            pygame.quit()
