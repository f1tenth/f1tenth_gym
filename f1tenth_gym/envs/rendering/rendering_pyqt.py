from __future__ import annotations
import logging
import math
from typing import Any, Callable, Optional
import signal

import cv2
import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6 import QtGui
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PIL import ImageColor

from .pyqt_objects import *
from ..track import Track
from .renderer import EnvRenderer, ObjectRenderer, RenderSpec

# one-line instructions visualized at the top of the screen (if show_info=True)
INSTRUCTION_TEXT = "Mouse click (L/M/R): Change POV - 'S' key: On/Off"

# Replicated from pyqtgraphs' example utils for ci pipelines to pass
from time import perf_counter
class FrameCounter(QtCore.QObject):
    sigFpsUpdate = QtCore.pyqtSignal(object)

    def __init__(self, interval=1000):
        super().__init__()
        self.count = 0
        self.last_update = 0
        self.interval = interval

    def update(self):
        self.count += 1

        if self.last_update == 0:
            self.last_update = perf_counter()
            self.startTimer(self.interval)

    def timerEvent(self, evt):
        now = perf_counter()
        elapsed = now - self.last_update
        fps = self.count / elapsed
        self.last_update = now
        self.count = 0
        self.sigFpsUpdate.emit(fps)

class PyQtEnvRenderer(EnvRenderer):
    """
    Renderer of the environment using PyQtGraph.
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
            rendering mode in ["human", "human_fast", 'unlimited', "rgb_array"]
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
        
        # create the canvas
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window = pg.GraphicsLayoutWidget()
        self.window.setWindowTitle("F1Tenth Gym")
        self.window.setGeometry(
            0, 0, self.render_spec.window_size, self.render_spec.window_size
        )
        self.canvas: pg.PlotItem = self.window.addPlot()

        # Disable interactivity
        self.canvas.setMouseEnabled(x=True, y=True)  # Disable mouse panning & zooming
        self.canvas.hideButtons()  # Disable corner auto-scale button
        self.canvas.setMenuEnabled(False)  # Disable right-click context menu

        legend = self.canvas.addLegend()  # This doesn't disable legend interaction
        # Override both methods responsible for mouse events
        legend.mouseDragEvent = lambda *args, **kwargs: None
        legend.hoverEvent = lambda *args, **kwargs: None
        # self.scene() is a pyqtgraph.GraphicsScene.GraphicsScene.GraphicsScene
        self.window.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.window.keyPressEvent = self.key_pressed

        # Remove axes
        self.canvas.hideAxis("bottom")
        self.canvas.hideAxis("left")

        # setting plot window background color to yellow
        self.window.setBackground("w")

        # fps and time renderer
        self.clock = FrameCounter()
        self.fps_renderer = TextRenderer(parent=self.canvas, position="bottom_left")
        self.time_renderer = TextRenderer(parent=self.canvas, position="bottom_right")
        self.bottom_info_renderer = TextRenderer(
            parent=self.canvas, position="bottom_center"
        )
        self.top_info_renderer = TextRenderer(parent=self.canvas, position="top_center")

        if self.render_mode in ["human", "human_fast", 'unlimited']:
            self.clock.sigFpsUpdate.connect(
                lambda fps: self.fps_renderer.render(f"FPS: {fps:.1f}")
            )

        colors_rgb = [
            [rgb for rgb in ImageColor.getcolor(c, "RGB")]
            for c in render_spec.vehicle_palette
        ]
        self.car_colors = [
            colors_rgb[i % len(colors_rgb)] for i in range(len(self.agent_ids))
        ]

        width, height = render_spec.window_size, render_spec.window_size

        # map metadata
        self.map_origin = track.spec.origin
        self.map_resolution = track.spec.resolution

        # load map image
        original_img = track.occupancy_map

        # convert shape from (W, H) to (W, H, 3)
        track_map = np.stack([original_img, original_img, original_img], axis=-1)

        # rotate and flip to match the track orientation
        track_map = np.rot90(track_map, k=1)  # rotate clockwise
        track_map = np.flip(track_map, axis=0)  # flip vertically

        self.image_item = pg.ImageItem(track_map)
        # Example: Transformed display of ImageItem
        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        # Translate image by the origin of the map
        tr.translate(self.map_origin[0], self.map_origin[1])
        # Scale image by the resolution of the map
        tr.scale(self.map_resolution, self.map_resolution)
        self.image_item.setTransform(tr)
        self.canvas.addItem(self.image_item)

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

        if self.render_mode in ["human", "human_fast", 'unlimited']:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            self.window.show()
        elif self.render_mode == "rgb_array":
            self.exporter = ImageExporter(self.canvas)

    def update(self, obs: dict) -> None:
        """
        Update the simulation obs to be rendered.

        Parameters
        ----------
            obs: simulation obs as dictionary
        """
        if self.cars is None:
            self.cars = [
                CarRenderer(
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                    parent=self.canvas,
                )
                for ic in range(len(self.agent_ids))
            ]

        # update cars obs and zoom level (updating points-per-unit)
        for i, id in enumerate(self.agent_ids):
            self.cars[i].update(obs, id)

        # update time
        self.sim_time = obs[self.agent_ids[0]]["sim_time"]

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

    def key_pressed(self, event: QtGui.QKeyEvent) -> None:
        """
        Handle key press events.

        Parameters
        ----------
        event : QtGui.QKeyEvent
            key event
        """
        if event.key() == QtCore.Qt.Key.Key_S:
            logging.debug("Pressed S key -> Enable/disable rendering")
            self.draw_flag = not self.draw_flag
            # self.draw_flag_changed = True

    def mouse_clicked(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse click events.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            mouse event
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            logging.debug("Pressed left button -> Follow Next agent")

            self.follow_agent_flag = True
            if self.agent_to_follow is None:
                self.agent_to_follow = 0
            else:
                self.agent_to_follow = (self.agent_to_follow + 1) % len(self.agent_ids)

            self.active_map_renderer = "car"
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            logging.debug("Pressed right button -> Follow Previous agent")

            self.follow_agent_flag = True
            if self.agent_to_follow is None:
                self.agent_to_follow = 0
            else:
                self.agent_to_follow = (self.agent_to_follow - 1) % len(self.agent_ids)

            self.active_map_renderer = "car"
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            logging.debug("Pressed middle button -> Change to Map View")

            self.follow_agent_flag = False
            self.agent_to_follow = None

            self.active_map_renderer = "map"

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current state in a frame.
        It renders in the order: map, cars, callbacks, info text.

        Returns
        -------
        Optional[np.ndarray]
            if render_mode is "rgb_array", returns the rendered frame as an array
        """
        if self.draw_flag:
            # call callbacks
            for callback_fn in self.callbacks:
                callback_fn(self)
            
            # draw cars
            for i in range(len(self.agent_ids)):
                self.cars[i].render()

            if self.follow_agent_flag:
                ego_x, ego_y = self.cars[self.agent_to_follow].pose[:2]
                self.canvas.setXRange(ego_x - 10 / self.render_spec.zoom_in_factor, ego_x + 10 / self.render_spec.zoom_in_factor)
                self.canvas.setYRange(ego_y - 10 / self.render_spec.zoom_in_factor, ego_y + 10 / self.render_spec.zoom_in_factor)
            else:
                self.canvas.autoRange()
                
            agent_to_follow_id = (
                self.agent_ids[self.agent_to_follow]
                if self.agent_to_follow is not None
                else None
            )
            self.bottom_info_renderer.render(
                text=f"Focus on: {agent_to_follow_id}"
            )

            if self.render_spec.show_info:
                self.top_info_renderer.render(text=INSTRUCTION_TEXT)

            self.time_renderer.render(text=f"{self.sim_time:.2f}")
            self.clock.update()
            self.app.processEvents()

            if self.render_mode in ["human", "human_fast", 'unlimited']:
                assert self.window is not None

            else:  
                # rgb_array mode => extract the frame from the canvas
                qImage = self.exporter.export(toBytes=True)

                width = qImage.width()
                height = qImage.height()

                ptr = qImage.bits()
                ptr.setsize(height * width * 4)
                frame = np.array(ptr).reshape(height, width, 4)  #  Copies the data
                
                return frame[:, :, :3] # remove alpha channel
        else:
            self.clock.update()
            self.app.processEvents()

            # if draw_flag is False, we just return the current frame without rendering anything
            if self.render_mode == "rgb_array":
                qImage = self.exporter.export(toBytes=True)

                width = qImage.width()
                height = qImage.height()

                ptr = qImage.bits()
                ptr.setsize(height * width * 4)
                frame = np.array(ptr).reshape(height, width, 4)
                return frame[:, :, :3]  # remove alpha channel

    def get_points_renderer(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> ObjectRenderer:
        return PointsRenderer(self, points, color, size)

    def get_lines_renderer(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> ObjectRenderer:
        return LinesRenderer(self, points, color, size)

    def get_closed_lines_renderer(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ) -> ObjectRenderer:
        return ClosedLinesRenderer(self, points, color, size)

    def close(self) -> None:
        """
        Close the rendering environment.
        """
        self.app.exit()
        

        
