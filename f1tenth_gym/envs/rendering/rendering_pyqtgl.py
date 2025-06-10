import time, logging
import pyqtgraph.opengl as gl
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
from ..track import Track
from .renderer import EnvRenderer, RenderSpec, ObjectRenderer
from PIL import ImageColor
from typing import Any, Callable, Optional
from .pyqtgl_objects import PointsRenderer, LinesRenderer, ClosedLinesRenderer, CarRenderer

class PyQtEnvRendererGL(EnvRenderer):
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
        self.agent_ids = agent_ids
        self.render_spec = render_spec
        self.render_mode = render_mode
        self.render_fps = render_fps
        if render_spec.focus_on:
            self.agent_to_follow_setting = self.agent_ids.index(render_spec.focus_on)
            self.agent_to_follow = self.agent_ids.index(render_spec.focus_on)
        else:
            self.agent_to_follow = None
        self.car_scale = 1.0
        
        fmt = QtGui.QSurfaceFormat()
        fmt.setSwapInterval(0)  # 0 = no vsync, 1 = vsync
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=20, elevation=90, azimuth=0)
        self.view.setBackgroundColor((25, 25, 25))
        self.window = QtWidgets.QMainWindow()
        self.window.setCentralWidget(self.view)
        self.window.setWindowTitle("F1Tenth Gym - OpenGL")
        self.window.setGeometry(0, 0, self.render_spec.window_size, self.render_spec.window_size)
        
        # self._enable_pan_only()
        self._init_map(track)
        
        # FPS label
        if self.render_mode in ["human", "human_fast"]:
            text_rgb = (140, 140, 140)
            self.fps_label = QtWidgets.QLabel(self.window)
            font = QtGui.QFont("Arial", 14)
            self.fps_label.setFont(font)
            self.fps_label.setStyleSheet(
                f"color: rgb({text_rgb[0]}, {text_rgb[1]}, {text_rgb[2]}); background-color: transparent; padding: 2px;"
            )
            self.fps_label.move(10, 10)
            self.fps_label.resize(100, 20)
            self.fps_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.fps_label.show()

        # Frame timer
        self.last_time = time.time()
        self.frame_count = 0

        self.cars = None
        self.sim_time = None
        self.callbacks = []
        self.draw_flag = True
        self.window.show()
        
        # Colors
        self.car_colors = [
            tuple(ImageColor.getcolor(c, "RGB")) for c in render_spec.vehicle_palette
        ]
        
    def _init_map(self, track):
        map_image = track.occupancy_map
        map_image = np.rot90(map_image, k=1)
        map_image = np.flip(map_image, axis=0)
        self.map_image = map_image

        # Normalize image for OpenGL
        self.map_origin = track.spec.origin
        px, py = self.map_origin[0], self.map_origin[1]
        res = self.map_resolution = track.spec.resolution
        
        map_rgb = np.stack([map_image]*3, axis=-1)
        alpha = np.ones((map_rgb.shape[0], map_rgb.shape[1], 1), dtype=np.uint8) * 255
        map_rgba = np.concatenate((map_rgb, alpha), axis=-1)
        image_item = gl.GLImageItem(map_rgba)
        image_item.translate(px, py, -0.01)  # Slightly below the map
        image_item.scale(res, res, 1)
        image_item.setGLOptions('translucent') 
        self.view.addItem(image_item)
        
    def _get_map_bounds(self):
        h, w = self.map_image.shape[:2]
        sx, sy = self.map_resolution, self.map_resolution
        ox, oy = self.map_origin[0], self.map_origin[1]
        min_xy = np.array([ox, oy])
        max_xy = np.array([ox + w * sx, oy + h * sy])
        return min_xy, max_xy
        
    def _center_camera_on_map(self):
        min_xy, max_xy = self._get_map_bounds()
        # Compute center and extent
        center = (min_xy + max_xy) / 2
        extent = max(max_xy - min_xy)
        # if self.config
        self.car_scale = extent/self.params['width'] / 100
        # Fixed height above map
        x, y = center
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(x, y, 1),             # camera position
            distance=extent * 0.8,  # zoom level
            elevation=90,                              # top-down
            azimuth=0                                  # no rotation
        )
    
    def _center_camera_on_car(self, car_idx=0, distance_reset=False):
        x, y = self.cars[car_idx].pose[:2]  # Get car position
        self.car_scale = 1.0
        if distance_reset:
            self.view.setCameraPosition(
                distance=self.params['width'] * 50,  # zoom level
            )
        self.view.setCameraPosition(
            pos=QtGui.QVector3D(x, y, 1),             # camera position
            elevation=90,                              # top-down
            azimuth=0                                  # no rotation
        )
        
    def _enable_pan_only(self):
        """Override GLViewWidget events to disable rotation and allow right-click panning."""
        self.view.pan_active = False
        self.view.pan_start = QtCore.QPoint()

        def mousePressEvent(event):
            if event.button() == QtCore.Qt.MouseButton.LeftButton: # NOTE: left button is used for panning
                self.view.pan_active = True
                self.view.pan_start = event.pos()
                event.accept()
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                logging.debug("Pressed right button -> Follow Next agent")
                if self.agent_to_follow is None:
                    self.agent_to_follow = 0
                else:
                    self.agent_to_follow = (self.agent_to_follow + 1) % len(self.agent_ids)
                self._center_camera_on_car(self.agent_to_follow, distance_reset=True)
            elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
                logging.debug("Pressed middle button -> Change to Map View")
                self._center_camera_on_map()
                self.agent_to_follow = None

        def mouseMoveEvent(event):
            if self.view.pan_active:
                delta = event.pos() - self.view.pan_start
                dx = -delta.y() * 0.08
                dy = -delta.x() * 0.08
                self.view.pan(dx, dy, 0)
                self.view.pan_start = event.pos()
                event.accept()
            else:
                event.ignore()

        def mouseReleaseEvent(event):
            self.view.pan_active = False
            event.accept()

        self.view.mousePressEvent = mousePressEvent
        self.view.mouseMoveEvent = mouseMoveEvent
        self.view.mouseReleaseEvent = mouseReleaseEvent

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
                    env_renderer=self,
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                )
                for ic in range(len(self.agent_ids))
            ]

        # update cars obs and zoom level (updating points-per-unit)
        for i, id in enumerate(self.agent_ids):
            self.cars[i].update(obs, id)

        # update time
        self.sim_time = obs[self.agent_ids[0]]["sim_time"]

    def render(self):
        if self.draw_flag:
            start_time = time.time()
            
            # call callbacks
            for callback_fn in self.callbacks:
                callback_fn(self)
            # if self.agent_to_follow is not None:
            #     self._center_camera_on_car(self.agent_to_follow)
            # draw cars
            for i in range(len(self.agent_ids)):
                self.cars[i].render(self.car_scale)
            
            self.app.processEvents()
            
            if self.render_mode in ["human", "human_fast"]:
                self._update_fps()
                # elapsed = time.time() - start_time
                # sleep_time = max(0.0, 1/self.render_fps - elapsed)
                # time.sleep(sleep_time)
        
    def add_renderer_callback(self, callback_fn):
        """
        Add a custom callback for visualization.

        Parameters
        ----------
        callback_fn : Callable[[EnvRenderer], None]
            callback function to be called at every rendering step
        """
        self.callbacks.append(callback_fn)
    
    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time

        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.0f}")
            self.last_time = now
            self.frame_count = 0
        self.view.update()

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

    def close(self):
        self.window.close()
