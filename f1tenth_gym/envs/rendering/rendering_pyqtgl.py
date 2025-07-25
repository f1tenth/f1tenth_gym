import signal
import time, logging
import os
import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Any, Optional, Union
from PIL import ImageColor
from PyQt6.QtGui import QImage
import OpenGL.GL as gl_module

from ..track import Track
from .renderer import EnvRenderer, RenderSpec, ObjectRenderer
from .pyqtgl_objects import PointsRenderer, LinesRenderer, ClosedLinesRenderer, CarRenderer
from .mesh_renderer import MeshRenderer
from PIL import Image, ImageDraw, ImageFont
import cv2

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
        self.camera_free_rotation = 0
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
        self.default_camera_dist = self.params['width'] * 70
        self.obs = None
        self.zoom_level = 1.0
        self.init = True
        
        fmt = QtGui.QSurfaceFormat()
        fmt.setSwapInterval(0)  # 0 = no vsync, 1 = vsync
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
        
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.view = gl.GLViewWidget() 

        self.view.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=self.default_camera_dist, elevation=90, azimuth=0)
        self.view.setBackgroundColor("w")
        self.view.resize(self.render_spec.window_size, self.render_spec.window_size) 
        self.prealloc_frame = np.zeros(
            (self.render_spec.window_size, self.render_spec.window_size, 3), dtype=np.uint8
        )

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("F1Tenth Gym - OpenGL")
        self.window.setGeometry(0, 0, self.render_spec.window_size, self.render_spec.window_size)
        
        if self.render_spec.car_model == "2d":
            if not self.camera_free_rotation: self._enable_pan_only()
            self.focused = True
        self._init_map(track)
        
        # FPS label
        text_rgb = (125, 125, 125)

        if self.render_spec.frame_output_info_label:
            self.lap_label = QtWidgets.QLabel(self.view)
            font = QtGui.QFont("Arial", 14)
            self.lap_label.setFont(font)
            self.lap_label.setStyleSheet(
                f"color: rgb({text_rgb[0]}, {text_rgb[1]}, {text_rgb[2]}); background-color: transparent; padding: 2px;"
            )
            self.lap_label.move(int(self.render_spec.window_size) - 220, 10)
            self.lap_label.resize(220, 30)
            self.lap_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.lap_label.show()
            
            self.fps_label = QtWidgets.QLabel(self.view)
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
        
        # Colors
        self.car_colors = [
            tuple(ImageColor.getcolor(c, "RGB")) for c in render_spec.vehicle_palette
        ]

        if self.render_mode in ["human", "human_fast", 'unlimited']:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            self.window.setCentralWidget(self.view)
            self.window.show()
        elif self.render_mode == "rgb_array":
            self.window.hide()

    def update_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        if self.cars is not None:
            for car in self.cars:
                car.update_params(params)

    def _apply_view_flip(self):
        """Apply a vertical flip transformation to the OpenGL view."""
        # Override the paintGL method to apply a Y-axis flip transformation
        # This will flip the view vertically so it matches the output frame orientation
        original_paintGL = self.view.paintGL
        def flipped_paintGL():
            gl_module.glPushMatrix()
            # Apply the flip transformation (flip Y-axis)
            gl_module.glScalef(-1.0, 1.0, 1.0)
            # Call the original paintGL
            original_paintGL()
            gl_module.glPopMatrix()
        self.view.paintGL = flipped_paintGL
        
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
        image_item = gl.GLImageItem(map_rgba, smooth=True)
        image_item.translate(px, py, -0.01)  # Slightly below the map
        image_item.scale(res, res, 1)
        image_item.setGLOptions('translucent') 
        if self.render_spec.render_map_img:
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
        if self.render_spec.bigger_car_when_map_centered:
            self.car_scale = extent/self.params['width'] / 120
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
                distance=self.default_camera_dist * self.zoom_level,  # zoom level
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
                self.focused = False
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                logging.debug("Pressed right button -> Follow Next agent")
                if self.agent_to_follow is None:
                    self.agent_to_follow = 0
                else:
                    self.agent_to_follow = (self.agent_to_follow + 1) % len(self.agent_ids)
                self._center_camera_on_car(self.agent_to_follow, distance_reset=True)
                self.focused = True
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
                
        def wheelEvent(event):
            delta = event.angleDelta().y()
            factor = 0.85 if delta > 0 else 1.15
            self.zoom_level *= factor
            event.accept()

        def mouseReleaseEvent(event):
            self.view.pan_active = False
            event.accept()

        self.view.mousePressEvent = mousePressEvent
        self.view.mouseMoveEvent = mouseMoveEvent
        self.view.mouseReleaseEvent = mouseReleaseEvent
        self.view.wheelEvent = wheelEvent

    def update(self, obs: dict) -> None:
        """
        Update the simulation obs to be rendered.

        Parameters
        ----------
            obs: simulation obs as dictionary
        """
        if self.cars is None:
            if self.render_spec.car_model == "3d":
                self.cars = [MeshRenderer(
                    env_renderer=self,
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                ) for ic in range(len(self.agent_ids))
                ]
            elif self.render_spec.car_model == "2d":
                self.cars = [CarRenderer(
                    env_renderer=self,
                    car_length=self.params["length"],
                    car_width=self.params["width"],
                    color=self.car_colors[ic],
                    render_spec=self.render_spec,
                    map_origin=self.map_origin[:2],
                    resolution=self.map_resolution,
                ) for ic in range(len(self.agent_ids))
                ]

        # update cars obs and zoom level (updating points-per-unit)
        for i, id in enumerate(self.agent_ids):
            self.cars[i].update(obs, id)

        # update time
        self.sim_time = obs[self.agent_ids[0]]["sim_time"]
        self.obs = obs
        

    def render(self):
        if self.draw_flag:
            if self.init:
                if self.render_mode != "rgb_array":
                    self.window.show()
                else:
                    self.font = ImageFont.truetype("arial.ttf", 20)
                self.init = False
            if self.obs is not None and self.render_spec.frame_output_info_label:
                self.lap_label.setText(f"Lap Time {self.obs[self.agent_ids[0]]['lap_time']:.2f}, " + 
                    f"Lap {int(self.obs[self.agent_ids[0]]['lap_count']):d}")
            start_time = time.time()
            
            # call callbacks
            for callback_fn in self.callbacks:
                callback_fn(self)
            if self.agent_to_follow is not None and \
                self.render_spec.car_model == "2d" and \
                not self.camera_free_rotation and \
                self.focused:
                    self._center_camera_on_car(self.agent_to_follow, distance_reset=True)
            # draw cars
            for i in range(len(self.agent_ids)):
                self.cars[i].render(self.car_scale)
            self.app.processEvents()
            
            self._update_fps()
            if self.render_fps < float('inf'):
                elapsed = time.time() - start_time
                sleep_time = max(0.0, 1/self.render_fps - elapsed)
                time.sleep(sleep_time)
                
            if self.render_mode == "rgb_array":
                # Use direct OpenGL framebuffer grab (current method)
                frame = self.grab_frame_as_rgb()
                return frame

    def grab_frame_as_rgb(self) -> np.ndarray:
        """
        Grab the current OpenGL frame buffer and overlay text labels as RGB numpy array.
        """
        # Make sure OpenGL context is active
        self.view.makeCurrent()
        qimg = self.view.grabFramebuffer()
        
        # Convert to RGB format for consistent pixel layout
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)

        # Extract raw bytes
        width, height = qimg.width(), qimg.height()
        ptr = qimg.bits()
        # Tell Python how many bytes to read (width*height*3 for RGB888)
        ptr.setsize(height * width * 3)  # 3 bytes per pixel for RGB888
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        
        img_array = img_array.copy()  # Ensure we have a contiguous copy
        # img_array = np.flip(img_array, axis=0)
        
        # Overlay text information using OpenCV
        if self.render_spec.frame_output_info_label:
            self._overlay_text_on_frame(img_array)

        return img_array

    def grab_frame_with_exporter(self) -> np.ndarray:
        """
        Grab the current frame by capturing the entire widget including Qt labels.
        Since ImageExporter doesn't work with GLViewWidget, we'll capture the widget directly.
        """
        # Make sure the widget is updated and everything is rendered
        
        self.app.processEvents()
        
        # Capture the entire view widget (this should include Qt labels if they're children)
        pixmap = self.view.grab()
        qimg = pixmap.toImage()
        
        # Convert to RGB format
        if qimg.format() != QImage.Format.Format_RGB888:
            qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        
        width = qimg.width()
        height = qimg.height()
        
        ptr = qimg.bits()
        ptr.setsize(height * width * 3)  # 3 bytes per pixel for RGB
        img_array = np.array(ptr).reshape(height, width, 3)
        
        # No need to flip since this captures the widget as displayed
        # (not the OpenGL framebuffer directly)
        
        # Ensure the array is contiguous in memory for video encoding
        # img_array = np.ascontiguousarray(img_array)
        

        return img_array

    def _overlay_text_on_frame(self, img_array: np.ndarray) -> None:
        """
        Overlay text information (FPS and lap data) onto the frame using TrueType fonts.
        """

        height, width = img_array.shape[:2]
        color = (140, 140, 140)
        if self.obs is not None:
            lap_time = self.obs[self.agent_ids[0]]['lap_time']
            lap_count = int(self.obs[self.agent_ids[0]]['lap_count'])
            lap_text = f"Lap Time: {lap_time:.2f}, Lap: {lap_count}"

            # Choose font and scale
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(lap_text, font, font_scale, thickness)
            x_pos = width - text_width - 10
            y_pos = 10 + text_height

            # Draw text directly on the numpy array (OpenCV uses BGR, so reverse color)
            cv2.putText(img_array, lap_text, (x_pos, y_pos), font, font_scale, color[::-1], thickness, cv2.LINE_AA)

        
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
        points: Union[list, np.ndarray],
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
        **kwargs
    ) -> ObjectRenderer:
        return PointsRenderer(self, points, color, size, **kwargs)

    def get_lines_renderer(
        self,
        points: Union[list, np.ndarray],
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1, 
        **kwargs
    ) -> ObjectRenderer:
        return LinesRenderer(self, points, color, size, **kwargs)

    def get_closed_lines_renderer(
        self,
        points: Union[list, np.ndarray],
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
        **kwargs
    ) -> ObjectRenderer:
        return ClosedLinesRenderer(self, points, color, size, **kwargs)

    def close(self):
        if self.render_mode != "rgb_array":
            self.window.close()
