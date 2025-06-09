import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtGui
from PyQt6 import QtCore
import numpy as np
from ..track import Track
from .renderer import EnvRenderer, RenderSpec
from PIL import ImageColor

class PyQtEnvRendererGL(EnvRenderer):
    def __init__(self, params, track, agent_ids, render_spec, render_mode, render_fps):
        super().__init__()
        self.params = params
        self.agent_ids = agent_ids
        self.render_spec = render_spec
        self.render_mode = render_mode
        self.render_fps = render_fps

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle("F1Tenth Gym - OpenGL")
        self.window.setGeometry(0, 0, self.render_spec.window_size, self.render_spec.window_size)
        self.window.setCameraPosition(distance=40)
        self.window.opts['center'] = QtGui.QVector3D(0, 0, 0)
        self.window.show()

        self.cars = []
        self.sim_time = None
        self.callbacks = []

        # Load and transform map
        self._init_map(track)
        
        # Colors
        self.car_colors = [
            tuple(ImageColor.getcolor(c, "RGB")) for c in render_spec.vehicle_palette
        ]

    def _init_map(self, track):
        img = track.occupancy_map
        img = np.stack([img]*3, axis=-1)
        img = np.rot90(img, k=1)
        img = np.flip(img, axis=0)

        # Normalize image for OpenGL
        img = img.astype(np.float32) / 255.0
        w, h = img.shape[:2]
        self.map_origin = track.spec.origin
        px, py = self.map_origin[0], self.map_origin[1]
        res = self.map_resolution = track.spec.resolution

        self.map_item = gl.GLImageItem(img)
        self.map_item.translate(px, py, 0)
        self.map_item.scale(res, res, 1)
        self.window.addItem(self.map_item)

    def update(self, state):
        if not self.cars:
            for i in range(len(self.agent_ids)):
                color = self.car_colors[i % len(self.car_colors)]
                car_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=color + (1.0,), size=5)
                self.window.addItem(car_item)
                self.cars.append(car_item)

        for i, car in enumerate(self.cars):
            x, y, _ = state["poses_x"][i], state["poses_y"][i], state["poses_theta"][i]
            car.setData(pos=np.array([[x, y, 0]]))

        self.sim_time = state["sim_time"]

    def render(self):
        for callback in self.callbacks:
            callback(self)
        self.app.processEvents()
        
    def add_renderer_callback(self, callback_fn):
        """
        Add a custom callback for visualization.

        Parameters
        ----------
        callback_fn : Callable[[EnvRenderer], None]
            callback function to be called at every rendering step
        """
        self.callbacks.append(callback_fn)

    def render_points(self, points, color=(0, 0, 255), size=3):
        color = [c / 255 for c in color] + [1.0]
        pts = np.hstack([points, np.zeros((len(points), 1))])
        item = gl.GLScatterPlotItem(pos=pts, color=color, size=size)
        self.window.addItem(item)
        return item

    def render_lines(self, points, color=(0, 0, 255), size=2):
        color = [c / 255 for c in color] + [1.0]
        pts = np.hstack([points, np.zeros((len(points), 1))])
        item = gl.GLLinePlotItem(pos=pts, color=color, width=size, mode='line_strip')
        self.window.addItem(item)
        return item

    def render_closed_lines(self, points, color=(0, 0, 255), size=2):
        return self.render_lines(np.vstack([points, points[0]]), color, size)

    def close(self):
        self.window.close()
