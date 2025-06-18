from __future__ import annotations
import numpy as np
import pyqtgraph.opengl as gl

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer
from typing import Optional, Any


class CarRenderer(ObjectRenderer):
    """
    Class to display the car.
    """
    def __init__(
        self,
        env_renderer: EnvRenderer,
        render_spec: RenderSpec,
        map_origin: tuple[float, float],
        resolution: float,
        car_length: float,
        car_width: float,
        color: list[int] | None = None,
        wheel_size: float = 0.2
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.wheel_size = wheel_size
        self.car_thickness = render_spec.car_tickness
        self.show_wheels = render_spec.show_wheels
        self.rgba = [c / 255 for c in color] + [1.0]
        self.rgba = np.array([self.rgba] * 2)
        self.scale = 1.0
        
        # Define centered rectangle in local coords
        hl = self.car_length / 2 # half-length
        hw = self.car_width / 2 # half-width
        self.base_rect = np.array([
            [-hl, -hw, 0],
            [ hl, -hw, 0],
            [ hl,  hw, 0],
            [-hl,  hw, 0],
        ], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.mesh = gl.GLMeshItem(
            vertexes=self.base_rect,
            faces=self.faces,
            faceColors=self.rgba,
            smooth=False,
            drawEdges=False,
            edgeColor=(0, 0, 0, 1)
        )
        env_renderer.view.addItem(self.mesh)

        self.origin = map_origin
        self.resolution = resolution

        self.color = color
        self.pose = (0, 0, 0)
        self.steering = 0

        # Tire params need to be updated
        self.tire_width = 0.1
        self.tire_length = self.wheel_size
        
    
    def apply_pose(self, pose, scale=1.0):
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        return ((self.base_rect * scale) @ R.T) + np.array([x, y, 0.01])
        
    def update(self, obs: dict[str, np.ndarray], id: str):        
        state = obs[id]["std_state"].astype(float)
        self.pose = (
            state[0],
            state[1],
            state[4],
        )
        if obs[id]["collision"] > 0:
            color = (255, 0, 0)
            self.rgba = [c / 255 for c in color] + [1.0]
            self.rgba = np.array([self.rgba] * 2)
        self.steering = state[2]
        
    def render(self, scale=1.0):
        # pass
        if scale != 1.0:
            transformed = self.apply_pose(self.pose, scale)
            self.mesh.resetTransform()
            self.mesh.setMeshData(vertexes=transformed, 
                                faces=self.faces,
                                faceColors=self.rgba,
                                smooth=False,
                                drawEdges=False,)
        else:
            self.mesh.resetTransform()
            self.mesh.rotate(self.pose[2] / np.pi * 180, 0, 0, 1)
            self.mesh.translate(self.pose[0], self.pose[1], 0.01)

class LinesRenderer(ObjectRenderer):
    def __init__(
        self, 
        env_renderer: EnvRenderer,
        points: list | np.ndarray, 
        color: Optional[tuple[int, int, int]] = (0, 0, 255), 
        size: Optional[int] = 1
        ):
        # Convert to 3D (z=0)
        self.points3d = np.hstack([points, np.zeros((points.shape[0], 1))])

        # Normalize color
        rgba = tuple([c / 255 for c in color] + [1.0])

        # Create the OpenGL line loop
        self.line = gl.GLLinePlotItem(
            pos=self.points3d,
            color=rgba,
            width=size,
            mode='line_strip',
            antialias=False,
        )
        self.line.setGLOptions('translucent')
        env_renderer.view.addItem(self.line)

    def update(self, points: np.ndarray):
        self.points3d = np.hstack([points, np.zeros((points.shape[0], 1))])
        self.line.setData(pos=self.points3d)
        
class ClosedLinesRenderer(ObjectRenderer):
    def __init__(
        self,
        env_renderer: EnvRenderer,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        size: float = 2.0
    ):
        # Ensure loop is closed
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        # Convert to 3D (z=0)
        self.points3d = np.hstack([points, np.ones((points.shape[0], 1)) * 0.01])

        # Normalize color
        rgba = tuple([c / 255 for c in color] + [1.0])

        # Create the OpenGL line loop
        self.line = gl.GLLinePlotItem(
            pos=self.points3d,
            color=rgba,
            width=size,
            mode='line_strip',
            antialias=False,
        )
        self.line.setGLOptions('translucent')
        env_renderer.view.addItem(self.line)

    def update(self, points: np.ndarray):
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        self.points3d = np.hstack([points, np.ones((points.shape[0], 1)) * 0.01])
        self.line.setData(pos=self.points3d)

class PointsRenderer(ObjectRenderer):
    def __init__(
        self,
        env_renderer: EnvRenderer,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        size: int = 5
    ):
        # Normalize color to (0–1)
        color_rgba = tuple([c / 255 for c in color] + [1.0])

        # Convert to 3D
        self.points = np.hstack([points, np.ones((points.shape[0], 1)) * 0.01])  # z = 0
        self.scatter = gl.GLScatterPlotItem(
            pos=self.points,
            color=color_rgba,
            size=size,
            pxMode=True,  # Use pixel-based sizing
        )
        self.scatter.setGLOptions('translucent')
        env_renderer.view.addItem(self.scatter)

    def update(self, points: np.ndarray):
        self.points = np.hstack([points, np.ones((points.shape[0], 1)) * 0.01])
        self.scatter.setData(pos=self.points)
