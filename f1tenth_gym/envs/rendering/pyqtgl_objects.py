from __future__ import annotations
import numpy as np
import time
import pyqtgraph.opengl as gl

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer
from typing import Optional, Any, Union


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
        color: Optional[list[int]] = None,
        wheel_size: float = 0.2
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.wheel_size = wheel_size
        self.car_thickness = render_spec.car_tickness
        self.show_wheels = render_spec.show_wheels
        self.opacity = 1.0
        self.rgba = [c / 255 for c in color] + [self.opacity]
        self.rgba = np.array([self.rgba] * 2)
        self.scale = 1.0
        self.z_offset = 0.3  # Offset for rendering above the ground
        
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
        self.env_renderer = env_renderer
        self.env_renderer.view.addItem(self.mesh)

        self.origin = map_origin
        self.resolution = resolution

        self.color = color
        self.pose = (0, 0, 0)
        self.steering = 0

        # Tire params need to be updated
        self.tire_width = 0.1
        self.tire_length = self.wheel_size
        
    def update_params(self, params: dict[str, Any]) -> None:
        """
        Update the parameters of the car renderer.
        """
        if "car_length" in params:
            self.car_length = params["car_length"]
        if "car_width" in params:
            self.car_width = params["car_width"]
        if "wheel_size" in params:
            self.wheel_size = params["wheel_size"]
        
        # Recalculate base rectangle
        hl = self.car_length / 2
        hw = self.car_width / 2
        self.base_rect = np.array([
            [-hl, -hw, 0],
            [ hl, -hw, 0],
            [ hl,  hw, 0],
            [-hl,  hw, 0],
        ], dtype=np.float32)

        self.env_renderer.view.removeItem(self.mesh)
        self.mesh = gl.GLMeshItem(
            vertexes=self.base_rect,
            faces=self.faces,
            faceColors=self.rgba,
            smooth=False,
            drawEdges=False,
            edgeColor=(0, 0, 0, 1)
        )
        self.env_renderer.view.addItem(self.mesh)

    def apply_pose(self, pose, scale=1.0):
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        return ((self.base_rect * scale) @ R.T) + np.array([x, y, self.z_offset])
        
    def update(self, obs: dict[str, np.ndarray], id: str):        
        state = obs[id]["std_state"].astype(float)
        self.pose = (
            state[0],
            state[1],
            state[4],
        )
        if obs[id]["collision"] > 0:
            color = (255, 0, 0)
            self.rgba = [c / 255 for c in color] + [self.opacity]
            self.rgba = np.array([self.rgba] * 2)
        self.steering = state[2]
        
    def render(self, scale=1.0):
        if self.scale != scale:
            # transformed = self.apply_pose(self.pose, 1.0)
            self.mesh.resetTransform()
            self.mesh.setMeshData(vertexes=self.base_rect * scale, 
                                faces=self.faces,
                                faceColors=self.rgba,
                                smooth=False,
                                drawEdges=False,)
            self.mesh.rotate(self.pose[2] / np.pi * 180, 0, 0, 1)
            self.mesh.translate(self.pose[0], self.pose[1], self.z_offset)
            self.scale = scale
        else:
            self.mesh.resetTransform()
            self.mesh.rotate(self.pose[2] / np.pi * 180, 0, 0, 1)
            self.mesh.translate(self.pose[0], self.pose[1], self.z_offset)
            
            
class LinesRenderer(ObjectRenderer):
    def __init__(
        self, 
        env_renderer: EnvRenderer,
        points: Union[list, np.ndarray], 
        color: Optional[tuple[int, int, int]] = (0, 0, 255), 
        size: Optional[int] = 1,
        z_offset: float = 0.02
        ):
        # Convert to 3D (z=0)
        self.z_offset = z_offset
        self.points3d = np.hstack([points, np.ones((points.shape[0], 1)) * z_offset])

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
        self.points3d = np.hstack([points, np.ones((points.shape[0], 1)) * self.z_offset])
        self.line.setData(pos=self.points3d)
        
class ClosedLinesRenderer(ObjectRenderer):
    def __init__(
        self,
        env_renderer: EnvRenderer,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        size: float = 2.0,
        z_offset: float = 0.01
    ):
        self.z_offset = z_offset
        # Ensure loop is closed
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        # Convert to 3D (z=0)
        points3d = np.hstack([points, np.ones((points.shape[0], 1)) * z_offset])

        # Normalize color
        rgba = tuple([c / 255 for c in color] + [1.0])

        # Create the OpenGL line loop
        self.line = gl.GLLinePlotItem(
            pos=points3d,
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
        if points.shape[1] == 2:
            points = np.hstack([points, np.ones((points.shape[0], 1)) * self.z_offset])
        else:
            points = points
        self.line.setData(pos=points)

class PointsRenderer(ObjectRenderer):
    def __init__(
        self,
        env_renderer: EnvRenderer,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        size: int = 5,
        z_offset: float = 0.03
    ):
        self.z_offset = z_offset
        # Normalize color to (0–1)
        color_rgba = tuple([c / 255 for c in color] + [1.0])

        # Convert to 3D
        if points.shape[1] == 2:
            points = np.hstack([points, np.ones((points.shape[0], 1)) * self.z_offset])
        else:
            points = points
        self.points_shape = points.shape
        self.scatter = gl.GLScatterPlotItem(
            pos=points,
            color=color_rgba,
            size=size,
            pxMode=True,  # Use pixel-based sizing
        )
        self.scatter.setGLOptions('translucent')
        env_renderer.view.addItem(self.scatter)

    def update(self, points: np.ndarray):
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] == 2:
            points = np.hstack([points, np.ones((points.shape[0], 1)) * self.z_offset])
        else:
            points = points
        if points.shape != self.points_shape:
            self.scatter.setData(pos=points)
            self.points_shape = points.shape
        else:
            self.scatter.pos[:] = points
            self.scatter.update()