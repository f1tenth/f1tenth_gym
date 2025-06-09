from __future__ import annotations
import numpy as np
# from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem
# from PyQt6 import QtGui
from pyqtgraph.opengl import GLLinePlotItem

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer
from ..collision_models import get_vertices, get_trmtx
from typing import Optional, Any

from numba import njit


class LinesRenderer(ObjectRenderer):
    def __init__(
        self, 
        env_renderer: EnvRenderer,
        points: list | np.ndarray, 
        color: Optional[tuple[int, int, int]] = (0, 0, 255), 
        size: Optional[int] = 1
        ):
        pen = pg.mkPen(color=pg.mkColor(*color), width=size)
        self.renderer = env_renderer.canvas.plot(
            points[:, 0], points[:, 1], pen=pen, fillLevel=None, antialias=True
        )
        
    def update(self, points: list | np.ndarray) -> None:
        self.renderer.updateItems(points)
        
class ClosedLinesRenderer(ObjectRenderer):
    def __init__(
        self, 
        env_renderer: EnvRenderer,
        points: list | np.ndarray, 
        color: Optional[tuple[int, int, int]] = (0, 0, 255), 
        size: Optional[int] = 1
        ):
        # Append the first point to the end to close the loop
        points = np.vstack([points, points[0]])
        pen = pg.mkPen(color=pg.mkColor(*color), width=size)
        pen.setCapStyle(pg.QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(pg.QtCore.Qt.PenJoinStyle.RoundJoin)
        self.renderer = env_renderer.canvas.plot(
            points[:, 0], points[:, 1], pen=pen, cosmetic=True, antialias=True
        ) ## setting pen=None disables line drawing
        
    def update(self, points: list | np.ndarray) -> None:
        self.renderer.updateItems(points)

class PointsRenderer(ObjectRenderer):
    def __init__(
        self, 
        env_renderer: EnvRenderer,
        points: list | np.ndarray, 
        color: Optional[tuple[int, int, int]] = (0, 0, 255), 
        size: Optional[int] = 1
        ):
        self.renderer = env_renderer.canvas.plot(
            points[:, 0],
            points[:, 1],
            pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=color, width=0),
            symbolBrush=pg.mkBrush(color=color, width=0),
            symbolSize=size,
        )
        
    def update(self, points: list | np.ndarray) -> None:
        self.renderer.setData(points)


class TextRenderer(ObjectRenderer):
    """
    Class to display text on the screen at a given position.

    Attributes
    ----------
    font : pygame.font.Font
        font object
    position : str | tuple
        position of the text on the screen
    text : pygame.Surface
        text surface to be displayed
    """
    def __init__(
        self,
        position: str | tuple,
        relative_font_size: int = 16,
        font_name: str = "Arial",
        parent: pg.PlotWidget = None,
    ) -> None:
        """
        Initialize text object.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen
        relative_font_size : int, optional
            font size relative to the window shape, by default 32
        font_name : str, optional
            font name, by default "Arial"
        """
        self.position = position
        self.text_label = pg.LabelItem("", parent=parent, size=str(relative_font_size) + 'pt', family=font_name, color=(125, 125, 125)) # create text label
        # Get the position and offset of the text
        position_tuple = self._position_resolver(self.position)
        offset_tuple = self._offset_resolver(self.position, self.text_label)
        # Set the position and offset of the text
        self.text_label.anchor(itemPos=position_tuple, parentPos=position_tuple, offset=offset_tuple)

    def _position_resolver(
        self, position: str | tuple[int, int]
    ) -> tuple[int, int]:
        """
        This function takes strings like "bottom center" and converts them into a location for the text to be displayed.
        If position is tuple, then passthrough.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen

        Returns
        -------
        tuple
            position of the text on the screen

        Raises
        ------
        ValueError
            if position is not a tuple or a string
        NotImplementedError
            if position is a string but not implemented
        """
        if isinstance(position, tuple) and len(position) == 2:
            return int(position[0]), int(position[1])
        elif isinstance(position, str):
            position = position.lower()
            if position == "bottom_right":
                return (1, 1)
            elif position == "bottom_left":
                return (0, 1)
            elif position == "bottom_center":
                return (0.5, 1)
            elif position == "top_right":
                return (1, 0)
            elif position == "top_left":
                return (0, 0)
            elif position == "top_center":
                return (0.5, 0)
            else:
                raise NotImplementedError(f"Position {position} not implemented.")
        else:
            raise ValueError(
                f"Position expected to be a tuple[int, int] or a string. Got {position}."
            )

    def _offset_resolver(
        self, position: str | tuple[int, int], text_label: pg.LabelItem
    ) -> tuple[int, int]:
        """
        This function takes strings like "bottom center" and converts them into a location for the text to be displayed.
        If position is tuple, then passthrough.

        Parameters
        ----------
        position : str | tuple
            position of the text on the screen

        Returns
        -------
        tuple
            position of the text on the screen

        Raises
        ------
        ValueError
            if position is not a tuple or a string
        NotImplementedError
            if position is a string but not implemented
        """
        if isinstance(position, tuple) and len(position) == 2:
            return int(position[0]), int(position[1])
        elif isinstance(position, str):
            position = position.lower()
            if position == "bottom_right":
                return (-text_label.width(), 0)
            elif position == "bottom_left":
                return (0, 0)
            elif position == "bottom_center":
                return (-text_label.width()/2, 0)
            elif position == "top_right":
                return (-text_label.width(), 0)
            elif position == "top_left":
                return (0, 0)
            elif position == "top_center":
                return (-text_label.width()/2, 0)
            else:
                raise NotImplementedError(f"Position {position} not implemented.")
        else:
            raise ValueError(
                f"Position expected to be a tuple[int, int] or a string. Got {position}."
            )
    
    def update(self) -> None:
        pass
        
    def render(self, text: str) -> None:
        """
        Render text on the screen.

        Parameters
        ----------
        text : str
            text to be displayed                 
        """
        self.text_label.setText(text)

class CarRendererGL:
    """
    Car renderer using PyQtGraph OpenGL backend.
    """
    def __init__(
        self,
        render_spec,
        map_origin: tuple[float, float],
        resolution: float,
        car_length: float,
        car_width: float,
        color: Optional[list[int]] = None,
        wheel_size: float = 0.2,
        parent=None,  # parent is GLViewWidget
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.wheel_size = wheel_size
        self.car_thickness = render_spec.car_tickness
        self.show_wheels = render_spec.show_wheels

        self.origin = map_origin
        self.resolution = resolution
        self.color = color or (0, 0, 0)
        self.pose = (0, 0, 0)
        self.steering = 0

        self.tire_width = 0.1
        self.tire_length = self.wheel_size

        # Construct chassis
        vertices = self._get_chassis_vertices(self.pose)
        self.chassis = GLLinePlotItem(
            pos=vertices,
            color=self._color_rgba(self.color),
            width=self.car_thickness,
            antialias=True,
            mode='line_strip'
        )
        parent.addItem(self.chassis)

        # Optional wheels
        if self.show_wheels:
            fl = self._get_tire_vertices(self.pose, 'fl', self.steering)
            self.fl_wheel = GLLinePlotItem(
                pos=fl,
                color=(0, 0, 0, 1),
                width=self.car_thickness,
                antialias=True,
                mode='line_strip'
            )
            parent.addItem(self.fl_wheel)

            fr = self._get_tire_vertices(self.pose, 'fr', self.steering)
            self.fr_wheel = GLLinePlotItem(
                pos=fr,
                color=(0, 0, 0, 1),
                width=self.car_thickness,
                antialias=True,
                mode='line_strip'
            )
            parent.addItem(self.fr_wheel)

    def update(self, obs: dict, agent_id: str):
        state = obs[agent_id]["std_state"].astype(float)
        self.pose = (state[0], state[1], state[4])
        if obs[agent_id]["collision"] > 0:
            self.color = (255, 0, 0)
        self.steering = state[2]

    def render(self):
        vertices = self._get_chassis_vertices(self.pose)
        self.chassis.setData(pos=vertices)

        if self.show_wheels:
            self.fl_wheel.setData(pos=self._get_tire_vertices(self.pose, 'fl', self.steering))
            self.fr_wheel.setData(pos=self._get_tire_vertices(self.pose, 'fr', self.steering))

    def _get_chassis_vertices(self, pose):
        verts = get_vertices(pose, self.car_length, self.car_width)
        verts = np.array([verts[0], verts[3], verts[2], verts[1], verts[0]])  # closed loop
        verts = np.hstack([verts, np.zeros((verts.shape[0], 1))])  # z=0
        return verts

    def _get_tire_vertices(self, pose, wheel_pos, steering):
        verts = _get_tire_vertices(
            pose, self.car_length, self.car_width,
            self.tire_width, self.tire_length,
            wheel_pos, steering
        )
        verts = np.vstack([verts, verts[0]])  # close loop
        verts = np.hstack([verts, np.zeros((verts.shape[0], 1))])
        return verts

    def _color_rgba(self, rgb):
        return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)
    



@njit(cache=True)
def _get_tire_vertices(pose, length, width, tire_width, tire_length, index, steering):
    """
    Utility function to return vertices of the car's tire given pose and size

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    pose_arr = np.array(pose)
    if index == 'fl':
        # Shift back, rotate
        H_shift = get_trmtx(np.array([-(length/2 - tire_length/2), -(width/2 - tire_width/2), 0]))
        H_steer = get_trmtx(np.array([0, 0, steering]))
        H_back = get_trmtx(np.array([length/2 - tire_length/2, width/2 - tire_width/2, 0]))
        H = get_trmtx(pose_arr)
        H = H.dot(H_back).dot(H_steer).dot(H_shift)
        fl = H.dot(np.asarray([[length / 2], [width / 2], [0.0], [1.0]])).flatten()
        fr = H.dot(np.asarray([[length / 2], [width / 2 - tire_width], [0.0], [1.0]])).flatten()
        rr = H.dot(np.asarray([[length / 2 - tire_length], [width / 2 - tire_width], [0.0], [1.0]])).flatten()
        rl = H.dot(np.asarray([[length / 2 - tire_length], [width / 2], [0.0], [1.0]])).flatten()
        rl = rl / rl[3]
        rr = rr / rr[3]
        fl = fl / fl[3]
        fr = fr / fr[3]
        vertices = np.asarray(
            [[rl[0], rl[1]], [fl[0], fl[1]], [fr[0], fr[1]], [rr[0], rr[1]], [rl[0], rl[1]]]
        )
    elif index == 'fr':
        # Shift back, rotate
        H_shift = get_trmtx(np.array([-(length/2 - tire_length/2), -(-width/2 + tire_width/2), 0]))
        H_steer = get_trmtx(np.array([0, 0, steering]))
        H_back = get_trmtx(np.array([length/2 - tire_length/2, -width/2 + tire_width/2, 0]))
        H = get_trmtx(pose_arr)
        H = H.dot(H_back).dot(H_steer).dot(H_shift)

        fl = H.dot(np.asarray([[length / 2], [-width / 2 + tire_width], [0.0], [1.0]])).flatten()
        fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.0], [1.0]])).flatten()
        rr = H.dot(np.asarray([[length / 2 - tire_length], [-width / 2], [0.0], [1.0]])).flatten()
        rl = H.dot(np.asarray([[length / 2 - tire_length], [-width / 2 + tire_width], [0.0], [1.0]])).flatten()
        rl = rl / rl[3]
        rr = rr / rr[3]
        fl = fl / fl[3]
        fr = fr / fr[3]
        # As it is only used for rendering, we can reorder the vertices and append the first point to close the polygon
        vertices = np.asarray(
            [[rl[0], rl[1]], [fl[0], fl[1]], [fr[0], fr[1]], [rr[0], rr[1]], [rl[0], rl[1]]]
        )

    return vertices