from __future__ import annotations
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem
from PyQt6 import QtGui

from .renderer import RenderSpec
from ..collision_models import get_vertices


class TextObject:
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
                return (-text_label.width(), -text_label.height())
            elif position == "bottom_left":
                return (0, -text_label.height())
            elif position == "bottom_center":
                return (-text_label.width()/2, -text_label.height())
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
        
    def render(self, text: str) -> None:
        """
        Render text on the screen.

        Parameters
        ----------
        text : str
            text to be displayed                 
        """
        self.text_label.setText(text)

class Car:
    """
    Class to display the car.
    """

    def __init__(
        self,
        render_spec: RenderSpec,
        map_origin: tuple[float, float],
        resolution: float,
        car_length: float,
        car_width: float,
        color: list[int] | None = None,
        wheel_size: float = 0.2,
        parent: pg.PlotWidget = None,
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
        self.chassis = None

        # Tire params need to be updated
        self.tire_width = 0.1
        self.tire_length = self.wheel_size

        vertices = get_vertices(self.pose, self.car_length, self.car_width)
        # vertices are rl, rr, fr, fl => Reorder to be rl, fl, fr, rr
        vertices = np.array([vertices[0], vertices[3], vertices[2], vertices[1]])
        # Append the first point to close the polygon
        vertices = np.vstack([vertices, vertices[0]])
        self.chassis : pg.PlotDataItem = parent.plot(vertices[:, 0], vertices[:, 1], pen=pg.mkPen(color=(0,0,0), width=self.car_thickness), fillLevel=0, brush=self.color)

    def update(self, state: dict[str, np.ndarray], idx: int):
        self.pose = (
            state["poses_x"][idx],
            state["poses_y"][idx],
            state["poses_theta"][idx],
        )
        self.color = (255, 0, 0) if state["collisions"][idx] > 0 else self.color
        self.steering = self.pose[2] + state["steering_angles"][idx]

    def render(self):
        vertices = get_vertices(self.pose, self.car_length, self.car_width)
        # vertices are rl, rr, fr, fl => Reorder to be rl, fl, fr, rr
        vertices = np.array([vertices[0], vertices[3], vertices[2], vertices[1]])
        # Append the first point to close the polygon
        vertices = np.vstack([vertices, vertices[0]])

        self.chassis.setData(vertices[:, 0], vertices[:, 1])



