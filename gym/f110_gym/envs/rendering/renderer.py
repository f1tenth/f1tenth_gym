from __future__ import annotations
import pathlib
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import yaml


@dataclass
class RenderSpec:
    window_size: int
    zoom_in_factor: float
    focus_on: str
    car_tickness: int
    show_wheels: bool
    show_info: Optional[bool] = True
    vehicle_palette: Optional[list[str]] = None

    def __init__(
        self,
        window_size: int = 800,
        focus_on: str = None,
        zoom_in_factor: float = 1.0,
        car_tickness: int = 1,
        show_wheels: bool = False,
        show_info: bool = True,
        vehicle_palette: list[str] = None,
    ) -> None:
        """
        Initialize rendering specification.

        Parameters
        ----------
        window_size : int, optional
            size of the square window, by default 800
        focus_on : str, optional
            focus on a specific vehicle, by default None
        zoom_in_factor : float, optional
            zoom in factor, by default 1.0 (no zoom)
        car_tickness : int, optional
            thickness of the car in pixels, by default 1
        show_wheels : bool, optional
            toggle rendering of line segments for wheels, by default False
        show_info : bool, optional
            toggle rendering of text instructions, by default True
        vehicle_palette : list, optional
            list of colors for rendering vehicles according to their id, by default None
        """
        self.window_size = window_size
        self.focus_on = focus_on
        self.zoom_in_factor = zoom_in_factor
        self.car_tickness = car_tickness
        self.show_wheels = show_wheels
        self.show_info = show_info
        self.vehicle_palette = vehicle_palette or ["#984ea3"]

    @staticmethod
    def from_yaml(yaml_file: str | pathlib.Path):
        """
        Load rendering specification from a yaml file.

        Parameters
        ----------
        yaml_file : str | pathlib.Path
            path to the yaml file

        Returns
        -------
        RenderSpec
            rendering specification object
        """
        with open(yaml_file, "r") as yaml_stream:
            try:
                config = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                print(ex)
        return RenderSpec(**config)


class EnvRenderer(ABC):
    """
    Abstract class for rendering the environment.
    """

    @abstractmethod
    def update(self, state: Any) -> None:
        """
        Update the state to be rendered.
        This is called at every rendering call.

        Parameters
        ----------
        state : Any
            state to be rendered, e.g. a list of vehicle states
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()

    @abstractmethod
    def render_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ):
        """
        Render a sequence of lines segments.

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : tuple[int, int, int], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : int, optional
            size of the line, by default 1
        """
        raise NotImplementedError()

    @abstractmethod
    def render_closed_lines(
        self,
        points: list | np.ndarray,
        color: Optional[tuple[int, int, int]] = (0, 0, 255),
        size: Optional[int] = 1,
    ):
        """
        Render a closed loop of lines (draw a line between the last and the first point).

        Parameters
        ----------
        points : list | np.ndarray
            list of points to render
        color : tuple[int, int, int], optional
            color as rgb tuple, by default blue (0, 0, 255)
        size : int, optional
            size of the line, by default 1
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """
        Close the rendering window.
        """
        raise NotImplementedError()
