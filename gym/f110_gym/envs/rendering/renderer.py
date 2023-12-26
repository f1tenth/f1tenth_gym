from __future__ import annotations
import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Any

import yaml


@dataclass
class RenderSpec:
    window_size: int
    zoom_in_factor: float
    focus_on: str
    car_tickness: int
    show_wheels: bool
    show_info: bool = True
    vehicle_palette: list[str] = None

    def __init__(
        self,
        window_size: int = 800,
        focus_on: str = None,
        zoom_in_factor: float = 1.5,
        car_tickness: int = 1,
        show_wheels: bool = False,
        show_info: bool = True,
        vehicle_palette: list[str] = None,
    ):
        self.window_size = window_size
        self.focus_on = focus_on
        self.zoom_in_factor = zoom_in_factor
        self.car_tickness = car_tickness
        self.show_wheels = show_wheels
        self.show_info = show_info
        self.vehicle_palette = vehicle_palette or ["#984ea3"]

    @staticmethod
    def from_yaml(yaml_file: Union[str, pathlib.Path]):
        with open(yaml_file, "r") as yaml_stream:
            try:
                config = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                print(ex)
        return RenderSpec(**config)


class EnvRenderer:
    render_callbacks = []

    @abstractmethod
    def update(self, state):
        """
        Update the state to be rendered.
        This is called at every rendering call.
        """
        raise NotImplementedError()

    def add_renderer_callback(self, callback_fn: callable):
        """
        Add a callback function to be called at every rendering call.
        This is called at the end of `update`.
        """
        self.render_callbacks.append(callback_fn)

    @abstractmethod
    def render_map(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        """
        Render the current state in a frame.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """
        Close the rendering window.
        """
        raise NotImplementedError()
