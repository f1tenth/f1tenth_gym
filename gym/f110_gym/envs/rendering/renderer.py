import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
import yaml
from PIL import Image


@dataclass
class RenderSpec:
    window_width: int
    window_height: int
    zoom_in_factor: float
    render_mode: str
    render_fps: int

    car_length = float
    car_width = float
    car_tickness = int

    def __init__(
        self,
        window_size: int = 1000,
        focus_on: str = "agent_0",
        zoom_in_factor: float = 1.2,
        render_fps: int = 30,
        car_length: float = 0.58,
        car_width: float = 0.31,
        car_tickness: int = 1,
    ):
        self.window_size = window_size
        self.focus_on = focus_on
        self.zoom_in_factor = zoom_in_factor
        self.render_fps = render_fps

        self.car_length = car_length
        self.car_width = car_width
        self.car_tickness = car_tickness

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
