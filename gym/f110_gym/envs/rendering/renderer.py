import pathlib
from abc import abstractmethod
from dataclasses import dataclass

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

    def __init__(self, window_width=1000, window_height=800, zoom_in_factor=1.2, render_fps=30,
                 car_length=0.58, car_width=0.31, render_mode="human"):
        self.window_width = window_width
        self.window_height = window_height
        self.zoom_in_factor = zoom_in_factor
        self.render_fps = render_fps

        self.car_length = car_length
        self.car_width = car_width

        self.render_mode = render_mode


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



