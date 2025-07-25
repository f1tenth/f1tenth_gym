from __future__ import annotations
import pathlib
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import yaml

class ConfigYAML:
    """
    Config class for yaml file
    Able to load and save yaml file to and from python object
    """
    def __init__(self) -> None:
        pass
    
    def load_file(self, filename):
        self.d = yaml.safe_load(pathlib.Path(filename).read_text())
        for key in self.d: 
            setattr(self, key, self.d[key]) 
    
    def from_yaml(self, filename):
        self.d = yaml.safe_load(pathlib.Path(filename).read_text())
        for key in self.d: 
            setattr(self, key, self.d[key]) 
    
    def save_file(self, filename):
        d = vars(self)
        class_d = vars(self.__class__)
        d_out = {}
        for key in list(class_d.keys()):
            if not (key.startswith('__') or \
                    key.startswith('load_file') or \
                    key.startswith('save_file')):
                if isinstance(class_d[key], np.ndarray):
                    d_out[key] = class_d[key].tolist()
                else:
                    d_out[key] = class_d[key]
        for key in list(d.keys()):
            if not (key.startswith('__') or \
                    key.startswith('load_file') or \
                    key.startswith('save_file')):
                if isinstance(d[key], np.ndarray):
                    d_out[key] = d[key].tolist()
                else:
                    d_out[key] = d[key]
        with open(filename, 'w+') as ff:
            yaml.dump_all([d_out], ff)


@dataclass
class RenderSpec(ConfigYAML):
    def __init__(
        self,
        window_size: int = 800,
        focus_on: str = None,
        zoom_in_factor: float = 1.0,
        car_tickness: int = 1,
        show_wheels: bool = False,
        show_info: bool = True,
        vehicle_palette: list[str] = None,
        render_type: str = "pygame",
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
        self.render_type = render_type

class EnvRenderer(ABC):
    """
    Abstract class for rendering the environment.
    """
    @abstractmethod
    def add_renderer_callback(self, callback_fn):
        """
        Add a custom callback for visualization.

        Parameters
        ----------
        callback_fn : Callable[[EnvRenderer], None]
            callback function to be called at every rendering step
        """
        raise NotImplementedError()
    
    @abstractmethod
    def update(self, obs: dict) -> None:
        """
        Update the state to be rendered.
        This is called at every rendering call.

        Parameters
        ----------
        obs : dict
            observations from the env to be rendered
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

class ObjectRenderer(ABC):
    
    @abstractmethod
    def __init__(self):
        """
        Initialize the point renderer.
        This should set up the necessary parameters for rendering points.
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Update the point renderer with new data.
        This is called at every rendering call.
        """
        raise NotImplementedError()