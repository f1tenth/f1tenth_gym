import pathlib
import os
from typing import Any, Optional

from .renderer import RenderSpec, EnvRenderer
# from ..track import Track This is due to a circular import


def make_renderer(
    params: dict[str, Any],
    track: "Track",
    agent_ids: list[str],
    render_mode: Optional[str] = None,
    render_fps: Optional[int] = 100,
) -> tuple[EnvRenderer, RenderSpec]:
    """
    Return an instance of the renderer and the rendering specification.

    Parameters
    ----------
    params : dict
        dictionary of renderer parameters
    track : Track
        track object
    agent_ids : list
        list of agent ids to render
    render_mode : str, optional
        rendering mode, by default None
    render_fps : int, optional
        rendering frames per second, by default 100
    """
    cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering.yaml"
    
    render_spec = RenderSpec()
    render_spec.from_yaml(cfg_file)

    # if render_spec.render_type == "pygame": 
    #     from .rendering_pygame import PygameEnvRenderer as EnvRenderer
    if render_spec.render_type == "pyqt6":
        if render_mode in ["rgb_array", "rgb_array_list"]:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from .rendering_pyqt import PyQtEnvRenderer as EnvRenderer
    elif render_spec.render_type == "pyqt6gl":
        if render_mode in ["rgb_array", "rgb_array_list"]:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer
    else:
        raise ValueError(f"Unknown render type: {render_spec.render_type}")

    if render_mode in ["human", "rgb_array", 'unlimited', "human_fast"]:
        renderer = EnvRenderer(
            params=params,
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
            render_fps=render_fps,
        )
    else:
        renderer = None
    return renderer, render_spec
