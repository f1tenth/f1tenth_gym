import pathlib
from typing import Any, Optional

from .renderer import RenderSpec, EnvRenderer
from ..track import Track


def make_renderer(
    params: dict[str, Any],
    track: Track,
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
    from .rendering_pygame import PygameEnvRenderer

    cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering.yaml"
    render_spec = RenderSpec.from_yaml(cfg_file)

    if render_mode in ["human", "rgb_array", "human_fast"]:
        renderer = PygameEnvRenderer(
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
