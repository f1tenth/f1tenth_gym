from __future__ import annotations
import pathlib
from typing import List, Tuple, Any

from f110_gym.envs.rendering.renderer import RenderSpec, EnvRenderer
from f110_gym.envs.track import Track


def make_renderer(
    params: dict[str, Any],
    track: Track,
    agent_ids: list[str],
    render_mode: str = None,
    render_fps: int = 100,
) -> Tuple[EnvRenderer, RenderSpec]:
    from f110_gym.envs.rendering.rendering_pygame import PygameEnvRenderer

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
