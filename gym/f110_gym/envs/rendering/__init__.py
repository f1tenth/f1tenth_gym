import pathlib
from typing import List, Tuple

from f110_gym.envs.rendering.renderer import RenderSpec, EnvRenderer
from f110_gym.envs.track import Track


def make_renderer(
    track: Track, agent_ids: List[str], render_mode: str = None
) -> Tuple[EnvRenderer, RenderSpec]:
    from f110_gym.envs.rendering.rendering_pygame import PygameEnvRenderer

    cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering.yaml"
    render_spec = RenderSpec.from_yaml(cfg_file)

    if render_mode in ["human_fast"]:
        cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering_fast.yaml"
        render_spec = RenderSpec.from_yaml(cfg_file)
        renderer = PygameEnvRenderer(
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
        )
    elif render_mode in ["human", "rgb_array"]:
        renderer = PygameEnvRenderer(
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
        )
    else:
        renderer = None
    return renderer, render_spec
