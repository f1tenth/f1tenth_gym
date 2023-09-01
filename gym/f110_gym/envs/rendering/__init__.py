import pathlib

from f110_gym.envs.rendering.renderer import RenderSpec, EnvRenderer
from f110_gym.envs.track import Track


def make_renderer(track: Track, render_mode: str = None) -> EnvRenderer:
    cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering.yaml"
    render_spec = RenderSpec.from_yaml(cfg_file)

    if render_mode in ["human_fast"]:
        from f110_gym.envs.rendering.rendering_pyglet import PygletEnvRenderer
        renderer = PygletEnvRenderer(track=track, render_spec=render_spec)
    elif render_mode in ["human", "rgb_array"]:
        from f110_gym.envs.rendering.rendering_pygame import PygameEnvRenderer
        renderer = PygameEnvRenderer(track=track, render_spec=render_spec, render_mode=render_mode)
    else:
        renderer = None
    return renderer, render_spec
