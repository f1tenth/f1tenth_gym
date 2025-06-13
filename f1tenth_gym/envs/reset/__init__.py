from __future__ import annotations
from .masked_reset import GridResetFn, AllTrackResetFn
from .map_reset import AllMapResetFn
from .reset_fn import ResetFn
from ..track import Track


def make_reset_fn(type: str | None, track: Track, num_agents: int, **kwargs) -> ResetFn:
    type = type or "rl_grid_static"

    try:
        refline_token, reset_token, shuffle_token = type.split("_")

        if refline_token == "map":
            reset_fn = {"random": AllMapResetFn}[reset_token]
            shuffle = {"static": False, "random": True}[shuffle_token]
            return reset_fn(track=track, num_agents=num_agents, shuffle=shuffle, **kwargs)
        
        # "cl" or "rl"
        refline = {"cl": track.centerline, "rl": track.raceline}[refline_token]
        reset_fn = {"grid": GridResetFn, "random": AllTrackResetFn}[reset_token]
        shuffle = {"static": False, "random": True}[shuffle_token]
        options = {"cl": {"move_laterally": True}, "rl": {"move_laterally": False}}[refline_token]

    except Exception as ex:
        raise ValueError(f"Invalid reset function type: {type}. Expected format: <refline>_<resetfn>_<shuffle>") from ex

    return reset_fn(reference_line=refline, num_agents=num_agents, shuffle=shuffle, **options, **kwargs)
