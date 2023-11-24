from f110_gym.envs.reset.masked_reset import GridResetFn, AllTrackResetFn
from f110_gym.envs.reset.reset_fn import ResetFn
from f110_gym.envs.track import Track


def make_reset_fn(type: str, track: Track, num_agents: int, **kwargs) -> ResetFn:
    if type == "grid_static":
        return GridResetFn(track=track, num_agents=num_agents, shuffle=False, **kwargs)
    elif type == "grid_random":
        return GridResetFn(track=track, num_agents=num_agents, shuffle=True, **kwargs)
    elif type == "random_static":
        return AllTrackResetFn(
            track=track, num_agents=num_agents, shuffle=False, **kwargs
        )
    elif type == "random_random":
        return AllTrackResetFn(
            track=track, num_agents=num_agents, shuffle=True, **kwargs
        )
    else:
        raise ValueError(f"invalid reset type {type}")
