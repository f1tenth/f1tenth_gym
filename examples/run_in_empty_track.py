import numpy as np

from waypoint_follow import PurePursuitPlanner
from f1tenth_gym.envs.track import Track
import gymnasium as gym


def main():
    """
    Demonstrate the creation of an empty map with a custom reference line.
    This is useful for testing and debugging control algorithms on standard maneuvers.
    """
    # create sinusoidal reference line with custom velocity profile
    xs = np.linspace(0, 100, 200)
    ys = np.sin(xs / 2.0) * 5.0
    velxs = 4.0 * (1 + (np.abs(np.cos(xs / 2.0))))

    # create track from custom reference line
    track = Track.from_refline(x=xs, y=ys, velx=velxs)

    # env and planner
    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": track,
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode="human",
    )
    planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

    # rendering callbacks
    env.add_render_callback(track.raceline.render_waypoints)
    env.add_render_callback(planner.render_lookahead_point)

    # simulation
    obs, info = env.reset()
    done = False
    env.render()

    while not done:
        speed, steer = planner.plan(
            obs["agent_0"]["pose_x"],
            obs["agent_0"]["pose_y"],
            obs["agent_0"]["pose_theta"],
            lookahead_distance=0.8,
            vgain=1.0,
        )
        action = np.array([[steer, speed]])
        obs, timestep, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()


if __name__ == "__main__":
    main()
