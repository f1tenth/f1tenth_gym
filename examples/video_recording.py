import time
import gymnasium as gym
import gymnasium.wrappers
import numpy as np

from waypoint_follow import PurePursuitPlanner


def main():
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1,
    }

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        },
        render_mode="rgb_array",
    )
    env = gymnasium.wrappers.RecordVideo(env, f"video_{time.time()}")
    track = env.unwrapped.track

    planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

    poses = np.array(
        [
            [
                track.raceline.xs[0],
                track.raceline.ys[0],
                track.raceline.yaws[0],
            ]
        ]
    )

    obs, info = env.reset(options={"poses": poses})
    done = False

    laptime = 0.0
    start = time.time()

    frames = [env.render()]
    while not done and laptime < 15.0:
        action = env.action_space.sample()
        for i, agent_id in enumerate(obs.keys()):
            speed, steer = planner.plan(
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
                work["tlad"],
                work["vgain"],
            )
            action[i] = [steer, speed]

        obs, step_reward, done, truncated, info = env.step(action)
        laptime += step_reward

        frame = env.render()
        frames.append(frame)

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)

    # close env to trigger video saving
    env.close()


if __name__ == "__main__":
    main()
