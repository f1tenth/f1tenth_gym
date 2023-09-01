import time
import gymnasium as gym
import gymnasium.wrappers
import numpy as np

from examples.waypoint_follow import PurePursuitPlanner


def main():
    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1,
    }  # 0.90338203837889}

    env = gym.make(
        "f110_gym:f110-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": "speed",
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        },
        render_mode="rgb_array",
    )
    env = gymnasium.wrappers.RecordVideo(env, f"video_{time.time()}")

    planner = PurePursuitPlanner(track=env.track, wb=0.17145 + 0.15875)

    poses = np.array(
        [
            [
                env.track.raceline.xs[0],
                env.track.raceline.ys[0],
                env.track.raceline.yaws[0],
            ]
        ]
    )

    obs, info = env.reset(options={"poses": poses})
    done = False

    laptime = 0.0
    start = time.time()

    frames = [env.render()]
    while not done and laptime < 15.0:
        agent_id = env.agent_ids[0]
        speed, steer = planner.plan(
            obs[agent_id]["pose_x"],
            obs[agent_id]["pose_y"],
            obs[agent_id]["pose_theta"],
            work["tlad"],
            work["vgain"],
        )
        action = np.array([[steer, speed]])
        obs, step_reward, done, truncated, info = env.step(action)
        laptime += step_reward

        frame = env.render()
        frames.append(frame)

        print(laptime)

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
