import numpy as np

from f110_gym.envs import F110Env
from f110_gym.envs.utils import deep_update


def pretty_print(dict: dict, col_width=15):
    keys = list(dict.keys())
    columns = ["key"] + [str(k) for k in dict[keys[0]]]

    # opening line
    for _ in columns:
        print("|" + "-" * col_width, end="")
    print("|")
    # header
    for col in columns:
        padding = max(0, col_width - len(col))
        print("|" + col[:col_width] + " " * padding, end="")
    print("|")
    # separator line
    for _ in columns:
        print("|" + "-" * col_width, end="")
    print("|")

    # table
    for key in keys:
        padding = max(0, col_width - len(str(key)))
        print("|" + str(key)[:col_width] + " " * padding, end="")
        for col in columns[1:]:
            padding = max(0, col_width - len(str(dict[key][col])))
            print("|" + str(dict[key][col])[:col_width] + " " * padding, end="")
        print("|")

    # footer
    for col in columns:
        print("|" + "-" * col_width, end="")
    print("|")


class BenchmarkRenderer:
    @staticmethod
    def _make_env(config={}, render_mode=None) -> F110Env:
        import gymnasium as gym
        import f110_gym

        base_config = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        }
        config = deep_update(base_config, config)

        env = gym.make(
            "f110_gym:f110-v0",
            config=config,
            render_mode=render_mode,
        )

        return env

    def benchmark_single_agent_rendering(self):
        import time

        sim_time = 15.0  # seconds
        results = {}

        for render_mode in [None, "human", "human_fast", "rgb_array", "rgb_array_list"]:
            env = self._make_env(render_mode=render_mode)
            env.reset()
            frame = env.render()

            print(
                f"Running simulation of {sim_time}s for render mode: {render_mode}..."
            )

            max_steps = int(sim_time / env.timestep)
            t0 = time.time()
            for _ in range(max_steps):
                action = env.action_space.sample()
                env.step(action)
                frame = env.render()
            tf = time.time()
            env.close()

            results[render_mode] = {
                "sim_time": sim_time,
                "elapsed_time": tf - t0,
                "fps": max_steps / (tf - t0),
            }

        pretty_print(results)

    def benchmark_n_agents_human_rendering(self):
        """
        This is meant to benchmark the human rendering mode, for increasing nr of agents.
        """
        import time

        sim_time = 15.0  # seconds
        render_mode = "human"

        results = {}

        for num_agents in [1, 2, 3, 4, 5, 10]:
            env = self._make_env(
                config={"num_agents": num_agents}, render_mode=render_mode
            )
            env.reset()
            frame = env.render()

            print(
                f"Running simulation of {num_agents} agents for render mode: {render_mode}..."
            )

            max_steps = int(sim_time / env.timestep)
            t0 = time.time()
            for _ in range(max_steps):
                action = env.action_space.sample()
                env.step(action)
                frame = env.render()
            tf = time.time()
            env.close()

            results[num_agents] = {
                "sim_time": sim_time,
                "elapsed_time": tf - t0,
                "fps": max_steps / (tf - t0),
            }

        pretty_print(results)

    def benchmark_callbacks_human_rendering(self):
        import time

        sim_time = 15.0  # seconds
        render_mode = "human"

        results = {}

        class GoStraightPlanner:
            def __init__(self, env, agent_id: str = "agent_0"):
                self.waypoints = np.stack(
                    [env.track.raceline.xs, env.track.raceline.ys]
                ).T
                self.pos = None
                self.agent_id = agent_id

            def plan(self, obs):
                state = obs[self.agent_id]
                self.pos = np.array([state["pose_x"], state["pose_y"]])
                return np.array([0.0, 2.5])

            def render_waypoints(self, e):
                e.render_closed_lines(points=self.waypoints, size=1)

            def render_position(self, e):
                if self.pos is not None:
                    points = self.pos[None]
                    e.render_points(points, size=1)

        for render_config in [[False, False], [True, False], [True, True]]:
            env = self._make_env(render_mode=render_mode)
            planner = GoStraightPlanner(env)

            show_path, show_point = render_config
            config_str = f"show_path={show_path}, show_point={show_point}"

            if show_path:
                env.add_render_callback(callback_func=planner.render_waypoints)

            if show_point:
                env.add_render_callback(callback_func=planner.render_position)

            rnd_idx = np.random.randint(0, len(env.track.raceline.xs))
            obs, _ = env.reset(
                options={
                    "poses": np.array(
                        [
                            [
                                env.track.raceline.xs[rnd_idx],
                                env.track.raceline.ys[rnd_idx],
                                env.track.raceline.yaws[rnd_idx],
                            ]
                        ]
                    )
                }
            )
            frame = env.render()

            print(
                f"Running simulation of {config_str} for render mode: {render_mode}..."
            )

            max_steps = int(sim_time / env.timestep)
            t0 = time.time()
            for _ in range(max_steps):
                action = planner.plan(obs=obs)
                obs, _, _, _, _ = env.step(np.array([action]))
                frame = env.render()
            tf = time.time()
            env.close()

            results[config_str] = {
                "sim_time": sim_time,
                "elapsed_time": tf - t0,
                "fps": max_steps / (tf - t0),
            }

        pretty_print(results)


if __name__ == "__main__":
    benchmark = BenchmarkRenderer()

    benchmark.benchmark_single_agent_rendering()
    benchmark.benchmark_n_agents_human_rendering()
    benchmark.benchmark_callbacks_human_rendering()
