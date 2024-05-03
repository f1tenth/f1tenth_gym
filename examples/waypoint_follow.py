import time
from typing import Tuple

import gymnasium as gym
import numpy as np
from numba import njit


"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(
            np.float32
        )  # NOTE: specify type or numba complains

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(
            V, start - point
        )  # NOTE: specify type or numba complains
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(
                V, start - point
            )  # NOTE: specify type or numba complains
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, track, wb):
        self.wheelbase = wb
        self.waypoints = np.stack(
            [track.raceline.xs, track.raceline.ys, track.raceline.vxs]
        ).T
        self.max_reacquire = 20.0

        self.drawn_waypoints = []
        self.lookahead_point = None
        self.current_index = None

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        # NOTE: specify type or numba complains
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        ).astype(np.float32)

    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.current_index is not None:
            points = self.waypoints[self.current_index : self.current_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=1)

    def _get_current_waypoint(
        self, waypoints, lookahead_distance, position, theta
    ) -> Tuple[np.ndarray, int]:
        """
        Returns the current waypoint to follow given the current pose.

        Args:
            waypoints: The waypoints to follow (Nx3 array)
            lookahead_distance: The lookahead distance [m]
            position: The current position (2D array)
            theta: The current heading [rad]

        Returns:
            waypoint: The current waypoint to follow (x, y, speed)
            i: The index of the current waypoint
        """
        wpts = waypoints[:, :2]
        lookahead_distance = np.float32(lookahead_distance)
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            t1 = np.float32(i + t)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, t1, wrap=True
            )
            if i2 is None:
                return None, None
            current_waypoint = np.empty((3,), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, -1]
            return current_waypoint, i
        elif nearest_dist < self.max_reacquire:
            # NOTE: specify type or numba complains
            return wpts[i, :], i
        else:
            return None, None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point, i = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        if lookahead_point is None:
            return 4.0, 0.0

        # for rendering
        self.lookahead_point = lookahead_point
        self.current_index = i

        # actuation
        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            lookahead_distance,
            self.wheelbase,
        )
        speed = vgain * speed

        return speed, steering_angle


def main():
    """
    main entry point
    """

    work = {
        "mass": 3.463388126201571,
        "lf": 0.15597534362552312,
        "tlad": 0.82461887897713965,
        "vgain": 1,
    }

    num_agents = 3
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": num_agents,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
            "reset_config": {"type": "rl_random_static"},
        },
        render_mode="human",
    )
    track = env.unwrapped.track

    planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

    env.unwrapped.add_render_callback(track.raceline.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_lookahead_point)

    obs, info = env.reset()
    done = False
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        action = env.action_space.sample()
        for i, agent_id in enumerate(obs.keys()):
            speed, steer = planner.plan(
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
                work["tlad"],
                work["vgain"],
            )
            action[i] = np.array([steer, speed])

        obs, step_reward, done, truncated, info = env.step(action)
        laptime += step_reward
        frame = env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
