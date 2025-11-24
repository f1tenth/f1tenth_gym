import numpy as np

from ..track import Raceline


def sample_around_waypoint(
    reference_line: Raceline,
    waypoint_id: int,
    n_agents: int,
    min_dist: float,
    max_dist: float,
    move_laterally: bool = True,
) -> np.ndarray:
    """
    Compute n poses around a given waypoint in the track.
    It iteratively samples the next agent within a distance range from the previous one.

    Args:
        - track: the track object
        - waypoint_id: the id of the first waypoint from which start the sampling
        - n_agents: the number of agents
        - min_dist: the minimum distance between two consecutive agents
        - max_dist: the maximum distance between two consecutive agents
        - move_laterally: if True, the agents are sampled on the left/right of the track centerline
    """
    current_wp_id = waypoint_id
    n_waypoints = reference_line.n

    poses = []
    rnd_sign = (
        np.random.choice([-1.0, 1.0]) if move_laterally else 0.0
    )  # random sign to sample lateral position (left/right)
    for i in range(n_agents):
        # compute pose from current wp_id
        wp = [
            reference_line.xs[current_wp_id],
            reference_line.ys[current_wp_id],
        ]
        next_wp_id = (current_wp_id + 1) % n_waypoints
        next_wp = [
            reference_line.xs[next_wp_id],
            reference_line.ys[next_wp_id],
        ]
        theta = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])

        x, y = wp[0], wp[1]
        if n_agents > 1:
            lat_offset = rnd_sign * (-1.0) ** i * (1.0 / n_agents)
            x += lat_offset * np.cos(theta + np.pi / 2)
            y += lat_offset * np.sin(theta + np.pi / 2)

        pose = np.array([x, y, theta])
        poses.append(pose)
        # find id of next waypoint which has mind <= dist <= maxd
        first_id, interval_len = (
            None,
            None,
        )  # first wp id with dist > mind, len of the interval btw first/last wp
        pnt_id = current_wp_id  # moving pointer to scan the next waypoints
        dist = 0.0
        while dist <= max_dist:
            # sanity check
            if pnt_id > n_waypoints - 1:
                pnt_id = 0
            # increment distance
            x_diff = reference_line.xs[pnt_id] - reference_line.xs[pnt_id - 1]
            y_diff = reference_line.ys[pnt_id] - reference_line.ys[pnt_id - 1]
            dist = dist + np.linalg.norm(
                [y_diff, x_diff]
            )  # approx distance by summing linear segments
            # look for sampling interval
            if first_id is None and dist >= min_dist:  # not found first id yet
                first_id = pnt_id
                interval_len = 0
            if (
                first_id is not None and dist <= max_dist
            ):  # found first id, increment interval length
                interval_len += 1
            pnt_id += 1
        # sample next waypoint
        current_wp_id = (first_id + np.random.randint(0, interval_len + 1)) % (
            n_waypoints
        )

    return np.array(poses)

def sample_around_pose(
    pose: np.ndarray,
    n_agents: int,
    min_dist: float,
    max_dist: float,
) -> np.ndarray:
    """
    Compute n poses around a given pose.
    It iteratively samples the next agent within a distance range from the previous one.
    Note: no guarantee that the agents are on the track nor that they are not colliding with the environment.

    Args:
        - pose: the initial pose
        - n_agents: the number of agents
        - min_dist: the minimum distance between two consecutive agents
        - max_dist: the maximum distance between two consecutive agents
    """
    current_pose = pose

    poses = []
    for i in range(n_agents):
        x, y, theta = current_pose
        pose = np.array([x, y, theta])
        poses.append(pose)
        # sample next pose
        dist = np.random.uniform(min_dist, max_dist)
        theta = np.random.uniform(-np.pi, np.pi)
        x += dist * np.cos(theta)
        y += dist * np.sin(theta)
        current_pose = np.array([x, y, theta])

    return np.array(poses)