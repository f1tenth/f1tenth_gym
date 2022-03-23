import time

from matplotlib.font_manager import json_dump
from matplotlib.pyplot import close, sca
import yaml
import gym
import numpy as np
from argparse import Namespace
import json
from track import *
from car import *
from car_controller import *
from ftg_planner import FollowTheGapPlanner

from OpenGL.GL import *
from numba import njit

from pyglet.gl import GL_POINTS
import pyglet

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid



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
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
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
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i, :]
        end = trajectory[i+1, :]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - \
            2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
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
            end = trajectory[(i+1) % trajectory.shape[0], :]+1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - \
                2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
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
    waypoint_y = np.dot(np.array(
        [np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


def main():
    """
    main entry point
    """

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # First car
    planner = FollowTheGapPlanner(0.8)
    planner.plot_lidar_data = False
    planner.draw_lidar_data = True
    planner.lidar_visualization_color = (255, 0, 255)



    # 2nd Car
    planner_2 = FollowTheGapPlanner(0.7)
    planner_2.plot_lidar_data = False
    planner_2.draw_lidar_data = True
    planner_2.lidar_visualization_color = (255, 255, 255)
   


    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_ftg(env_renderer)
        planner_2.render_ftg(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path,
                   map_ext=conf.map_ext, num_agents=2)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx -0.5 , conf.sy + 2, conf.stheta]]))

    car = env.sim.agents[0] 
    car_2 = env.sim.agents[1]

    env.render()

    laptime = 0.0
    start = time.time()

    render_index = 0
    while not done:


        ranges = obs['scans'][0]
        ranges_oponent = obs['scans'][1]

        # print("scan_angles", car.scan_angles)
        # print("side_distances", car.side_distances)
        # print("Scans",  obs['scans'][0])
        # print("obs", obs)
        # print("Car state", car_state)

        # First car
        odom_1 = {
            'pose_x': obs['poses_x'][0],
            'pose_y': obs['poses_y'][0],
            'pose_theta': obs['poses_theta'][0],
            'linear_vel_x': obs['linear_vels_x'][0],
            'linear_vel_y': obs['linear_vels_y'][0],
            'angular_vel_z': obs['ang_vels_z'][0]
        }

        speed, steer =  planner.process_observation(ranges, odom_1)
        accl, sv = pid(speed, steer, car.state[3], car.state[2], car.params['sv_max'], car.params['a_max'], car.params['v_max'], car.params['v_min'])

        # Second car
        odom_2 = {
            'pose_x': obs['poses_x'][1],
            'pose_y': obs['poses_y'][1],
            'pose_theta': obs['poses_theta'][1],
            'linear_vel_x': obs['linear_vels_x'][1],
            'linear_vel_y': obs['linear_vels_y'][1],
            'angular_vel_z': obs['ang_vels_z'][1]
        }

        speed_2, steer_2 = planner_2.process_observation(ranges_oponent, odom_2)
        accl_2, sv_2 = pid(speed_2, steer_2, car_2.state[3], car_2.state[2], car_2.params['sv_max'], car_2.params['a_max'], car_2.params['v_max'], car_2.params['v_min'])


        obs, step_reward, done, info = env.step(np.array([[ accl, sv],[ accl_2, sv_2]]))

        laptime += step_reward
        env.render(mode='human')
        render_index += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()
