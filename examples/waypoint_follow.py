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


class PurePursuitPlanner:
    """
    Example Planner
    """
    car_model = Car()
    track = Track()
    car_controller = CarController(track)

    vertex_list = pyglet.graphics.vertex_list(2,
        ('v2i', (10, 15, 30, 35)),
        ('c3B', (0, 0, 255, 0, 255, 0))
    )

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.


        self.track = Track()
        self.car_model = Car()
        self.car_controller = CarController(self.track)

        self.currentPosition = np.array([0.0, 0.0])
        self.nextPosition = np.array([0.0, 0.0])
        self.nextPositions = [[0.0, 0.0]]

        self.drawn_waypoints = []
        self.lidar_border_points = 1080 * [[0,0]]

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

        waypoints_x = self.waypoints[:,1]
        waypoints_y = self.waypoints[:,2]

        # print("WP", self.waypoints[:,1].shape)


    def render_test(self, e):
        

        points = np.array(self.currentPosition)
        scaled_points = 50.*points

        # print("Scaled Points", scaled_points)

        # e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[0], scaled_points[1], 0.]),
        #             ('c3B', (255, 0, 0)))

        # Render lidar data
        for i in range(len(self.lidar_border_points)):
            e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.lidar_border_points[i][0], self.lidar_border_points[i][1], 0.]),
                        ('c3B', (255, 0, 255)))


        points = np.array(self.nextPosition)
        scaled_points = 50.*points
        # self.vertex_list.delete()

        self.vertex_list.delete()
        self.vertex_list =  e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[0], scaled_points[1], 0.]), ('c3B', (0, 255, 0)))
        


        points = np.array(self.car_controller.simulated_history)
        chosen_trajectory = points[0][:][:]
        chosen_trajectory_positions = chosen_trajectory[:][:]
        c = 0

        

        if(points.shape[0] != 1):

            points = points.reshape((points.shape[0] * points.shape[1],7))
            trajectory = points[:,:2]

            scaled_points = 50.*trajectory

            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()

            c = c + 140
            self.vertex_list = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                        ('c3B', (0, c, 255 -c) * howmany ))


        self.car_controller.simulated_history  = [[0,0,0,0,0,0,0]]


        # for i in range(points.shape[0]):
        #      self.vertex_list = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
        #                 ('c3B', (0, 255,0)))


    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        return
        points = np.vstack(
            (self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack(
            (self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(
            position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

            
        self.nextPosition = self.car_model.state[:2]
        mppi_speed, mppi_steering = self.car_controller.plan()

        if(mppi_speed == -10):
            return mppi_speed, mppi_steering
        return speed, mppi_steering


def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312,
            'tlad': 1.82461887897713965, 'vgain': 1.30338203837889}

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # planner = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner = FollowTheGapPlanner(0.8)
    planner.plot_lidar_data = False

    planner_oponent = FollowTheGapPlanner(0.7)

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

    env = gym.make('f110_gym:f110-v0', map=conf.map_path,
                   map_ext=conf.map_ext, num_agents=2)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx -0.5 , conf.sy + 2, conf.stheta]]))


    env.render()

    laptime = 0.0
    start = time.time()


    car = env.sim.agents[0] 
    car_2 = env.sim.agents[1]   

    render_index = 0
    while not done:


        car_state = env.sim.agents[0].state
        ranges = obs['scans'][0]
        ranges_oponent = obs['scans'][1]



        # print("scan_angles", car.scan_angles)
        # print("side_distances", car.side_distances)
        # print("Scans",  obs['scans'][0])
        # print("obs", obs)

        odom_1 = {
            'pose_x': obs['poses_x'][0],
            'pose_y': obs['poses_y'][0],
            'pose_theta': obs['poses_theta'][0],
            'linear_vel_x': obs['linear_vels_x'][0],
            'linear_vel_y': obs['linear_vels_y'][0],
            'angular_vel_z': obs['ang_vels_z'][0]
        }
        print("odom_1", odom_1)

        # print("Car state", car_state)
        odom_2 = {
            'pose_x': obs['poses_x'][1],
            'pose_y': obs['poses_y'][1],
            'pose_theta': obs['poses_theta'][1],
            'linear_vel_x': obs['linear_vels_x'][1],
            'linear_vel_y': obs['linear_vels_y'][1],
            'angular_vel_z': obs['ang_vels_z'][1]
        }
        

        speed, steer =  planner.process_observation(ranges, odom_1)
        accl, sv = pid(speed, steer, car.state[3], car.state[2], car.params['sv_max'], car.params['a_max'], car.params['v_max'], car.params['v_min'])
        # sv = steer


        speed_2, steer_2 = planner_oponent.process_observation(ranges_oponent, odom_2)
        accl_2, sv_2 = pid(speed_2, steer_2, car_2.state[3], car_2.state[2], car_2.params['sv_max'], car_2.params['a_max'], car_2.params['v_max'], car_2.params['v_min'])

        obs, step_reward, done, info = env.step(np.array([[ accl, sv],[ accl_2, sv_2]]))
        # obs, step_reward, done, info = env.step(np.array([[ accl, sv],[ 0,0]]))

        laptime += step_reward
        env.render(mode='human')
        render_index += 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()
