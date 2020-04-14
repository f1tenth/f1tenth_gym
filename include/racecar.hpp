// MIT License

// Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "pose_2d.hpp"
#include "scan_simulator_2d.hpp"
#include "car_state.hpp"
#include "car_params.hpp"
#include "car_odom.hpp"
#include "car_obs.hpp"
#include "ackermann_kinematics.hpp"
#include "ks_kinematics.hpp"
#include "st_kinematics.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
// #include <chrono> // do we need to keep timestamp for each car?
#include <math.h>

using namespace racecar_simulator;

class RaceCar {
public:
    RaceCar(double time_step, double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass, bool is_ego);
    virtual ~RaceCar();
    void set_map(std::vector<double> &map, int map_height, int map_width, double map_resolution, double origin_x, double origin_y, double free_threshold);
    void reset();
    void reset_bypose(Pose2D pose);
    CarObs update_scan();
    void update_pose();
    void update_op_poses(const std::vector<Pose2D> &op_poses);
    void set_velocity(double vel);
    void set_steering_angle(double ang);
    Pose2D get_pose();
    double get_accel();
    double get_steer_vel();
    void update_params(double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass);
private:
    // objects
    CarState state;
    CarParams params;
    CarOdom odom;
    Pose2D pose;
    ScanSimulator2D scan_simulator;

    // flags
    bool map_exists;
    bool ego;

    // car params
    double scan_distance_to_base_link;
    double car_width, car_length;
    double max_speed, max_steering_angle;
    double max_accel, max_decel, max_steering_vel;
    double accel, steer_angle_vel;
    double width;

    // sim param
    double delt_t;

    // scan params
    std::vector<double> cosines;
    double scan_fov, scan_ang_incr;
    std::vector<double> scan_angles;

    // current scan
    std::vector<double> current_scan;

    // collision flag
    bool in_collision = false;

    // which beam in laser in collision
    double collision_angle;

    // ttc lower bound
    double ttc_threshold;

    // map params
    double map_free_threshold;

    // PI
    const double PI = 3.141592653;

    // distaces form lidar to edge of car
    std::vector<double> car_distances;

    // knowledge of other agents index doesn't matter?
    std::vector<Pose2D> opponent_poses;

    // steering delay
    int steering_delay_buffer_length;
    std::vector<double> steer_buffer;

    Eigen::Matrix4d get_transformation_matrix(const Pose2D &pose);
    Pose2D transform_between_frames(const Pose2D &p1, const Pose2D &p2);
    void ray_cast_opponents(std::vector<double> &scan, const Pose2D &scan_pose);
    void check_ttc();
    void set_accel(double acceleration);
    void set_steering_angle_vel(double steer_vel);
    double compute_steer_vel(double desired_angle);
    void compute_accel(double desired_velocity);
    double get_range(const Pose2D &pose, double beam_theta, Eigen::Vector2d line_segment_a, Eigen::Vector2d line_segment_b);
    bool are_collinear(Eigen::Vector2d pt_a, Eigen::Vector2d pt_b, Eigen::Vector2d pt_c);
    double cross(Eigen::Vector2d v1, Eigen::Vector2d v2);
};