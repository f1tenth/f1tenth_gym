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

#include "racecar.hpp"
using namespace racecar_simulator;

RaceCar::~RaceCar() {
    // do stuff
    // std::cout << "Race car - Shutting down race car instance." << std::endl;
}

RaceCar::RaceCar(double time_step, double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass, bool is_ego) {
    // initialize car state and driving commands
    state = {.x=0, .y=0, .theta=0, .velocity=0, .steer_angle=0, .angular_velocity=0, .slip_angle=0, .st_dyn=false};
    odom = {.x=0., .y=0., .z=0., .qx=0., .qy=0., .qz=0., .qw=1., .linear_x=0., .linear_y=0., .linear_z=0., .angular_x=0., .angular_y=0., .angular_z=0.};
    accel = 0.0;
    steer_angle_vel = 0.0;

    steering_delay_buffer_length = 1;
    steer_buffer = std::vector<double>(steering_delay_buffer_length);

    // ego identifier
    ego = is_ego;

    // car params, these are fixed ones
    double scan_std_dev;
    int scan_beams = 1080;
    params.wheelbase = 0.3302;

    // normal sim op
    scan_fov = 4.7;
    scan_std_dev = 0.01;

    // only for lut building
    // scan_fov = 6.28;
    // scan_std_dev = 0;

    map_free_threshold = 0.8;
    scan_distance_to_base_link = 0.275;
    max_speed = 20.0;
    max_accel = 9.51;
    max_decel = 13.26;
    max_steering_vel = 3.2;
    max_steering_angle = 0.4189;
    // params.friction_coeff = 0.523;
    // params.h_cg = 0.074;
    // params.l_r = 0.170;
    // params.l_f = 0.147;
    // params.cs_f = 4.718;
    // params.cs_r = 5.4562;
    // params.I_z = 0.04712;
    // params.mass = 3.51;
    width = 0.28;
    // car_width = 0.28;
    car_width = 0.31;
    car_length = 0.58;

    in_collision = false;

    // car params, these are variable
    params.friction_coeff = mu;
    params.h_cg = h_cg;
    params.l_r = l_r;
    params.l_f = params.wheelbase - params.l_r;
    params.cs_f = cs_f;
    params.cs_r = cs_r;
    params.I_z = I_z;
    params.mass = mass;

    delt_t = time_step;

    ttc_threshold = 0.005;

    // scan sim init
    scan_simulator = ScanSimulator2D(scan_beams, scan_fov, scan_std_dev);
    // std::cout << "Race car - Initialized scan sim for race car." << std::endl;
    scan_ang_incr = scan_simulator.get_angle_increment();

    // reserve space for current scan
    current_scan = std::vector<double>(scan_beams);

    // reserve space for scan angles
    scan_angles = std::vector<double>(scan_beams);

    // precomp consines and distance from lidar to edge of car
    cosines = std::vector<double>(scan_beams);
    car_distances = std::vector<double>(scan_beams);
    double dist_to_sides = width/2.0;
    double dist_to_front = params.wheelbase - scan_distance_to_base_link;
    double dist_to_back = scan_distance_to_base_link;

    for (int i=0; i<scan_beams; i++) {
        double angle = -scan_fov/2.0 + i*scan_ang_incr;
        scan_angles[i] = angle;
        cosines[i] = std::cos(angle);
        if (angle > 0) {
            if (angle < PI/2.0) {
                // between 0 and pi/2
                double to_side = dist_to_sides/std::sin(angle);
                double to_front = dist_to_front/std::cos(angle);
                car_distances[i] = std::min(to_side, to_front);
            } else {
                // between pi/2 and pi
                double to_side = dist_to_sides/std::cos(angle-PI/2.0);
                double to_back = dist_to_back/std::sin(angle-PI/2.0);
                car_distances[i] = std::min(to_side, to_back);
            }
        } else {
            if (angle > -PI/2.0) {
                // between 0 and -pi/2
                double to_side = dist_to_sides/std::sin(-angle);
                double to_front = dist_to_front/std::cos(-angle);
                car_distances[i] = std::min(to_side, to_front);
            } else {
                // between -pi/2 and -pi
                double to_side = dist_to_sides/std::cos(-angle-PI/2.0);
                double to_back = dist_to_back/std::sin(-angle-PI/2.0);
                car_distances[i] = std::min(to_side, to_back);
            }
        }
    }
    // std::cout << "x: " << state.x << ", y: " << state.y << ", theta: " << state.theta << std::endl;
}

void RaceCar::update_params(double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass) {
    params.friction_coeff = mu;
    params.h_cg = h_cg;
    params.l_r = l_r;
    params.l_f = params.wheelbase - params.l_r;
    params.cs_f = cs_f;
    params.cs_r = cs_r;
    params.I_z = I_z;
    params.mass = mass;
}

void RaceCar::set_map(std::vector<double> &map, int map_height, int map_width, double map_resolution, double origin_x, double origin_y, double free_threshold) {
    Pose2D map_origin;
    map_origin.x = origin_x;
    map_origin.y = origin_y;
    map_origin.theta = 0.0;
    scan_simulator.set_map(map, map_height, map_width, map_resolution, map_origin, free_threshold);
    map_exists = true;
    // std::cout << "Race car - Map Updated." << std::endl;
}

void RaceCar::reset() {
    if (ego) {
        state = {.x=-3.0, .y=0, .theta=0, .velocity=0, .steer_angle=0, .angular_velocity=0, .slip_angle=0, .st_dyn=false};
    } else {
        state = {.x=0, .y=0, .theta=0, .velocity=0, .steer_angle=0, .angular_velocity=0, .slip_angle=0, .st_dyn=false};
    }
    accel = 0.0;
    steer_angle_vel = 0.0;
    in_collision = false;
}

void RaceCar::reset_bypose(Pose2D pose) {
    // don't need to worry about ego or not??
    state = {.x=pose.x, .y=pose.y, .theta=pose.theta, .velocity=0, .steer_angle=0, .angular_velocity=0, .slip_angle=0, .st_dyn=false};
    accel = 0.0;
    steer_angle_vel = 0.0;
    in_collision = false;
}

Eigen::Matrix4d RaceCar::get_transformation_matrix(const Pose2D &pose) {
    // get transformation matrix from global frame to input pose
    double x = pose.x;
    double y = pose.y;
    double theta = pose.theta;
    double cosine = std::cos(theta);
    double sine = std::sin(theta);
    Eigen::Matrix4d rot;
    rot << cosine, -sine, 0.0, x, sine, cosine, 0.0, y,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return rot;
}

Pose2D RaceCar::transform_between_frames(const Pose2D &p1, const Pose2D &p2) {
    // p1 p2 both local frames
    // transform p2 into p1's frame

    Eigen::Vector4d p1_homo;
    p1_homo << p1.x, p1.y, 0.0, 1.0;
    Eigen::Vector4d p2_homo;
    p2_homo << p2.x, p2.y, 0.0, 1.0;

    Eigen::Matrix4d rot_1 = get_transformation_matrix(p1);
    Eigen::Matrix4d rot_1_inv = rot_1.inverse();

    Eigen::Vector4d p2_vec_transformed = rot_1_inv * p2_homo;
    p2_vec_transformed = p2_vec_transformed / p2_vec_transformed(3);

    Pose2D transformed_pose;
    transformed_pose.x = p2_vec_transformed(0);
    transformed_pose.y = p2_vec_transformed(1);
    transformed_pose.theta = p2.theta - p1.theta;

    return transformed_pose;
}

double RaceCar::cross(Eigen::Vector2d v1, Eigen::Vector2d v2) {
    return v1(0) * v2(1) - v1(1) * v2(0);
}

double RaceCar::get_range(const Pose2D &pose, double beam_theta, Eigen::Vector2d la, Eigen::Vector2d lb) {
    Eigen::Vector2d o {pose.x, pose.y};
    // Eigen::Vector2d la {line_segment_a(0), line_segment_a(1)};
    // Eigen::Vector2d lb {line_segment_b(0), line_segment_b(1)};
    Eigen::Vector2d v1 = o - la;
    Eigen::Vector2d v2 = lb - la;
    Eigen::Vector2d v3 {std::cos(beam_theta + PI/2.0), std::sin(beam_theta + PI/2.0)};

    double denom = v2.dot(v3);

    double x = INFINITY;

    if (std::fabs(denom) > 0.0) {
        double d1 = cross(v2, v1) / denom;
        double d2 = v1.dot(v3) / denom;
        if (d1 >= 0.0 && d2 >= 0.0 && d2 <= 1.0) {
            x = d1;
        }
    } else if (are_collinear(o, la, lb)) {
        double dist_a = (la - o).norm();
        double dist_b = (lb - o).norm();
        x = std::min(dist_a, dist_b);
    }
    return x;
}

bool RaceCar::are_collinear(Eigen::Vector2d pt_a, Eigen::Vector2d pt_b, Eigen::Vector2d pt_c) {
    double tol = 1e-8;
    auto ba = pt_b - pt_a;
    auto ca = pt_a - pt_c;
    bool col = (std::fabs(cross(ba, ca)) < tol);
    return col;
}

void RaceCar::ray_cast_opponents(std::vector<double> &scan, const Pose2D &scan_pose) {
    for (Pose2D op_pose : opponent_poses) {
        // get bounding box of the other cars

        double x = op_pose.x;
        double y = op_pose.y;
        double theta = op_pose.theta;

        Eigen::Vector2d diff_x {std::cos(theta), std::sin(theta)};
        Eigen::Vector2d diff_y {-std::sin(theta), std::cos(theta)};
        diff_x = (car_length/2) * diff_x;
        diff_y = (car_width/2) * diff_y;

        auto c1 = diff_x - diff_y;
        auto c2 = diff_x + diff_y;
        auto c3 = diff_y - diff_x;
        auto c4 = -diff_x - diff_y;

        Eigen::Vector2d corner1 {x+c1(0), y+c1(1)};
        Eigen::Vector2d corner2 {x+c2(0), y+c2(1)};
        Eigen::Vector2d corner3 {x+c3(0), y+c3(1)};
        Eigen::Vector2d corner4 {x+c4(0), y+c4(1)};

        // std::cout << "corner 1: " << corner1(0) << ", " << corner1(1) << std::endl;
        // std::cout << "corner 2: " << corner2(0) << ", " << corner2(1) << std::endl;
        // std::cout << "corner 3: " << corner3(0) << ", " << corner3(1) << std::endl;
        // std::cout << "corner 4: " << corner4(0) << ", " << corner4(1) << std::endl;

        std::vector<Eigen::Vector2d> bounding_boxes {corner1, corner2, corner3, corner4, corner1};

        // modify scan
        for (size_t i=0; i < scan_angles.size(); i++) {
            for (size_t j=0; j<4; j++) {
                double range = get_range(scan_pose, scan_pose.theta + scan_angles[i], bounding_boxes[j], bounding_boxes[j+1]);
                if (range < scan[i]) {
                    scan[i] = range;
                }
            }
        }
    }
}

// void RaceCar::ray_cast_opponents(std::vector<double> &scan, const Pose2D &scan_pose) {
//     for (Pose2D op_pose : opponent_poses) {
//         // get four corners of car
//         Eigen::Matrix4d op_trans_mat = get_transformation_matrix(op_pose);
//         Eigen::Vector4d rear_left_homo;
//         rear_left_homo << 0.0, car_width/2, 0.0, 1.0;
//         Eigen::Vector4d rear_right_homo;
//         rear_right_homo << 0.0, -car_width/2, 0.0, 1.0;
//         Eigen::Vector4d front_left_homo;
//         front_left_homo << scan_distance_to_base_link, car_width/2, 0.0, 1.0;
//         Eigen::Vector4d front_right_homo;
//         front_right_homo << scan_distance_to_base_link, -car_width/2, 0.0, 1.0;

//         // transform four corners into op car's frame
//         Eigen::Vector4d rear_left_transformed = op_trans_mat*rear_left_homo;
//         rear_left_transformed = rear_left_transformed / rear_left_transformed(3);
//         Eigen::Vector4d rear_right_transformed = op_trans_mat*rear_right_homo;
//         rear_right_transformed = rear_right_transformed / rear_right_transformed(3);
//         Eigen::Vector4d front_left_transformed = op_trans_mat*front_left_homo;
//         front_left_transformed = front_left_transformed / front_left_transformed(3);
//         Eigen::Vector4d front_right_transformed = op_trans_mat*front_right_homo;
//         front_right_transformed = front_right_transformed / front_right_transformed(3);

//         // turn into pose2ds
//         Pose2D rear_left = {.x=rear_left_transformed(0), .y=rear_left_transformed(1), op_pose.theta};
//         Pose2D rear_right = {.x=rear_right_transformed(0), .y=rear_right_transformed(1), op_pose.theta};
//         Pose2D front_left = {.x=front_left_transformed(0), .y=front_left_transformed(1), op_pose.theta};
//         Pose2D front_right = {.x=front_right_transformed(0), .y=front_right_transformed(1), op_pose.theta};

//         // transform four points to ego car's frame
//         Pose2D rear_left_ego = transform_between_frames(scan_pose, rear_left);
//         Pose2D rear_right_ego = transform_between_frames(scan_pose, rear_right);
//         Pose2D front_left_ego = transform_between_frames(scan_pose, front_left);
//         Pose2D front_right_ego = transform_between_frames(scan_pose, front_right);
//         // rear left right
//         double op_rl_x = rear_left_ego.x;
//         double op_rl_y = rear_left_ego.y;
//         double op_rr_x = rear_right_ego.x;
//         double op_rr_y = rear_right_ego.y;
//         // front left right
//         double op_fl_x = front_left_ego.x;
//         double op_fl_y = front_left_ego.y;
//         double op_fr_x = front_right_ego.x;
//         double op_fr_y = front_right_ego.y;

//         // four corners from cartesian to polar
//         double rl_r = std::sqrt(std::pow(op_rl_x, 2) + std::pow(op_rl_y, 2));
//         double rl_theta = std::atan2(op_rl_y, op_rl_x);

//         double rr_r = std::sqrt(std::pow(op_rr_x, 2) + std::pow(op_rr_y, 2));
//         double rr_theta = std::atan2(op_rr_y, op_rr_x);

//         double fl_r = std::sqrt(std::pow(op_fl_x, 2) + std::pow(op_fl_y, 2));
//         double fl_theta = std::atan2(op_fl_y, op_fl_x);

//         double fr_r = std::sqrt(std::pow(op_fr_x, 2) + std::pow(op_fr_y, 2));
//         double fr_theta = std::atan2(op_fr_y, op_fr_x);

//         std::vector<double> list_ranges {{rl_r, rr_r, fl_r, fr_r}};
//         std::vector<double> list_thetas {{rl_theta, rr_theta, fl_theta, fr_theta}};

//         for (size_t i=0; i<list_ranges.size(); i++) {
//             if (fabs(list_thetas[i]) > scan_simulator.get_field_of_view()/2.) {
//                 continue;
//             } else {
//                 int scan_idx = static_cast<int>((list_thetas[i]-(-scan_simulator.get_field_of_view()/2.))/scan_simulator.get_angle_increment());
//                 if (scan[scan_idx] >= list_ranges[i]) {
//                     scan[scan_idx] = list_ranges[i];
//                 }
//             }
//         }
//     }
// }

void RaceCar::check_ttc() {
    if (state.velocity != 0) {
        // std::cout << "Race car - current scan size: " << current_scan.size() << std::endl;
        // std::cout << "Race car - car_distances size: " << car_distances.size() << std::endl;
        // std::cout << "Race car - scan_angles size: " << scan_angles.size() << std::endl;
        for (size_t i=0; i<current_scan.size(); i++) {
            // calc projected vel
            double proj_velocity = state.velocity * cosines[i];
            double ttc = (current_scan[i] - car_distances[i])/proj_velocity;
            // if ttc small, count as collision
            if ((ttc < ttc_threshold) && (ttc >= 0.0)) {
                in_collision = true;
                collision_angle = scan_angles[i];
                // stops vehicle, no collision physics
                state.velocity = 0.0;
                state.angular_velocity = 0.0;
                state.slip_angle = 0.0;
                state.steer_angle = 0.0;
                steer_angle_vel = 0.0;
                accel = 0.0;
                // no need to check others
                // std::cout << "Race car - In collision." << std::endl;
                break;
            }
        }
    }
    // std::cout << "Race car - TTC done" << std::endl;
}

// update_scan gets called by the simulator once all vehicle states are updated
// according to their corresponding update_pose
// checks for collision
// returns a full observation
CarObs RaceCar::update_scan() {
    // modify scan
    // need to raycast after all poses are updated
    Pose2D scan_pose;
    scan_pose.x = state.x + scan_distance_to_base_link*std::cos(state.theta);
    scan_pose.y = state.y + scan_distance_to_base_link*std::sin(state.theta);
    scan_pose.theta = state.theta;
    // if (opponent_poses.size() > 1) {
    //     ray_cast_opponents(current_scan, scan_pose);
    // }

    // check ttc
    check_ttc();
    // std::cout << "Race car - TTC check done." << std::endl;
    // if (opponent_poses.size() > 1) {
    ray_cast_opponents(current_scan, scan_pose);
    // }
    CarObs observation;
    observation.odom = odom;
    observation.pose = pose;
    observation.scan = current_scan;
    observation.in_collision = in_collision;
    observation.collision_angle = collision_angle;
    // std::cout << "Race car - obs updated." << std::endl;
    return observation;
}

// update_pose only handles updating the vehicle state and updating odometry
// This will be called by the simulator first
void RaceCar::update_pose() {
    if (!map_exists) {
        // std::cout << "Race car - WARNING: updating pose without a map.";
    }
    // TODO: constant time step?
    state = STKinematics::update(
                state,
                accel,
                steer_angle_vel,
                params,
                delt_t);
    // clip
    state.velocity = std::min(std::max(state.velocity, -max_speed), max_speed);
    state.steer_angle = std::min(std::max(state.steer_angle, -max_steering_angle), max_steering_angle);

    // update scan
    Pose2D scan_pose;
    scan_pose.x = state.x + scan_distance_to_base_link*std::cos(state.theta);
    scan_pose.y = state.y + scan_distance_to_base_link*std::sin(state.theta);
    scan_pose.theta = state.theta;
    current_scan = scan_simulator.scan(scan_pose);
    // update odom
    odom.x = state.x;
    odom.y = state.y;
    odom.z = 0.0;
    // TODO: theta to quaternion
    // odom.qx
    odom.linear_x = state.velocity;
    odom.linear_y = 0.0;
    odom.linear_z = 0.0;
    odom.angular_x = 0.0;
    odom.angular_y = 0.0;
    odom.angular_z = state.angular_velocity;
    // update pose 2d
    pose.x = state.x;
    pose.y = state.y;
    pose.theta = state.theta;

}

// setters
void RaceCar::update_op_poses(const std::vector<Pose2D> &op_poses) {
    // = operator because std vec copy
    opponent_poses = op_poses;
}

void RaceCar::set_velocity(double vel) {
    compute_accel(vel);
}

void RaceCar::set_steering_angle(double ang) {
    double actual_ang = 0.0;
    if (steer_buffer.size() < steering_delay_buffer_length) {
        steer_buffer.push_back(ang);
        actual_ang = 0.0;
    } else {
        steer_buffer.insert(steer_buffer.begin(), ang);
        actual_ang = steer_buffer.back();
        steer_buffer.pop_back();
    }
    set_steering_angle_vel(compute_steer_vel(actual_ang));
}

void RaceCar::set_accel(double acceleration) {
    accel = std::min(std::max(acceleration, -max_accel), max_accel);
    // std::cout << "Race car - accel: " << accel << std::endl;
}

void RaceCar::set_steering_angle_vel(double steer_vel) {
    steer_angle_vel = std::min(std::max(steer_vel, -max_steering_vel), max_steering_vel);
}

// getters
Pose2D RaceCar::get_pose() {
    Pose2D current_pose;
    current_pose.x = state.x;
    current_pose.y = state.y;
    current_pose.theta = state.theta;
    return current_pose;
}

// debug getters
double RaceCar::get_accel() {
    return accel;
}
double RaceCar::get_steer_vel() {
    return steer_angle_vel;
}

// vel calc
double RaceCar::compute_steer_vel(double desired_angle) {
    // get difference between current and desired
    // std::cout << "Race car - desired ang in compute: " << desired_angle << std::endl;
    // std::cout << "Race car - state ang in compute: " << state.steer_angle << std::endl;
    double dif = (desired_angle - state.steer_angle);
    // std::cout << "Race car - dif in compute: " << dif << std::endl;
    // calculate velocity
    double steer_vel;
    if (std::fabs(dif) > .0001)  // if the difference is not trivial
        steer_vel = (dif / std::fabs(dif)) * max_steering_vel;
    else {
        steer_vel = 0;
    }
    // std::cout << "Race car - steer vel in compute: " << steer_vel << std::endl;

    return steer_vel;
}

void RaceCar::compute_accel(double desired_velocity) {
    // get difference between current and desired
    double dif = (desired_velocity - state.velocity);
    if (state.velocity > 0) {
        if (dif > 0) {
            // accelerate
            double kp = 2.0 * max_accel / max_speed;
            set_accel(kp * dif);
        } else {
            // brake
            accel = -max_decel;
        }
    } else {
        if (dif > 0) {
            // brake
            accel = max_decel;

        } else {
            // accelerate
            double kp = 2.0 * max_accel / max_speed;
            set_accel(kp * dif);
        }
    }
}
