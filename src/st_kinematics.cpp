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

#include <cmath>

#include "car_state.hpp"
#include "st_kinematics.hpp"
#include <iostream>

using namespace racecar_simulator;

// Implementation based off of Single Track Dynamics defined in CommonRoad: Vehicle Models
// https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf

CarState STKinematics::update(
        const CarState start,
        double accel,
        double steer_angle_vel,
        CarParams p,
        double dt) {


    double thresh = .5; // cut off to avoid singular behavior
    double err = .03; // to avoid flip flop
    if (!start.st_dyn)
        thresh += err;

    // if velocity is low or negative, use normal Kinematic Single Track dynamics
    if (start.velocity < thresh) {
        return update_k(
                    start,
                    accel,
                    steer_angle_vel,
                    p,
                    dt);
    }


    CarState end;

    double g = 9.81; // m/s^2

    // compute first derivatives of state
    double x_dot = start.velocity * std::cos(start.theta + start.slip_angle);
    double y_dot = start.velocity * std::sin(start.theta + start.slip_angle);
    double v_dot = accel;
    double steer_angle_dot = steer_angle_vel;
    double theta_dot = start.angular_velocity;

    // for eases of next two calculations
    double rear_val = g * p.l_r - accel * p.h_cg;
    double front_val = g * p.l_f + accel * p.h_cg;

    // in case velocity is 0
    double vel_ratio, first_term;
    if (start.velocity == 0) {
        vel_ratio = 0;
        first_term = 0;
    }
    else {
        vel_ratio = start.angular_velocity / start.velocity;
        first_term = p.friction_coeff / (start.velocity * (p.l_r + p.l_f));
    }

    double theta_double_dot = (p.friction_coeff * p.mass / (p.I_z * p.wheelbase)) *
            (p.l_f * p.cs_f * start.steer_angle * (rear_val) +
             start.slip_angle * (p.l_r * p.cs_r * (front_val) - p.l_f * p.cs_f * (rear_val)) -
             vel_ratio * (std::pow(p.l_f, 2) * p.cs_f * (rear_val) + std::pow(p.l_r, 2) * p.cs_r * (front_val)));\

    double slip_angle_dot = (first_term) *
            (p.cs_f * start.steer_angle * (rear_val) -
             start.slip_angle * (p.cs_r * (front_val) + p.cs_f * (rear_val)) +
             vel_ratio * (p.cs_r * p.l_r * (front_val) - p.cs_f * p.l_f * (rear_val))) -
            start.angular_velocity;


    // crude friction calc (should be rolling friction)
    //    double friction_term = 0;
    //    if (start.velocity > 0) {
    //        friction_term = -p.friction_coeff;
    //    } else if (start.velocity < 0) {
    //        friction_term = p.friction_coeff;
    //    }
    //    double fr_factor = .1;
    //    v_dot += fr_factor * friction_term;


    // update state
    end.x = start.x + x_dot * dt;
    end.y = start.y + y_dot * dt;
    end.theta = start.theta + theta_dot * dt;
    end.velocity = start.velocity + v_dot * dt;
    end.steer_angle = start.steer_angle + steer_angle_dot * dt;
    end.angular_velocity = start.angular_velocity + theta_double_dot * dt;
    end.slip_angle = start.slip_angle + slip_angle_dot * dt;
    end.st_dyn = true;


    //    std::cout << "start x:           " << start.x << std::endl;
    //    std::cout << "start y:           " << start.y << std::endl;
    //    std::cout << "start theta:       " << start.theta << std::endl;
    //    std::cout << "start velocity:    " << start.velocity << std::endl;
    //    std::cout << "start steer angle: " << start.steer_angle << std::endl;
    //    std::cout << "start ang vel:     " << start.angular_velocity << std::endl;
    //    std::cout << "start slip angle:  " << start.slip_angle << std::endl;


    //    std::cout << "x dot:                 " << x_dot << std::endl;
    //    std::cout << "y dot:                 " << y_dot << std::endl;
    //    std::cout << "theta dot:             " << theta_dot << std::endl;
    //    std::cout << "v dot (input):         " << v_dot << std::endl;
    //    std::cout << "steer ang dot (input): " << steer_angle_dot << std::endl;
    //    std::cout << "theta double dot:      " << theta_double_dot << std::endl;
    //    std::cout << "slip ang dot:          " << slip_angle_dot << std::endl;

    //    std::cout << std::endl;


    return end;
}

CarState STKinematics::update_k(
        const CarState start,
        double accel,
        double steer_angle_vel,
        CarParams p,
        double dt) {

    CarState end;

//    std::cout << "update_k is called" << std::endl;


    // compute first derivatives of state
    double x_dot = start.velocity * std::cos(start.theta);
    double y_dot = start.velocity * std::sin(start.theta);
    double v_dot = accel;
    double steer_angle_dot = steer_angle_vel;
    double theta_dot = start.velocity / p.wheelbase * std::tan(start.steer_angle);
    double theta_double_dot = accel / p.wheelbase * std::tan(start.steer_angle) +
            start.velocity * steer_angle_vel / (p.wheelbase * std::pow(std::cos(start.steer_angle), 2));
    double slip_angle_dot = 0;


    // crude friction calc (should be rolling friction)
    //    double friction_term = 0;
    //    if (start.velocity > 0) {
    //        friction_term = -p.friction_coeff;
    //    } else if (start.velocity < 0) {
    //        friction_term = p.friction_coeff;
    //    }
    //    double fr_factor = .1;
    //    v_dot += fr_factor * friction_term;


    // update state
    end.x = start.x + x_dot * dt;
    end.y = start.y + y_dot * dt;
    end.theta = start.theta + theta_dot * dt;
    end.velocity = start.velocity + v_dot * dt;
    end.steer_angle = start.steer_angle + steer_angle_dot * dt;
    end.angular_velocity = start.angular_velocity + theta_double_dot * dt;
    end.slip_angle = start.slip_angle + slip_angle_dot * dt;
    end.st_dyn = false;


    return end;

}
