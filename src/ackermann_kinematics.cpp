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

#include "pose_2d.hpp"
#include "ackermann_kinematics.hpp"

using namespace racecar_simulator;

double AckermannKinematics::angular_velocity(
    double velocity,
    double steering_angle,
    double wheelbase) {
  return velocity * std::tan(steering_angle) / wheelbase;
}

Pose2D AckermannKinematics::update(
    const Pose2D start, 
    double velocity, 
    double steering_angle, 
    double wheelbase, 
    double dt) {

  Pose2D end;

  double dthetadt = angular_velocity(velocity, steering_angle, wheelbase);
  end.theta = start.theta + dthetadt * dt;

  // The solution to the integral of
  // dxdt = v * cos(theta)
  // dydt = v * cos(theta)
  if (dthetadt == 0) {
    end.x = start.x + dt * velocity * std::cos(end.theta);
    end.y = start.y + dt * velocity * std::sin(end.theta);
  } else {
    end.x = start.x + (velocity/dthetadt) * (std::sin(end.theta) - std::sin(start.theta));
    end.y = start.y + (velocity/dthetadt) * (std::cos(start.theta) - std::cos(end.theta));
  }

  return end;
}
