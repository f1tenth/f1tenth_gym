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
