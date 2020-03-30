#pragma once

#include "pose_2d.hpp"

namespace racecar_simulator {

class AckermannKinematics {

public:

    static double angular_velocity(
            double velocity,
            double steering_angle,
            double wheelbase);

    static Pose2D update(
            const Pose2D start,
            double velocity,
            double steering_angle,
            double wheelbase,
            double dt);

};

}
