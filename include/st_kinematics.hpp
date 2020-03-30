#pragma once

#include "car_state.hpp"
#include "car_params.hpp"

namespace racecar_simulator {

class STKinematics {

public:

    static CarState update(
            const CarState start,
            double accel,
            double steer_angle_vel,
            CarParams p,
            double dt);


    static CarState update_k(
            const CarState start,
            double accel,
            double steer_angle_vel,
            CarParams p,
            double dt);
};

}
