#pragma once

namespace racecar_simulator {

struct CarState {
    double x; // x position
    double y; // y position
    double theta; // orientation
    double velocity;
    double steer_angle;
    double angular_velocity;
    double slip_angle;
    bool st_dyn;
};

}
