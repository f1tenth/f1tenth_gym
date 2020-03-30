#pragma once

namespace racecar_simulator {

struct CarOdom {
    // default is in car frame
    // pose
    // position
    double x;
    double y;
    double z;
    // orientation
    double qx;
    double qy;
    double qz;
    double qw;

    // twist
    // linear
    double linear_x;
    double linear_y;
    double linear_z;
    // angular
    double angular_x;
    double angular_y;
    double angular_z;
};
}