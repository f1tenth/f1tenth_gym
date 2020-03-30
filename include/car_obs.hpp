#pragma once
#include "car_odom.hpp"
#include "car_state.hpp"
#include "pose_2d.hpp"
#include <vector>

namespace racecar_simulator {

struct CarObs {
	// full observation from the car
	// consists of full scan, odometry, pose2d, collision bool, collision location
	// can be expanded to have IMU etc.
	CarOdom odom;
    Pose2D pose;
	std::vector<double> scan;
	bool in_collision;
	double collision_angle;
};

}