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
#include "ackermann_kinematics.hpp"
#include "scan_simulator_2d.hpp"

#include "car_state.hpp"
#include "car_params.hpp"
#include "ks_kinematics.hpp"
#include "st_kinematics.hpp"

#include "racecar.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>

using namespace racecar_simulator;

class StandaloneSimulator {
public:
    bool map_exists;
	StandaloneSimulator(int num_cars, double timestep, double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass);
	virtual ~StandaloneSimulator();
	void set_map(std::vector<double> map, int map_height, int map_width, double map_resolution, double origin_x, double origin_y, double free_threshold);
	std::vector<CarObs> step(std::vector<double> velocities, std::vector<double> steering_angles);
    void update_params(double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass);
	bool get_map_status();
    bool check_collision();
	void reset();
    void reset_bypose(std::vector<Pose2D> &poses);

private:
	// sim constants
    // time step
    double delt_t;

    // multi-agents
    int ego_agent_idx = 0;
    double safety_radius = 1.0;
    int num_agents;
    std::vector<RaceCar> agents;
    std::vector<Pose2D> agent_poses;
    std::vector<CarObs> current_obs;

    // car params - CHECK racecar.cpp BEFORE CHANGING
    double car_width = 0.31;
    double car_length = 0.58;
    double scan_distance_to_base_link = 0.275;
    double wheel_base = 0.3302;
    
    // all agents have the same map
    std::vector<double> map;
    int map_height, map_width;
    double map_resolution, origin_x, origin_y, free_threshold;
    Eigen::Matrix4d get_transformation_matrix(const Pose2D &pose);
};