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




// simulator manages all the race car instances
// order of update at each step is:
//      1. pose, vel, odom, save odom to instance attributes according to action in
//      2. vector of agent poses in the simulator class
//      3. scan for each instance, save to instance attributes
//      4. modified scan for each instance according to agent poses
//      5. create full observation with zmq
//      TODO: decide whether this class should have zmq at all or should another
//            wrapper for zmq created

#include "simulator.hpp"
#include "gjk.hpp"
using namespace racecar_simulator;

StandaloneSimulator::StandaloneSimulator(int num_cars, double timestep, double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass) {
    num_agents = num_cars;
    delt_t = timestep;
    map_exists = false;
    // spawn agents in constructor
    agent_poses.reserve(num_agents);
    // agents.reserve(num_agents);
    for (int i=0; i<num_agents; i++) {
        if (i == ego_agent_idx) {
            // TODO: something might have to be different for ego
            RaceCar ego_car = RaceCar(delt_t, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, true);
            // std::cout << "Simulator - Ego car initialized." << std::endl;
            agents.push_back(ego_car);
            // std::cout << "Simulator - Ego car pushed into list." << std::endl;
        } else {
            RaceCar agent = RaceCar(delt_t, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, false);
            // std::cout << "Simulator - Race car agent initialized." << std::endl;
            agents.push_back(agent);
            // std::cout << "Simulator - Race car pushed into list." << std::endl;
        }
    }
}

StandaloneSimulator::~StandaloneSimulator() {
    // destructor
    // std::cout << "Simulator - Simulator instance shutting down." << std::endl;
}

// set map for simulator
void StandaloneSimulator::set_map(std::vector<double> map, int map_height, int map_width, double map_resolution, double origin_x, double origin_y, double free_threshold) {
    this->map = map;
    this->map_height = map_height;
    this->map_width = map_width;
    this->map_resolution = map_resolution;
    this->origin_x = origin_x;
    this->origin_y = origin_y;
    this->free_threshold = free_threshold;
    // std::cout << "Simulator - Map Updated for Simulator Instance." << std::endl;

    for (RaceCar &agent : agents) {
        agent.set_map(map, map_height, map_width, map_resolution, origin_x, origin_y, free_threshold);
    }
    // std::cout << "Simulator - Map Updated for Racear Instances." << std::endl;

    map_exists = true;
}

void StandaloneSimulator::update_params(double mu, double h_cg, double l_r, double cs_f, double cs_r, double I_z, double mass) {
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass);
    }
}

bool StandaloneSimulator::get_map_status() {
    // std::cout << "Simulator - map exists: " << map_exists << std::endl;
    return map_exists;
}

// step through one time step
// TODO: use input actions, assuming in order, and vectors same size
std::vector<CarObs> StandaloneSimulator::step(std::vector<double> velocities, std::vector<double> steering_angles) {
    if (!map_exists) {
        // std::cout << "Simulator - Map is not set, update map first." << std::endl;
    }
    // std::cout << "Simulator - agents size: " << agents.size() << std::endl;
    // action input
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].set_velocity(velocities[i]);
        agents[i].set_steering_angle(steering_angles[i]);
        // std::cout << "Simulator - agent accel: " << agents[i].get_accel() << std::endl;
        // std::cout << "Simulator - agent desired ang: " << steering_angles[i] << std::endl;
    }
    // std::cout << "Simulator - agents action input set." << std::endl;
    // update all agent poses
    for (size_t i=0; i<agents.size(); i++) {
        agents[i].update_pose();
        agent_poses[i] = agents[i].get_pose();
        // std::cout << "Simulator - poses x: " << agent_poses[i].x << ", y: " << agent_poses[i].y << std::endl;
    }
    // std::cout << "Simulator - agents pose updated." << std::endl;

    // update opponent poses for each agent
    // TODO: make >2 case work in the future
    // if (agents.size() > 1) {
    //     for (size_t i=0; i<agents.size(); i++) {
    //         std::vector<Pose2D> opponent_poses(agent_poses.size()-1);
    //         std::copy(agent_poses.cbegin(), agent_poses.cbegin()+i, opponent_poses.begin());
    //         std::copy(agent_poses.cbegin()+i+1, agent_poses.cend(), opponent_poses.begin()+i);
    //         agents[i].update_op_poses(opponent_poses);
    //     }
    // }

    // update opponent poses for each agent, only 2 atm
    std::vector<Pose2D> ego_pose, op_pose;
    ego_pose.push_back(agent_poses[0]);
    op_pose.push_back(agent_poses[1]);
    agents[0].update_op_poses(op_pose);
    agents[1].update_op_poses(ego_pose);

    // std::cout << "Simulator - agent opponent poses updated." << std::endl;

    std::vector<CarObs> all_obs;
    // update scan for all agents
    for (size_t i=0; i<agents.size(); i++) {
        CarObs agent_obs = agents[i].update_scan();
        all_obs.push_back(agent_obs);
    }
    // check collision between agents
    bool collision = check_collision();
    // if collision between agents, change observation in_collision to true and collisan_angle to specific number
    if (collision) {
        for (size_t i=0; i<agents.size(); i++) {
            all_obs[i].in_collision = true;
            all_obs[i].collision_angle = -100; // specific number
        }
    }
    current_obs.clear();
    current_obs = all_obs;
    // std::cout << "Simulator - Done stepping." << std::endl;
    return all_obs;
}

// check collision between agents, assume only 2 now
// TODO: need to check pair-wise collision
bool StandaloneSimulator::check_collision() {
    // assume only two agents at the moment
    Pose2D ego_pose = agents[0].get_pose();
    Pose2D op_pose = agents[1].get_pose();
    double ego_x = ego_pose.x;
    double ego_y = ego_pose.y;
    double op_x = op_pose.x;
    double op_y = op_pose.y;

    // check safety bubble intersection
    if (sqrt(std::pow((ego_x-op_x),2) + std::pow((ego_y-op_y), 2)) > safety_radius) {
        return false; // far away enough to not do gjk
    } else {
        // gjk, create bounding boxes for ego and op
        // get transformation matrices for two car poses
        Eigen::Matrix4d op_trans_mat = get_transformation_matrix(op_pose);
        Eigen::Matrix4d ego_trans_mat = get_transformation_matrix(ego_pose);
        // bounding boxes in car frame, same for all cars
        Eigen::Vector4d rear_left_homo, rear_right_homo, front_left_homo, front_right_homo;
        // rear_left_homo << 0.0, car_width/2, 0.0, 1.0;
        rear_left_homo << -car_length/2, car_width/2, 0.0, 1.0;
        // rear_right_homo << 0.0, -car_width/2, 0.0, 1.0;
        rear_right_homo << -car_length/2, -car_width/2, 0.0, 1.0;
        // front_left_homo << scan_distance_to_base_link, car_width/2, 0.0, 1.0;
        front_left_homo << car_length/2, car_width/2, 0.0, 1.0;
        // front_right_homo << scan_distance_to_base_link, -car_width/2, 0.0, 1.0;
        front_right_homo << car_length/2, -car_width/2, 0.0, 1.0;
        // transform bounding boxes - ego
        Eigen::Vector4d ego_rear_left_transformed = ego_trans_mat*rear_left_homo;
        Eigen::Vector4d ego_rear_right_transformed = ego_trans_mat*rear_right_homo;
        Eigen::Vector4d ego_front_left_transformed = ego_trans_mat*front_left_homo;
        Eigen::Vector4d ego_front_right_transformed = ego_trans_mat*front_right_homo;
        ego_rear_left_transformed = ego_rear_left_transformed / ego_rear_left_transformed(3);
        ego_rear_right_transformed = ego_rear_right_transformed / ego_rear_right_transformed(3);
        ego_front_left_transformed = ego_front_left_transformed / ego_front_left_transformed(3);
        ego_front_right_transformed = ego_front_right_transformed / ego_front_right_transformed(3);
        // transform bounding boxes - op
        Eigen::Vector4d op_rear_left_transformed = op_trans_mat*rear_left_homo;
        Eigen::Vector4d op_rear_right_transformed = op_trans_mat*rear_right_homo;
        Eigen::Vector4d op_front_left_transformed = op_trans_mat*front_left_homo;
        Eigen::Vector4d op_front_right_transformed = op_trans_mat*front_right_homo;
        op_rear_left_transformed = op_rear_left_transformed / op_rear_left_transformed(3);
        op_rear_right_transformed = op_rear_right_transformed / op_rear_right_transformed(3);
        op_front_left_transformed = op_front_left_transformed / op_front_left_transformed(3);
        op_front_right_transformed = op_front_right_transformed / op_front_right_transformed(3);
        // get vertices
        vec2 ego_rl{-ego_rear_left_transformed(1), ego_rear_left_transformed(0)};
        vec2 ego_rr{-ego_rear_right_transformed(1), ego_rear_right_transformed(0)};
        vec2 ego_fl{-ego_front_left_transformed(1), ego_front_left_transformed(0)};
        vec2 ego_fr{-ego_front_right_transformed(1), ego_front_right_transformed(0)};
        std::vector<vec2> ego_vertices{ego_rl, ego_rr, ego_fr, ego_fl};
        vec2 op_rl{-op_rear_left_transformed(1), op_rear_left_transformed(0)};
        vec2 op_rr{-op_rear_right_transformed(1), op_rear_right_transformed(0)};
        vec2 op_fl{-op_front_left_transformed(1), op_front_left_transformed(0)};
        vec2 op_fr{-op_front_right_transformed(1), op_front_right_transformed(0)};
        std::vector<vec2> op_vertices{op_fl, op_rl, op_rr, op_fr};
        // get counts
        // size_t count1 = 4;
        // size_t count2 = 4;
        // check gjk
        // for (size_t i=0; i<ego_vertices.size(); i++) {
        //     std::cout << "current ego vertex: (" << ego_vertices[i].x << ", " << ego_vertices[i].y << ")" << std::endl;
        // }
        // for (size_t j=0; j<op_vertices.size(); j++) {
        //     std::cout << "current op vertex: (" << op_vertices[j].x << ", " << op_vertices[j].y << ")" << std::endl;
        // }
        int collision = gjk(op_vertices, ego_vertices);

        // std::cout << "gjk collision: " << collision << std::endl;
        return static_cast<bool>(collision);
    }
}

// reset the simulator
void StandaloneSimulator::reset() {
    // num_agents = 0;
    // agents.clear();
    // agent_poses.clear();
    current_obs.clear();
    // map.clear();
    // map_height = 0;
    // map_width = 0;
    // map_resolution = 0.0;
    // origin_x = 0.0;
    // origin_y = 0.0;
    // free_threshold = 0.0;
    // map_exists = false;
    for (RaceCar &agent : agents) {
        agent.reset();
    }
}

// reset the simulator by poses
void StandaloneSimulator::reset_bypose(std::vector<Pose2D> &poses) {
    current_obs.clear();
    for (size_t i=0; i<poses.size(); i++) {
        agents[i].reset_bypose(poses[i]);
    }
}


// utils
Eigen::Matrix4d StandaloneSimulator::get_transformation_matrix(const Pose2D &pose) {
    // get transformation matrix car frame to global frame
    double x = pose.x;
    double y = pose.y;
    double theta = pose.theta;
    double cosine = std::cos(theta);
    double sine = std::sin(theta);
    Eigen::Matrix4d T;
    T << cosine, -sine, 0.0, x, sine, cosine, 0.0, y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return T;
}