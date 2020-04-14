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

// zmq stuff
#include "zhelpers.hpp"
#include <string>

// sim stuff
#include "simulator.hpp"

// serialization stuff
#include "sim_requests.pb.h"
#include <unistd.h>

// general
#include <stdexcept>
#include <string>
#include <iostream>
#include <math.h>

using namespace racecar_simulator;
// arguments in are [executable, timestep, number_of_agents, port number]
int main(int argc, char const *argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    // TODO: add more args

    // Read arguments
    std::string ts_str = argv[1];
    std::string num_agents_str = argv[2];
    std::string port_num_str = argv[3];
    std::string mu_str = argv[4];
    std::string h_cg_str = argv[5];
    std::string l_r_str = argv[6];
    std::string cs_f_str = argv[7];
    std::string cs_r_str = argv[8];
    std::string I_z_str = argv[9];
    std::string mass_str = argv[10];
    double ts, mu, h_cg, l_r, cs_f, cs_r, I_z, mass;
    int num_agents;
    int port_num;
    try {
        std::size_t ts_pos;
        std::size_t num_agents_pos;
        std::size_t port_num_pos;
        std::size_t mu_pos;
        std::size_t h_cg_pos;
        std::size_t l_r_pos;
        std::size_t cs_f_pos;
        std::size_t cs_r_pos;
        std::size_t I_z_pos;
        std::size_t mass_pos;
        ts = std::stod(ts_str, &ts_pos);
        mu = std::stod(mu_str, &mu_pos);
        h_cg = std::stod(h_cg_str, &h_cg_pos);
        l_r = std::stod(l_r_str, &l_r_pos);
        cs_f = std::stod(cs_f_str, &cs_f_pos);
        cs_r = std::stod(cs_r_str, &cs_r_pos);
        I_z = std::stod(I_z_str, &I_z_pos);
        mass = std::stod(mass_str, &mass_pos);
        num_agents = std::stoi(num_agents_str, &num_agents_pos);
        port_num = std::stoi(port_num_str, &port_num_pos);
        if (ts_pos < ts_str.size() || num_agents_pos < num_agents_str.size() || port_num_pos < port_num_str.size()) {
            std::cerr << "Sim server - Trailing characters after number: " << '\n';
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Sim server - Invalid number: " << '\n';
    } catch (std::out_of_range const &ex) {
        std::cerr << "Sim server - Number out of range: " << '\n';
    }

    // std::cout << "Sim server - Num agents: " << num_agents << std::endl;
    // std::cout << "Sim server - Port number: " << port_num << std::endl;
    // Instantiate simulator instance
    StandaloneSimulator sim(num_agents, ts, mu, h_cg, l_r, cs_f, cs_r, I_z, mass);
    // std::cout << "Sim server - Sim initialized." << std::endl;

    // default
    int ego_idx = 0;

    // Instantiate zmq context
    zmq::context_t context(1);

    // prepare socket for req-rep
    zmq::socket_t socket(context, ZMQ_PAIR);
    socket.connect("tcp://localhost:"+std::to_string(port_num));

    while (1) {
        // std::cout << "Sim server - map_exists is: " << sim.map_exists << std::endl;
        zmq::message_t message;
        // receive request
        socket.recv(&message);
        std::string smessage(static_cast<char*>(message.data()), message.size());
        // if kill
        if (smessage == "dead") {
            context.close();
            socket.close();
            std::cout << "Killing Sim Server..." << std::endl;
            return 0;
        }
        // deserailization
        racecar_simulator_standalone::SimRequest sim_request_proto;
        bool success = sim_request_proto.ParseFromString(smessage);
        // TODO: what if failure
        int request_type = sim_request_proto.type();
        // std::cout << "Sim server - request type: " << request_type << std::endl;
        // update map or step sim
        if (request_type == 0) {
            // step
            if (!sim.get_map_status()) {
                std::cout << "Sim server - Map not set for sim, skipping request" << std::endl;
                // Create step fail response proto
                racecar_simulator_standalone::SimResponse sim_response_proto;
                sim_response_proto.set_type(1);
                racecar_simulator_standalone::StepResponse *step_fail_ptr = sim_response_proto.mutable_step_result();
                step_fail_ptr->set_result(1);
                std::string sim_response_string;
                // Serialization
                sim_response_proto.SerializeToString(&sim_response_string);
                zmq::message_t step_result(sim_response_string.size());
                memcpy(step_result.data(), sim_response_string.data(), sim_response_string.size());
                // Sending response
                socket.send(step_result);
            } else {
                // std::cout << "Sim server - Stepping simulator." << std::endl;
                std::vector<double> requested_vel;
                for (auto &vel : sim_request_proto.step_request().requested_vel()) {
                    requested_vel.push_back(vel);
                }
                std::vector<double> requested_ang;
                for (auto &ang : sim_request_proto.step_request().requested_ang()) {
                    requested_ang.push_back(ang);
                }
                std::vector<CarObs> observations = sim.step(requested_vel, requested_ang);
                // Create response proto with observation
                racecar_simulator_standalone::SimResponse sim_response_proto;
                sim_response_proto.set_type(0);
                racecar_simulator_standalone::SimObservation *sim_obs_ptr = sim_response_proto.mutable_sim_obs();
                sim_obs_ptr->set_ego_idx(ego_idx);
                // Loop through observations vector and push back to proto
                for (size_t i=0; i<observations.size(); i++) {
                    racecar_simulator_standalone::CarObservation *car_obs_proto = sim_obs_ptr->add_observations();
                    CarObs current_obs = observations[i];
                    // fill in scan
                    *car_obs_proto->mutable_scan() = {current_obs.scan.begin(), current_obs.scan.end()};
                    // fill in pose
                    car_obs_proto->set_pose_x(current_obs.pose.x);
                    car_obs_proto->set_pose_y(current_obs.pose.y);
                    car_obs_proto->set_theta(current_obs.pose.theta);
                    // fill in velocities
                    car_obs_proto->set_linear_vel_x(current_obs.odom.linear_x);
                    car_obs_proto->set_linear_vel_y(current_obs.odom.linear_y);
                    car_obs_proto->set_ang_vel_z(current_obs.odom.angular_z);
                    // fill in collision info
                    car_obs_proto->set_collision(current_obs.in_collision);
                    car_obs_proto->set_collision_angle(current_obs.collision_angle);
                }

                // Serialization
                std::string sim_response_string;
                sim_response_proto.SerializeToString(&sim_response_string);
                zmq::message_t step_result(sim_response_string.size());
                memcpy(step_result.data(), sim_response_string.data(), sim_response_string.size());
                // Send full obs response
                socket.send(step_result);
                // std::cout << "Sim server - sent back step result." << std::endl;
            }
        } else if (request_type == 1) {
            // update map
            std::vector<double> map;
            for (auto &pixel : sim_request_proto.map_request().map()) {
                map.push_back(pixel);
            }
            double origin_x = sim_request_proto.map_request().origin_x();
            double origin_y = sim_request_proto.map_request().origin_y();
            double map_resolution = sim_request_proto.map_request().map_resolution();
            double free_threshold = sim_request_proto.map_request().free_threshold();
            int map_height = sim_request_proto.map_request().map_height();
            int map_width = sim_request_proto.map_request().map_width();

            sim.set_map(map, map_height, map_width, map_resolution, origin_x, origin_y, free_threshold);
            // Create map update response proto
            racecar_simulator_standalone::SimResponse sim_response_proto;
            sim_response_proto.set_type(2);
            racecar_simulator_standalone::MapResponse *map_response_ptr = sim_response_proto.mutable_map_result();
            map_response_ptr->set_result(0);
            // serialization
            std::string sim_response_string;
            sim_response_proto.SerializeToString(&sim_response_string);
            zmq::message_t set_map_response(sim_response_string.size());
            memcpy(set_map_response.data(), sim_response_string.data(), sim_response_string.size());
            // send set map response
            socket.send(set_map_response);
        } else if (request_type == 2) {
            // reset sim
            sim.reset();
            // Create reset response proto
            racecar_simulator_standalone::SimResponse sim_response_proto;
            sim_response_proto.set_type(3);
            racecar_simulator_standalone::SimResetResponse *reset_response_ptr = sim_response_proto.mutable_reset_resp();
            reset_response_ptr->set_result(0);
            // serialization
            std::string sim_response_string;
            sim_response_proto.SerializeToString(&sim_response_string);
            zmq::message_t reset_response(sim_response_string.size());
            memcpy(reset_response.data(), sim_response_string.data(), sim_response_string.size());
            // send reset response
            socket.send(reset_response);
        } else if (request_type == 3) {
            // reset sim
            sim.reset();

            // grab params from proto
            double mu = sim_request_proto.update_request().mu();
            double h_cg = sim_request_proto.update_request().h_cg();
            double l_r = sim_request_proto.update_request().l_r();
            double cs_f = sim_request_proto.update_request().cs_f();
            double cs_r = sim_request_proto.update_request().cs_r();
            double I_z = sim_request_proto.update_request().i_z();
            double mass = sim_request_proto.update_request().mass();

            // update sim params
            sim.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass);

            // send back response
            racecar_simulator_standalone::SimResponse sim_response_proto;
            sim_response_proto.set_type(4);
            racecar_simulator_standalone::UpdateParamResponse *update_response_ptr = sim_response_proto.mutable_update_resp();
            update_response_ptr->set_result(0);
            // serialization
            std::string update_response_string;
            sim_response_proto.SerializeToString(&update_response_string);
            zmq::message_t update_response(update_response_string.size());
            memcpy(update_response.data(), update_response_string.data(), update_response_string.size());
            socket.send(update_response);

        } else if (request_type == 4) {
            // grap poses
            std::vector<double> car_x;
            for (auto &x : sim_request_proto.reset_bypose_request().car_x()) {
                car_x.push_back(x);
            }
            std::vector<double> car_y;
            for (auto &y : sim_request_proto.reset_bypose_request().car_y()) {
                car_y.push_back(y);
            }
            std::vector<double> car_theta;
            for (auto &theta : sim_request_proto.reset_bypose_request().car_theta()) {
                car_theta.push_back(theta);
            }
            std::vector<Pose2D> poses;
            for (size_t i=0; i < car_x.size(); i++) {
                Pose2D current_pose {.x=car_x[i], .y=car_y[i], .theta=car_theta[i]};
                poses.push_back(current_pose);
            }
            // reset sim by pose
            sim.reset_bypose(poses);

            // send back response
            // Create reset response proto
            racecar_simulator_standalone::SimResponse sim_response_proto;
            sim_response_proto.set_type(3);
            racecar_simulator_standalone::SimResetResponse *reset_response_ptr = sim_response_proto.mutable_reset_resp();
            reset_response_ptr->set_result(0);
            // serialization
            std::string sim_response_string;
            sim_response_proto.SerializeToString(&sim_response_string);
            zmq::message_t reset_response(sim_response_string.size());
            memcpy(reset_response.data(), sim_response_string.data(), sim_response_string.size());
            // send reset response
            socket.send(reset_response);

        }
    }
    return 0;
}