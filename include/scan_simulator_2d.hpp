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

#pragma once

#include <random>

#include "pose_2d.hpp"

namespace racecar_simulator {

class ScanSimulator2D {

  private:

    // Laser settings
    int num_beams;
    double field_of_view;
    double scan_std_dev;
    double angle_increment;

    // Ray tracing settings
    double ray_tracing_epsilon;

    // The distance transform
    double resolution;
    size_t width, height;
    Pose2D origin;
    std::vector<double> dt;

    // Static output vector
    std::vector<double> scan_output;

    // Noise generation
    std::mt19937 noise_generator;
    std::normal_distribution<double> noise_dist;

    // Precomputed constants
    double origin_c;
    double origin_s;
    int theta_discretization;
    double theta_index_increment;

  public:
    std::vector<double> sines;
    std::vector<double> cosines;

    ScanSimulator2D() {}

    ScanSimulator2D(
        int num_beams_, 
        double field_of_view_, 
        double scan_std_dev_, 
        double ray_tracing_epsilon=0.0001,
        int theta_discretization=2000);

    void set_map(
        const std::vector<double> & map, 
        size_t height, 
        size_t width, 
        double resolution,
        const Pose2D & origin,
        double free_threshold);
    void set_map(const std::vector<double> &map, double free_threshold);

    void scan(const Pose2D & pose, double * scan_data);
    const std::vector<double> scan(const Pose2D & pose);

    double distance_transform(double x, double y) const;

    double trace_ray(double x, double y, double theta_index) const;

    void xy_to_row_col(double x, double y, int * row, int * col) const;
    int row_col_to_cell(int row, int col) const;
    int xy_to_cell(double x, double y) const;

    double get_field_of_view() const {return field_of_view;}
    double get_angle_increment() const {return angle_increment;}
    int get_theta_discret() const {return theta_discretization;}
    int get_num_beams() const {return num_beams;}
};

}
