#pragma once

#include <vector>
#include <limits>

namespace racecar_simulator {

class DistanceTransform {

private:

    static constexpr double inf = std::numeric_limits<double>::infinity();
    
public:
    DistanceTransform(size_t max_size);

    static void distance_squared_1d(
        const std::vector<double> & input,
        std::vector<double> & output);

    static void distance_squared_2d(
        std::vector<double> & input,
        size_t width,
        size_t height,
        double boundary_value=0);

    static void distance_2d(
        std::vector<double> & input,
        size_t width,
        size_t height,
        double resolution=1,
        double boundary_value=0);
};

}
