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

#include <cstddef>
#include <vector>
#include <cmath>

#include "distance_transform.hpp"

// Implementation based on the paper
// Distance Transforms of Sampled Functions
// Pedro F. Felzenszwalb and Daniel P. Huttenlocher
// http://people.cs.uchicago.edu/~pff/papers/dt.pdf

using namespace racecar_simulator;

void DistanceTransform::distance_squared_1d(
    const std::vector<double> & input, 
    std::vector<double> & output) {

  // Each parabola has the form
  //
  //     input[index] + (query - index)^2

  // The grid location of the i-th parabola is parabola_idxs[i]
  std::vector<size_t> parabola_idxs(input.size());
  parabola_idxs[0] = 0;

  // The range of the i-th parabola is
  // parabola_boundaries[i] to parabola_boundaries[i + 1]
  // Initialize all of the ranges to extend to infinity
  std::vector<double> parabola_boundaries(input.size() + 1);
  parabola_boundaries[0] = -inf;
  parabola_boundaries[1] = inf;

  // The number of parabolas in the lower envelope
  int num_parabolas = 0;

  // Compute the lower envelope over all grid cells
  double intersection_point;
  for (size_t idx = 1; idx < input.size(); idx++) {

    num_parabolas++;

    do {
      num_parabolas--;
      // The location of the rightmost parabola in the lower envelope
      int parabola_idx = parabola_idxs[num_parabolas];

      // Compute the intersection point between the current and rightmost parabola
      // by solving for the intersection point, p:
      //
      // input[idx] + (p - idx)^2 = input[parabolaIdx] - (p - parabolaIdx)^2
      //
      intersection_point = (
              (input[idx] + idx * idx)
              -
              (input[parabola_idx] + parabola_idx * parabola_idx))
              /
              (2 * (idx - parabola_idx));

      // If the intersection point is before the boundary,
      // the rightmost parabola is not actually part of the
      // lower envelope, so decrease k and repeat.
    } while (intersection_point <= parabola_boundaries[num_parabolas]);

    // Move to the next parabola
    num_parabolas ++;

    parabola_idxs[num_parabolas] = idx;
    parabola_boundaries[num_parabolas] = intersection_point;
    parabola_boundaries[num_parabolas+1] = inf;
  }

  int parabola = 0;
  for (size_t idx = 0; idx < input.size(); idx++) {
    // Find the parabola corresponding to the index
    while (parabola_boundaries[parabola + 1] < idx) parabola++;

    // Compute the value of the parabola
    int idx_dist = idx - parabola_idxs[parabola];
    output[idx] = idx_dist * idx_dist + input[parabola_idxs[parabola]];
  }
}

void DistanceTransform::distance_squared_2d(
    std::vector<double> & input, 
    size_t width, 
    size_t height, 
    double boundary_value) {

  // Transform along the columns
  std::vector<double> col_vec(height + 2);
  std::vector<double> col_dt(height + 2);
  for (size_t col = 0; col < width; col++) {
    col_vec[0] = boundary_value;
    col_vec[height + 1] = boundary_value;
    for (size_t row = 0; row < height; row++) {
      col_vec[row + 1] = input[row * width + col];
    }
    distance_squared_1d(col_vec, col_dt);
    for (size_t row = 0; row < height; row++) {
      input[row * width + col] = col_dt[row + 1];
    }
  }

  // Transform along the rows
  std::vector<double> row_vec(width + 2);
  std::vector<double> row_dt(width + 2);
  for (size_t row = 0; row < height; row++) {
    row_vec[0] = boundary_value;
    row_vec[width + 1] = boundary_value;
    for (size_t col = 0; col < width; col++) {
      row_vec[col + 1] = input[row * width + col];
    }
    distance_squared_1d(row_vec, row_dt);
    for (size_t col = 0; col < width; col++) {
      input[row * width + col] = row_dt[col + 1];
    }
  }
}

void DistanceTransform::distance_2d(
    std::vector<double> & input, 
    size_t width, 
    size_t height, 
    double resolution,
    double boundary_value) {

  distance_squared_2d(input, width, height, boundary_value);
  for (size_t i = 0; i < input.size(); i++) {
    input[i] = resolution * sqrt(input[i]);
  }
}
