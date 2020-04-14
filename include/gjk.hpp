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
#include <iostream>
#include <vector>
//https://github.com/kroitor/gjk.c/blob/master/gjk.c

// namespace racecar_simulator {

struct _vec2{double x; double y;};
typedef struct _vec2 vec2;

// vector arithmetic
vec2 subtract(vec2 &a, vec2 &b);
vec2 negate(vec2 &v);
vec2 perpendicular(vec2 &v);
double dotProduct(vec2 &a, vec2 &b);
double lengthSquared(vec2 &v);

// triple product used to calculate perpendicular normal vectors
vec2 tripleProduct(vec2 &a, vec2 &b, vec2 &c);

// compute average center(roughly)
vec2 averagePoint(std::vector<vec2> &vertices);

// get furthest vertex along a direction
size_t indexOfFurthestPoint(std::vector<vec2> &vertices, vec2 &d);

//Minkowski sum support func for GJK
vec2 support(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2, vec2 &d);

// GJK yes/no test
int gjk(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2);

double Perturbation();
vec2 Jostle(vec2 &a);
// }