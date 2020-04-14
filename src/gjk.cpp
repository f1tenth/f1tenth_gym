// MIT License

// Implementation based on: https://github.com/kroitor/gjk.c/blob/master/gjk.c

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


#include "gjk.hpp"
// using namespace racecar_simulator;

// vector arithmetic
vec2 subtract(vec2 &a, vec2 &b) {
    vec2 sub;
    sub.x = a.x - b.x;
    sub.y = a.y - b.y;
    return sub;
}

vec2 negate(vec2 &v) {
    vec2 v_n;
    v_n.x = -v.x;
    v_n.y = -v.y;
    return v_n;
}

vec2 perpendicular(vec2 &v) {
    vec2 p = {v.y, -v.x};
    return p;
}

double dotProduct(vec2 &a, vec2 &b) {
    return a.x*b.x+a.y*b.y;
}

double lengthSquared(vec2 &v) {
    return v.x*v.x + v.y*v.y;
}

// triple product used to calculate perpendicular normal vectors
vec2 tripleProduct(vec2 &a, vec2 &b, vec2 &c) {
    vec2 r;
    double ac = a.x*c.x + a.y*c.y; // perform a.dot(c)
    double bc = b.x*c.x + b.y*c.y; // perform b.dot(c)
    // perform b*a.dot(c) - a*b.dot(c)
    r.x = b.x*ac - a.x*bc;
    r.y = b.y*ac - a.y*bc;
    return r;
}

// compute average center(roughly)
vec2 averagePoint(std::vector<vec2> &vertices) {
    size_t count = vertices.size();
    vec2 avg = {0., 0.};
    for (size_t i=0; i<count; i++) {
        avg.x += vertices[i].x;
        avg.y += vertices[i].y;
    }
    avg.x /= count;
    avg.y /= count;
    return avg;
}

// get furthest vertex along a direction
size_t indexOfFurthestPoint(std::vector<vec2> &vertices, vec2 &d) {
    size_t count = vertices.size();
    double maxProduct = dotProduct(d, vertices[0]);
    size_t index = 0;
    for (size_t i=1; i<count; i++) {
        double product = dotProduct(d, vertices[i]);
        if (product > maxProduct) {
            maxProduct = product;
            index = i;
        }
    }
    return index;
}

//Minkowski sum support func for GJK
vec2 support(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2, vec2 &d) {
    vec2 d_n = negate(d);
    // furthest point of first body along an arbitrary direction
    size_t i = indexOfFurthestPoint(vertices1, d);
    // furthest point of second body along the opposite direction
    size_t j = indexOfFurthestPoint(vertices2, d_n);
    // subtract (Minkowski sum) the two points to see if bodies overlap
    vec2 support = subtract(vertices1[i], vertices2[j]);
    return support;
}

// GJK yes/no test
int gjk(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2) {
    size_t index = 0; // index of current vertex of simplex
    vec2 a, b, c, d, ao, ab, ac, abperp, acperp;
    std::vector<vec2> simplex(3);

    vec2 position1 = averagePoint(vertices1);
    vec2 position2 = averagePoint(vertices2);
    // std::cout << "average point 1: (" << position1.x << ", " << position1.y << ")" << std::endl;
    // std::cout << "average point 2: (" << position2.x << ", " << position2.y << ")" << std::endl;

    // initial direction from the center of 1st body to the center of 2nd body
    d = subtract(position1, position2);
    // std::cout << "direction: (" << d.x << ", " << d.y << ")" << std::endl;

    // if initial direction is zero - set to any arbitrary axis (choose x here)
    if ((d.x == 0) && (d.y == 0)) {
        d.x = 1.;
    }

    // set the first support as initial point of the new simplex
    a = support(vertices1, vertices2, d);
    simplex[0] = a;

    // for (size_t k=0; k<simplex.size(); k++) {
    //     std::cout << simplex[k].x << ", " << simplex[k].y << std::endl;
    // }

    if (dotProduct(a, d) <= 0) {
        // std::cout << dotProduct(a, d) << std::endl;
        // std::cout << a.x << ", " << a.y << "; " << d.x << ", " << d.y << std::endl;
        return 0; // no collision
    }

    d = negate(a); // the next search direction is always towards the origin, so d = -a
    int iter_count = 0;
    while (1) {
        iter_count++;
        a = support(vertices1, vertices2, d);
        simplex[++index] = a;
        if (dotProduct(a, d) <= 0) {
            return 0; // no collision
        }
        ao = negate(a); // from point A to origin is just -a
        
        // simplex has 2 points (a line segment, not a triangle yet)
        if (index < 2) {
            b = simplex[0];
            ab = subtract(b, a); // from point A to B
            d = tripleProduct(ab, ao, ab); // normal to AB towards origin
            if (lengthSquared(d) == 0) {
                d = perpendicular(ab);
            }
            continue; // skip to next iter
        }

        b = simplex[1];
        c = simplex[0];
        ab = subtract(b, a); // from point A to B
        ac = subtract(c, a); // from point A to C

        acperp = tripleProduct(ab, ac, ac);

        if (dotProduct(acperp, ao) >= 0) {
            d = acperp; // new direction is normal to AC towards origin
        } else {
            abperp = tripleProduct(ac, ab, ab);
            if (dotProduct(abperp, ao) < 0) {
                return 1; // collision
            }
            simplex[0] = simplex[1]; // swap first element (point C)
            d = abperp; // new direction is normal to AB towards origin
        }
        simplex[1] = simplex[2]; // swap element in the middle (point B)
        --index;
    }
    
    return 0;
}

double Perturbation();
vec2 Jostle(vec2 a);