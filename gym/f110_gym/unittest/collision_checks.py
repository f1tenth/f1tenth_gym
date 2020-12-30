# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Prototype of Utility functions and GJK algorithm for Collision checks between vehicles
Originally from https://github.com/kroitor/gjk.c
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit

@njit
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector

    Args:
        pt (np.ndarray, (2,)): input vector

    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1*temp
    return pt


@njit
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors

    Args:
        a, b, c (np.ndarray, (2,)): input vectors

    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b*ac - a*bc


@njit
def avgPoint(vertices):
    """
    Return the average point of multiple vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (np.ndarray, (2,)): average point of the vertices
    """
    return np.sum(vertices, axis=0)/vertices.shape[0]


@njit
def indexOfFurthestPoint(vertices, d):
    """
    Return the index of the vertex furthest away along a direction in the list of vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit
def support(vertices1, vertices2, d):
    """
    Minkowski sum support function for GJK

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
        d (np.ndarray, (2, )): direction to find the support along

    Returns:
        support (np.ndarray, (n, 2)): Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b-a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b-a
        ac = c-a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False


"""
Unit tests for GJK collision checks
Author: Hongrui Zheng
"""

import time
import unittest

class CollisionTests(unittest.TestCase):
    def setUp(self):
        # test params
        np.random.seed(1234)

        # Collision check body
        self.vertices1 = np.asarray([[4,11.],[5,5],[9,9],[10,10]])
        

    def test_random_collision(self):
        # perturb the body by a small amount and make sure it all collides with the original body
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            self.assertTrue(collision(a,b))

    def test_fps(self):
        # also perturb the body but mainly want to test GJK speed
        start = time.time()
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            collision(a, b)
        elapsed = time.time() - start
        fps = 1000/elapsed
        print('fps:', fps)
        self.assertTrue(fps>500)

if __name__ == '__main__':
    unittest.main()