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
from numba.np.extensions import cross2d
import jax.numpy as jnp
from jax import jit
import jax


@njit(cache=True)
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
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
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
    return b * ac - a * bc


@njit(cache=True)
def avgPoint(vertices):
    """
    Return the average point of multiple vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (np.ndarray, (2,)): average point of the vertices
    """
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """
    Return the index of the vertex furthest away along a direction in the list of vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
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


@njit(cache=True)
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
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a

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


@njit(cache=True)
def collision_multiple(vertices):
    """
    Check pair-wise collisions for all provided vertices

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): all vertices for checking pair-wise collision

    Returns:
        collisions (np.ndarray (num_vertices, )): whether each body is in collision
        collision_idx (np.ndarray (num_vertices, )): which index of other body is each index's body is in collision, -1 if not in collision
    """
    collisions = np.zeros((vertices.shape[0],))
    collision_idx = -1 * np.ones((vertices.shape[0],))
    # looping over all pairs
    for i in range(vertices.shape[0] - 1):
        for j in range(i + 1, vertices.shape[0]):
            # check collision
            vi = np.ascontiguousarray(vertices[i, :, :])
            vj = np.ascontiguousarray(vertices[j, :, :])
            ij_collision = collision(vi, vj)
            # fill in results
            if ij_collision:
                collisions[i] = 1.0
                collisions[j] = 1.0
                collision_idx[i] = j
                collision_idx[j] = i

    return collisions, collision_idx

@jit
def collision_multiple_map(vertices, pixel_centers):
    """
    Check vertices collision with map occupancy
    Rasters car polygon to map occupancy
    vmap across number of cars, and number of occupied pixels

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): agent rectangle vertices, ccw winding order
        pixel_centers (np.ndarray (HxW, 2)): x, y position of pixel centers of map image

    Returns:
        collisions (np.ndarray (num_bodies, )): whether each body is in collision with map
    """
    edges = jnp.roll(vertices, 1, axis=1) - vertices
    center_p = pixel_centers[:, None, None] - edges
    cross_prods = jnp.cross(center_p, edges)
    left_of = jnp.where(cross_prods <= 0, 1.0, 0.0)
    all_left_of = jnp.sum(left_of, axis=-1)
    collisions = jnp.where(jnp.sum(jnp.where(all_left_of == 4.0, 1.0, 0.0), axis=0) > 0.0, 1.0, 0.0)
    return collisions



@jit
def collision_multiple_map_jaxloop(vertices, pixel_centers):
    """
    Check vertices collision with map occupancy
    Rasters car polygon to map occupancy
    JAX impl is about twice faster than the Numba impl

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): agent rectangle vertices, ccw winding order
        pixel_centers (np.ndarray (HxW, 2)): x, y position of pixel centers of map image

    Returns:
        collisions (np.ndarray (num_bodies, )): whether each body is in collision with map
    """
    collisions = jnp.zeros((vertices.shape[0],))
    # check if center of pixel to the LEFT of all 4 edges
    # loop because vectorizing is way slower
    for car_ind in range(vertices.shape[0]):
        left_of = jnp.empty((4, pixel_centers.shape[0]))
        for v_ind in range(-1, 3):
            edge = vertices[car_ind, v_ind + 1] - vertices[car_ind, v_ind]
            center_p = pixel_centers - vertices[car_ind, v_ind]
            left_of = left_of.at[v_ind + 1, :].set((jnp.cross(center_p, edge) <= 0))
        ls = jnp.any((jnp.sum(left_of, axis=0) == 4.0))
        collisions = collisions.at[car_ind].set(jnp.where(ls, 1.0, 0.0))

    return collisions


@njit(cache=True)
def collision_multiple_map_nb(vertices, pixel_centers):
    """
    Check vertices collision with map occupancy
    Rasters car polygon to map occupancy

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): agent rectangle vertices, ccw winding order
        pixel_centers (np.ndarray (HxW, 2)): x, y position of pixel centers of map image

    Returns:
        collisions (np.ndarray (num_bodies, )): whether each body is in collision with map
    """
    collisions = np.zeros((vertices.shape[0],))
    # check if center of pixel to the LEFT of all 4 edges
    # loop because vectorizing is way slower
    for car_ind in range(vertices.shape[0]):
        left_of = np.empty((4, pixel_centers.shape[0]))
        for v_ind in range(-1, 3):
            edge = vertices[car_ind, v_ind + 1] - vertices[car_ind, v_ind]
            center_p = pixel_centers - vertices[car_ind, v_ind]
            left_of[v_ind + 1, :] = cross2d(center_p, edge) <= 0
        ls = np.any((np.sum(left_of, axis=0) == 4.0))
        collisions[car_ind] = np.where(ls, 1.0, 0.0)
    return collisions


"""
Utility functions for getting vertices by pose and shape
"""


@njit(cache=True)
def get_trmtx(pose):
    """
    Get transformation matrix of vehicle frame -> global frame

    Args:
        pose (np.ndarray (3, )): current pose of the vehicle

    return:
        H (np.ndarray (4, 4)): transformation matrix
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = np.cos(th)
    sin = np.sin(th)
    H = np.array(
        [
            [cos, -sin, 0.0, x],
            [sin, cos, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return H


@njit(cache=True)
def get_vertices(pose, length, width):
    """
    Utility function to return vertices of the car body given pose and size

    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width

    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    H = get_trmtx(pose)
    rl = H.dot(np.asarray([[-length / 2], [width / 2], [0.0], [1.0]])).flatten()
    rr = H.dot(np.asarray([[-length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    fl = H.dot(np.asarray([[length / 2], [width / 2], [0.0], [1.0]])).flatten()
    fr = H.dot(np.asarray([[length / 2], [-width / 2], [0.0], [1.0]])).flatten()
    rl = rl / rl[3]
    rr = rr / rr[3]
    fl = fl / fl[3]
    fr = fr / fr[3]
    vertices = np.asarray(
        [[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]]
    )
    return vertices
