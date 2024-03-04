import numpy as np
from numba import njit


@njit(cache=True)
def perpendicular(pt):
    """Return a 2-vector's perpendicular vector

    Parameters
    ----------
    pt : np.ndarray
        input vector

    Returns
    -------
    np.ndarray
        perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def tripleProduct(a, b, c):
    """Return triple product of three vectors

    Parameters
    ----------
    a : np.ndarray
        input vector
    b : np.ndarray
        input vector
    c : np.ndarray
        input vector

    Returns
    -------
    np.ndarray
        triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def avgPoint(vertices):
    """Return the average point of multiple vertices

    Parameters
    ----------
    vertices : np.ndarray
        the vertices we want to find avg on

    Returns
    -------
    np.ndarray
        average point of the vertices
    """
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """Return the index of the vertex furthest away along a direction in the list of vertices

    Parameters
    ----------
    vertices : np.ndarray
        the vertices we want to find index on
    d : np.ndarray
        direction

    Returns
    -------
    int
        index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """Minkowski sum support function for GJK

    Parameters
    ----------
    vertices1 : np.ndarray
        vertices of the first body
    vertices2 : np.ndarray
        vertices of the second body
    d : np.ndarray
        direction to find the support along

    Returns
    -------
    np.ndarray
        Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def collision(vertices1, vertices2):
    """GJK test to see whether two bodies overlap

    Parameters
    ----------
    vertices1 : np.ndarray
        vertices of the first body
    vertices2 : np.ndarray
        vertices of the second body

    Returns
    -------
    boolean
        True if two bodies collide
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
    """Check pair-wise collisions for all provided vertices

    Parameters
    ----------
    vertices : np.ndarray
        all vertices for checking pair-wise collision

    Returns
    -------
    collisions : np.ndarray
        whether each body is in collision
    collision_idx : np.ndarray
        which index of other body is each index's body is in collision, -1 if not in collision
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


@njit(cache=True)
def get_trmtx(pose):
    """Get transformation matrix of vehicle frame -> global frame

    Parameters
    ----------
    pose : np.ndarray
        current pose of the vehicle

    Returns
    -------
    np.ndarray
        transformation matrix
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
    """Utility function to return vertices of the car body given pose and size

    Parameters
    ----------
    pose : np.ndarray
        current world coordinate pose of the vehicle
    length : float
        car length
    width : float
        car width

    Returns
    -------
    np.ndarray
        corner vertices of the vehicle body
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
