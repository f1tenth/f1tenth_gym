import numpy as np
import math

def column(matrix, i):
        return [row[i] for row in matrix]
        

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    return history


def squared_distance(p1, p2):
    squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
    return squared_distance


def get_distance_from_point_to_points(new_point, point_cloud):
    min_squared_dist = 10000000

    new_points = len(point_cloud) * [new_point]
    point_cloud = np.array(point_cloud)
    distances = (point_cloud - new_points ) **2
    distances[:, 0] = distances[:, 0] + distances[:, 1]
    distances = distances[:,0]
    min_squared_dist = np.min(distances)

    # Old: forloop instead of np
    # for point in point_cloud:
    #     squared_dist = (point[0]- new_point[0])**2 + ( point[1] - new_point[1])**2
    #     if(squared_dist < min_squared_dist):
    #         min_squared_dist = squared_dist
    
    return math.sqrt(min_squared_dist)


