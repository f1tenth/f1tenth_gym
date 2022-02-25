import numpy as np


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