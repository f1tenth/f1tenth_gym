from globals import *

import math
import numpy as np


#Scale of the race track

def squared_distance(p1, p2):
    squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
    return squared_distance

'''
Represents the "real track", calculated by the l2race server
'''
class Track:

    def __init__(self):
        
        # track_info = np.load('racing/tracks/{}_info.npy'.format(TRACK_NAME), allow_pickle=True).item()
        # print("track_info", track_info)

        waypoints = np.loadtxt( './example_waypoints.csv', delimiter=';', skiprows= 3)

        self.waypoints_x = waypoints[:,1]
        self.waypoints_y = waypoints[:,2]

        # self.AngleNextCheckpointEast =   track_info['AngleNextCheckpointEast']
        # self.AngleNextCheckpointRelative =   track_info['AngleNextCheckpointRelative']


        self.track_name = TRACK_NAME
        self.waypoints = [[self.waypoints_x[i], self.waypoints_y[i]] for i in range(len(self.waypoints_x))]
        self.initial_position = self.waypoints[0]



    
    def distance_to_track(self, p):
        min_distance = 100000
        waypoint_index = 0
        for i in range(len(self.waypoints_x)):
            waypoint = [self.waypoints_x[i], self.waypoints_y[i]]
            dist = squared_distance(p, waypoint)
            if(dist < min_distance):
                min_distance = dist
                waypoint_index = i
        print("waypoint index", waypoint_index)        
        return math.sqrt(min_distance)

    def get_closest_index(self, p):

        min_distance = 100000
        waypoint_index = 0
        for i in range(len(self.waypoints_x)):
            waypoint = [self.waypoints_x[i], self.waypoints_y[i]]
            dist = squared_distance(p, waypoint)
            if(dist < min_distance):
                min_distance = dist
                waypoint_index = i
        return waypoint_index
    

    def distance_to_waypoint(self, p, waypoint_index):
        waypoint = [self.waypoints_x[waypoint_index], self.waypoints_y[waypoint_index]]
        dist = squared_distance(p, waypoint)
        return math.sqrt(dist)

   

    def draw_track(self):
        plt.scatter(self.waypoints_x,self.waypoints_y)

        interpolation_length = 100
        w_x = self.waypoints_x
        w_y = self.waypoints_y

        plt.savefig("track.png")



