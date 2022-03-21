from globals import *
from util import *
import math
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom
import time



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

        self.segments = []
        self.line_strings = []


    
    def get_ordered_list(points, x, y):
        points.sort(key = lambda p: (p.x - x)**2 + (p.y - y)**2)
        return points

    def update_line_strings(self):
        self.line_strings = []
        filtered_segments = self.segments
        for segment in filtered_segments:
            if(len(segment) < 3): continue
            line = geom.LineString(segment)
            self.line_strings.append(line)


    def add_new_lidar_points_to_segments(self, points):

        # start = time.time()
        
        # return
        d_treshold = 0.8
        for point in points:

            connected_segment_index = -1
            segment_index = 0
            new_segment = True
            combine_segments = []

            for segment in self.segments:

                closest_dist = get_distance_from_point_to_points(point, segment)

               

                # Avoid taking into account already measured points
                if(closest_dist < 0.2): 
                    new_segment = False
                    segment_index+=1
                    continue

                if(closest_dist < d_treshold):
                    segment.append(point)
                    # if new point already belongs to a segment, but also mathes another one, we can combine the segments
                    if(not new_segment):
                        if(segment_index != connected_segment_index):
                            combine_segments.append([connected_segment_index,segment_index]) 
                    connected_segment_index = segment_index
                    new_segment = False
                segment_index+=1
            if(new_segment):
                segment = [point]
                self.segments.append(segment)

            # Combine segments
            i = 0
            for c in combine_segments:
                self.segments[c[0] - i] += self.segments[c[1] -i ]
                self.segments.pop(c[1] - i)
                i += 1



        if DRAW_TRACK_SEGMENTS: 
            # print("segments", self.segments)
            plt.clf()
            # plt.xlim(-6, 6)
            # plt.ylim(-2, 10)
            for segment in self.segments:
                x_val = [x[0] for x in segment]
                y_val = [x[1] for x in segment]
                plt.plot(x_val,y_val,'o')

            # plt.show()
            plt.savefig("Myfile.png",format="png")

      
        self.update_line_strings()
    
        # end = time.time()
        # print("Combine linestrings", end - start)

    
    
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



