from pyglet.gl import GL_POINTS
import pyglet
import numpy as np
import math
from globals import *
import matplotlib.pyplot as plt


class FollowTheGapPlanner:
    """
    Example Planner
    """
   
 

    def __init__(self, speed_fraction = 1):
    
        print("Controller initialized")
        # self.currentPosition = np.array([0.0, 0.0])
        # self.nextPosition = np.array([0.0, 0.0])
        # self.nextPositions = [[0.0, 0.0]]

        # self.drawn_waypoints = []
        self.lidar_border_points = 1080 * [[0,0]]
        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)
        self.simulation_index = 0
        self.speed_fraction = speed_fraction
        self.plot_lidar_data = False
        self.draw_lidar_data = False
        self.lidar_visualization_color = (0, 0, 0)
        self.lidar_live_gaps = []

        self.vertex_list = pyglet.graphics.vertex_list(2,
        ('v2i', (10, 15, 30, 35)),
        ('c3B', (0, 0, 255, 0, 255, 0))
    )



    def render_ftg(self, e):


        if not self.draw_lidar_data: return

        # Render lidar data
        # for i in range(len(self.lidar_border_points)):
        #     e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.lidar_border_points[i][0], self.lidar_border_points[i][1], 0.]),
        #                 ('c3B', (255, 0, 255)))

        self.vertex_list.delete()
        
        scaled_points = np.array(self.lidar_border_points)
        howmany = scaled_points.shape[0]
        scaled_points_flat = scaled_points.flatten()

        self.vertex_list = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat), ('c3B', self.lidar_visualization_color * howmany ))





    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']

        points = []
        angles = []
        distances = []
        self.lidar_border_points = []

        # Take into account size of car
        scans = [x - 0.3 for x in ranges]

        # Use all sensor data
        # for i in range(1080):
        #     p1 = car_state[0] + scans[i] * math.cos(car.scan_angles[i] + car_state[4])
        #     p2 = car_state[1] + scans[i] * math.sin(car.scan_angles[i] + car_state[4])
        #     planner.lidar_border_points.append([50* p1, 50* p2])

        # Use only a part
        max_dist = 0
        for i in range(20, 88): # Only front cone, not all 180 degree (use 0 to 108 for all lidar points)
            index = 10*i # Only use every 10th lidar point

            p1 = pose_x + scans[index] * math.cos(self.lidar_scan_angles[index] + pose_theta)
            p2 = pose_y + scans[index] * math.sin(self.lidar_scan_angles[index] + pose_theta)
            points.append((p1,p2))
            angles.append(self.lidar_scan_angles[index])
            distances.append(scans[index])
            self.lidar_border_points.append([50* p1, 50* p2])
            self.lidar_live_points.append([50* p1, 50* p2])
            if( scans[index] > max_dist):
                max_dist = scans[index]

        angles_unfiltered = angles.copy()
        distances_unfiltered = distances.copy()
        

        closest_distance = 10000
        closest_distance_index = 0

        # Filter distances

        # ignore close points:
        for i in range(len(distances)):

            if(distances[i] < 1.5):
                distances[i] = 0

            if (distances[i] > 6):
                distances[i] = 6

            # Set points near closest distance to 0
            if(distances[i] < closest_distance):
                closest_distance =distances[i]
                closest_distance_index = i


        # IGNORE neighbors of closest point
        for i in range(closest_distance_index - 3, closest_distance_index + 3):
            if( i < len(distances)):
                distances[i] = 0

        # Find gaps
        gaps = []
        gap_open = False
        gap_opening_angle = 0
        gap_starting_index = 0
        gap_treshold = 1.499
        max_distance = 0
        gap_found = False
  

        for i in range(len(distances) - 1):
            # Rising
            if(not gap_open):
                if(distances[i] < distances[i+1] - gap_treshold):
                    gap_opening_angle = angles[i+1]  # + math.sin(0.05) * distances[i]
                    gap_starting_index = i+1
                    gap_open = True
             

            # Falling
            if(gap_open):
                if(max_distance < distances[i]):
                     max_distance = distances[i]

                if(distances[i] > distances[i+1] + gap_treshold ):
                    gap_closing_angle = angles[i] #- math.sin(0.05) * distances[i]
                    gap_closing_index = i
                    gap = [gap_opening_angle,  gap_closing_angle, gap_starting_index, gap_closing_index]
                    gaps.append(gap)
                    gap_open = False
                    gap_found = True
          
        self.lidar_live_gaps = gaps

        # Find largest Gap
        largest_gap_angle = 0
        largest_gap_index = 0
        largest_gap_center = 0
        for i in range(len(gaps)):
            gap = gaps[i]
            gap_angle = abs(gap[1] - gap[0])
            if(gap_angle) > largest_gap_angle:
                largest_gap_angle = gap_angle
                largest_gap_index = i
                largest_gap_center = (gap[0] + gap[1]) / 2



        # Speed Calculation

        speed = self.speed_fraction * max_distance - 3 * abs(largest_gap_center)
        if(speed < 0.1): speed = 0.1 #Dont stand still

        if max_distance > 8:
            speed = 14

        # Emergency Brake
        if(not gap_found):
            speed = 0.0
            print("Emergency Brake")

        # print("Speed", speed)


        # Plotting
        if(self.plot_lidar_data):
            if(self.simulation_index % 10 == 0):
                plt.clf()
                plt.title("Lidar Data")
                plt.plot(angles, distances)
                plt.plot(angles_unfiltered, distances_unfiltered)

                for gap in gaps:
                    plt.axvline(x=gap[0], color='k', linestyle='--')
                    plt.axvline(x=gap[1], color='k', linestyle='--')

                plt.axvline(x=largest_gap_center, color='red', linestyle='--')

                plt.savefig("lidar.png")


        steering_angle = largest_gap_center
        
        self.simulation_index += 1
        return speed, steering_angle


        

