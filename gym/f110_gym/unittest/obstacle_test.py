import unittest
import numpy as np
import os, csv 



class F110Env:
    def __init__(self):
        self.map_path = 'example_map'
    

    def load_centerline(self, file_name=None):
        """
        Loads a centerline from a csv file. 
        Note: the file must be in the same folder as the map which the simulator loads.

        Args:
            file_name (string): the name of a csv file with the centerline of the track in the form [x_i, y_i, w_l_i, w_r_i], location and width

        Returns:
            center_pts (np.ndarray): the loaded center point location
            widths (np.ndarray): the widths of the track
        """
        if file_name is None:
            # map_path = os.path.splitext(self.map_path)[0]
            file_name = self.map_path + '_centerline.csv'
        else:
            file_name = self.map_path + file_name

        track = []
        with open(file_name, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            for lines in csvFile:  
                track.append(lines)
        track = np.array(track)

        center_pts = track[:, 0:2]
        widths = track[:, 2:4]

        return center_pts, widths

    def add_obstacles(self, n_obstacles, obstacle_size=[0.5, 0.5]):
        """
        Adds a set number of obstacles to the envioronment using the track centerline. 
        Note: this function requires a csv file with the centerline points in it which can be loaded. 
        Updates the renderer and the map kept by the laser scaner for each vehicle in the simulator

        Args:
            n_obstacles (int): number of obstacles to add
            obstacle_size (list(2)): rectangular size of obstacles
            
        Returns:
            None
        """
        # map_img = np.copy(self.empty_map_img)
        # scan_sim = self.sim.agents[0].scan_simulator

        obs_size_m = np.array(obstacle_size)
        # obs_size_px = np.array(obs_size_m / scan_sim.map_resolution, dtype=int)
        
        center_pts, widths = self.load_centerline()

        # randomly select certain idx's
        rand_idxs = np.random.randint(1, len(center_pts)-1, n_obstacles)
        
        # randomly select location within box of minimum_width around the center point
        rands = np.random.uniform(-1, 1, size=(n_obstacles, 2))
        obs_locations = center_pts[rand_idxs, :] + rands * widths[rand_idxs]

        # change the values of the img at each obstacle location
        obs_locations = np.array(obs_locations)

        return obs_locations

    


class ObstacleTest(unittest.TestCase):
    def setUp(self):
        self.env = F110Env()

    def test_obstacle_locations(self):
        known_locations = np.array([[-28.0539, 19.5926],
                                    [23.8548, 137877],
                                    [31.5787, 20.7230],
                                    [-51.7873, -4.8519],
                                    [-32.5076, -6.9832]])

        np.random.seed(1234)
        generated_locations = self.env.add_obstacles(5)
        self.assertEquals(known_locations.all(), generated_locations.all())


if __name__ == '__main__':
    unittest.main()