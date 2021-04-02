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
Rendering of F1Tenth Gym environment on Google Colab through JavaScript
Author: Eoin Gogarty
"""

# imports
import os
import yaml
import cv2
import time
import IPython
import numpy as np

class Colab(object):

    CHANNEL_ID = "CH0"          # static variable for unique broadcast channel
    BATCH_SEND_INTERVAL = 0.25  # timing interval for sending poses
    JS_FRAME_RATE = 60          # frame rate of Javascript display
    MAX_BATCH_SIZE = 100        # keep this small to prevent any JS delays
    
    # this is the minimum number of poses that need to be sent every second
    MIN_BATCH_SIZE = int(np.ceil(BATCH_SEND_INTERVAL * JS_FRAME_RATE * 2))

    def __init__(self, map_path, map_extension, num_agents, start_poses, car_dimensions, timestep):

        def load_html(filename):
            with open(filename, 'r') as f:
                return f.read()

        def get_bytes(image_array):
          _, image_buffer_array = cv2.imencode(".png", image_array)
          byte_image = image_buffer_array.tobytes()
          return byte_image

        # assign unique channel to this class instance (then increment)
        self.channel_id = Colab.CHANNEL_ID
        self.channel_id_extra = -1
        Colab.CHANNEL_ID = Colab.CHANNEL_ID[:2] + str(1 + int(Colab.CHANNEL_ID[2:]))

        # load map
        self.map_path = map_path
        self.map_extension = map_extension
        self.load_map()

        # load in HTML code as string
        html_code = load_html(os.path.dirname(os.path.abspath(__file__)) + '/colab.html')
        # get map as binary image array
        map_image_binary = get_bytes(self.map_image_array)
        # substitute in all runtime variables as strings
        html_code = html_code.replace("{","{{")
        html_code = html_code.replace("}","}}")
        html_code = html_code.replace('insert_channel_here', self.channel_id)
        html_code = html_code.replace('"insert_cars_here"', ''.join([f'<div class="car" id="car-{i}"></div>' for i in range(num_agents)]))
        html_code = html_code.replace('"insert_binary_image_here"',"{map_image_binary}")
        html_code = html_code.format(map_image_binary=map_image_binary)
        html_code = html_code.replace('btoa(b', 'btoa(')
        self.html_code = html_code
        # and start the display
        self.start(start_poses, car_dimensions)

        # initialise timers for updating visuals
        self.timestep = timestep
        self.last_send = time.time()

    def start(self, start_poses, car_dimensions):
        # make a temporary copy of main HTML
        html_code = self.html_code
        # scale car dimensions accordingly
        self.car_width = car_dimensions[0] / self.map_resolution
        self.car_length = car_dimensions[1] / self.map_resolution
        html_code = html_code.replace('"insert_car_width_here"', str(self.car_width))
        html_code = html_code.replace('"insert_car_length_here"', str(self.car_length))
        # reset starting poses
        self.start_poses = start_poses
        html_code = html_code.replace('"insert_start_poses_here"', str(self.adjust_car_poses(*self.start_poses)))
        # batch poses together
        self.batch_poses = []
        self.frame_counter = 1
        # append extra ID to channel ID
        self.channel_id_extra += 1
        html_code = html_code.replace(self.channel_id, self.channel_id + '-' + str(self.channel_id_extra))
        # start display
        display(IPython.display.HTML(html_code))

    def update_cars(self, p_x, p_y, p_t, done, mode):
        self.batch_poses.append(self.adjust_car_poses(p_x, p_y, p_t))
        # set to True to run one loop
        done_flag = True
        while done_flag and (len(self.batch_poses) > 0):
            # if simulation is done, then run loop till no poses
            done_flag = done
            # send poses at given time interval
            if (time.time() - self.last_send) > self.BATCH_SEND_INTERVAL:
                self.last_send = time.time()
                # check for mode
                if mode == "human":
                    self.batch_size = self.MAX_BATCH_SIZE
                    self.preprocess_max_length = int(self.batch_size / (self.timestep * self.JS_FRAME_RATE))
                else:
                    self.batch_size = self.MIN_BATCH_SIZE
                    self.preprocess_max_length = len(self.batch_poses)
                # convert to realtime at 60fps
                batch_poses = self.batch_poses[:self.preprocess_max_length]
                batch_poses = self.convert_poses_realtime(batch_poses)
                # enumerate list with frame indices
                batch_poses = [[i + self.frame_counter, poses] for i, poses in enumerate(batch_poses)]
                # increment frame counter
                self.frame_counter += len(batch_poses)
                js_code = '''
                console.log("Sending poses on {channel_id} [{frames}]");
                const senderChannel = new BroadcastChannel("{channel_id}");
                senderChannel.postMessage({poses});
                '''.format(poses=batch_poses,
                        frames=self.frame_counter,
                        channel_id=(self.channel_id + '-' + str(self.channel_id_extra)))
                display(IPython.display.Javascript(js_code))
                # remove sent poses
                self.batch_poses = self.batch_poses[self.preprocess_max_length:]

    def load_map(self):
        # load map config
        config_ext = '.yaml' if os.path.isfile(self.map_path + '.yaml') else '.yml'
        with open(self.map_path + config_ext, 'r') as stream:
            self.map_config = yaml.safe_load(stream)
            self.map_resolution = self.map_config['resolution']
            self.map_origin = self.map_config['origin']
        # load map image
        map_image_array = cv2.imread(self.map_path + self.map_extension, 0)
        # crop whitespace bordering map
        crop_horizontal = ~np.all(map_image_array == 255, axis=1)
        crop_vertical = ~np.all(map_image_array == 255, axis=0)
        self.map_image_array = (map_image_array[crop_horizontal])[:, crop_vertical]
        self.crop_offset = np.argmax(crop_vertical), np.argmax(crop_horizontal)

    def adjust_car_poses(self, poses_x, poses_y, poses_theta):
        poses = [[x, y] for x, y in zip(poses_x, poses_y)] # organise by car
        poses_offset = np.array(poses) - self.map_origin[:2]
        poses_scaled = poses_offset / self.map_resolution
        poses_cropped = poses_scaled - self.crop_offset
        # check for theta overflow and have to negate angle (not sure why)
        poses_theta = [- t % np.pi for t in poses_theta]
        return [[x, y, t] for (x, y), t in zip(poses_cropped, poses_theta)]

    def convert_poses_realtime(self, batch_poses):
        # temporary array for new mapped values
        altered_poses = [-1] * self.batch_size
        # fractional indices for mapping
        mapping = np.linspace(0, self.batch_size - 1, len(batch_poses))
        # for each index in new array, find nearest element in old array
        for i in range(self.batch_size):
            nearest = self.find_nearest(mapping, i)
            altered_poses[i] = batch_poses[nearest]
        # return stretched/dilated poses
        return altered_poses

    def find_nearest(self, array, value):
        # return index of element in array that is nearest to value
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))):
            return idx - 1
        else:
            return idx
