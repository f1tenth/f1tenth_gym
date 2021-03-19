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
import IPython
import numpy as np

class Colab(object):

    # static variable for unique broadcast channel
    CHANNEL_ID = "CH0"

    # car constants
    CAR_LENGTH = 0.58
    CAR_WIDTH = 0.31

    # batch size for sending poses
    MIN_BATCH = 200

    def get_id(self):
      return self.CHANNEL_ID

    def __init__(self, map_path, map_extension, num_agents, start_poses=None):

        def load_html(filename):
            with open(filename, 'r') as f:
                return f.read()

        def get_bytes(image_array):
          _, image_buffer_array = cv2.imencode(".png", image_array)
          byte_image = image_buffer_array.tobytes()
          return byte_image

        # assign unique channel to this class instance (then increment)
        self.channel_id = Colab.CHANNEL_ID
        Colab.CHANNEL_ID = Colab.CHANNEL_ID[:2] + str(1 + int(Colab.CHANNEL_ID[2:]))

        # load map
        self.map_path = map_path
        self.map_extension = map_extension
        self.load_map()

        # scale car dimensions accordingly
        self.car_length = self.CAR_LENGTH / self.map_resolution
        self.car_width = self.CAR_WIDTH / self.map_resolution

        self.start_poses = start_poses

        # load in HTML code as string
        html_code = load_html('/content/f1tenth_gym/gym/f110_gym/envs/colab.html')
        map_image_binary = get_bytes(self.map_image_array)
        # substitute in all runtime variables as strings
        html_code = html_code.replace("{","{{")
        html_code = html_code.replace("}","}}")
        html_code = html_code.replace('insert_channel_here', self.channel_id)
        html_code = html_code.replace('"insert_cars_here"', ''.join([f'<div class="car" id="car-{i}"></div>' for i in range(num_agents)]))
        html_code = html_code.replace('"insert_car_width_here"', str(self.car_width))
        html_code = html_code.replace('"insert_car_length_here"', str(self.car_length))
        html_code = html_code.replace('"insert_binary_image_here"',"{map_image_binary}")
        html_code = html_code.replace('"insert_start_poses_here"', str(self.adjust_car_poses(*self.start_poses)))
        html_code = html_code.format(map_image_binary=map_image_binary)
        html_code = html_code.replace('btoa(b', 'btoa(')
        # and start the display
        display(IPython.display.HTML(html_code))

        # batch poses together
        self.batch_poses = []
        self.frame_counter = 0

    def update_cars(self, p_x, p_y, p_t, done):
        self.batch_poses.append([self.frame_counter, self.adjust_car_poses(p_x, p_y, p_t)])
        self.frame_counter += 1
        if (len(self.batch_poses) >= self.MIN_BATCH) or done:
            js_code = '''
            const senderChannel = new BroadcastChannel("{channel_id}");
            senderChannel.postMessage({poses});
            '''.format(poses=self.batch_poses, channel_id=self.channel_id)
            display(IPython.display.Javascript(js_code))
            self.batch_poses = []

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