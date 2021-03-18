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

from PIL import Image

class Colab(object):

    # car constants
    CAR_LENGTH = 0.58
    CAR_WIDTH = 0.31

    MIN_BATCH = 100

    def __init__(self, map_path, map_extension, num_agents, start_poses=None):

        def load_binary(filename):
            with open(filename, 'rb') as f:
                return f.read()

        def load_html(filename):
            with open(filename, 'r') as f:
                return f.read()

        def get_bytes(image):
          _, image_buffer_array = cv2.imencode(".jpg", np.array(image))
          byte_image = image_buffer_array.tobytes()
          return byte_image

        self.map_path = map_path
        self.map_extension = map_extension
        self.load_map()

        # scale car dimensions accordingly
        self.car_length = self.CAR_LENGTH / self.map_resolution
        self.car_width = self.CAR_WIDTH / self.map_resolution

        html_code = load_html('/content/f1tenth_gym/gym/f110_gym/envs/colab.html')
        map_image_binary = get_bytes(self.map_image)
        car_replace_tag = 'insert_cars_here'
        image_replace_tag = '"insert_binary_image_here"'
        html_code = html_code.replace(car_replace_tag, ''.join([f'<div class="car" id="car-{i}"></div>' for i in range(num_agents)])) 
        html_code = html_code.replace("{","{{")
        html_code = html_code.replace("}","}}")
        html_code = html_code.replace(image_replace_tag,"{map_image_binary}")
        html_code = html_code.format(map_image_binary=map_image_binary)
        html_code = html_code.replace('btoa(b', 'btoa(')
        # and start the display
        display(IPython.display.HTML(html_code))

        # batch poses together
        self.batch_poses = []
        self.frame_counter = 0

    def update_cars(self, p_x, p_y, p_t):
        self.batch_poses.append([self.frame_counter, self.adjust_car_poses(p_x, p_y, p_t)])
        self.frame_counter += 1
        if len(self.batch_poses) >= self.MIN_BATCH:
            js_code = '''
            const senderChannel = new BroadcastChannel('channel');
            senderChannel.postMessage({poses});
            '''.format(poses=self.batch_poses)
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
        map_image_array = np.array(Image.open(self.map_path + self.map_extension))
        # crop whitespace bordering map
        crop_horizontal = ~np.all(map_image_array == 255, axis=1)
        crop_vertical = ~np.all(map_image_array == 255, axis=0)
        self.map_image = Image.fromarray((map_image_array[crop_horizontal])[:, crop_vertical])
        self.crop_offset = np.argmax(crop_vertical), np.argmax(crop_horizontal)

    def adjust_car_poses(self, poses_x, poses_y, poses_theta):
        poses = [[x, y] for x, y in zip(poses_x, poses_y)] # organise by car
        poses_offset = np.array(poses) - self.map_origin[:2]
        poses_scaled = poses_offset / self.map_resolution
        poses_cropped = poses_scaled - self.crop_offset
        poses_inverted = poses_cropped * [1, -1] + [0, self.map_image.size[1]]
        return [[x, y, t] for (x, y), t in zip(poses_inverted, poses_theta)]