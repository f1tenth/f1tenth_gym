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
Prototype of Utility functions and classes for simulating 2D LIDAR scans
Author: Hongrui Zheng
"""

import os
import unittest

import numpy as np
from f110_gym.envs.laser_models import ScanSimulator2D


class ScanTests(unittest.TestCase):
    def setUp(self):
        # test params
        self.num_beams = 1080
        self.fov = 4.7

        self.num_test = 10
        self.test_poses = np.zeros((self.num_test, 3))
        self.test_poses[:, 2] = np.linspace(-1.0, 1.0, num=self.num_test)

        # legacy gym data
        wdir = os.path.dirname(os.path.abspath(__file__))
        sample_scan = np.load(f"{wdir}/legacy_scan.npz")
        self.berlin_scan = sample_scan["berlin"]
        self.skirk_scan = sample_scan["skirk"]

    def test_map_berlin(self, debug=False):
        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        new_berlin = np.empty((self.num_test, self.num_beams))
        scan_sim.set_map(map_name="Berlin")
        # scan gen loop
        for i in range(self.num_test):
            test_pose = self.test_poses[i]
            new_berlin[i, :] = scan_sim.scan(pose=test_pose, rng=scan_rng)
        diff = self.berlin_scan - new_berlin
        mse = np.mean(diff**2)
        # print('Levine distance test, norm: ' + str(norm))

        if debug:
            # plotting
            import matplotlib.pyplot as plt

            theta = np.linspace(-self.fov / 2.0, self.fov / 2.0, num=self.num_beams)
            plt.polar(theta, new_berlin[1, :], ".", lw=0)
            plt.polar(theta, self.berlin_scan[1, :], ".", lw=0)
            plt.show()

        self.assertLess(mse, 2.0)

    def test_map_skirk(self, debug=False):
        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        new_skirk = np.empty((self.num_test, self.num_beams))
        scan_sim.set_map(map_name="Skirk")
        print("map set")
        # scan gen loop
        for i in range(self.num_test):
            test_pose = self.test_poses[i]
            new_skirk[i, :] = scan_sim.scan(pose=test_pose, rng=scan_rng)
        diff = self.skirk_scan - new_skirk
        mse = np.mean(diff**2)
        print("skirk distance test, mse: " + str(mse))

        if debug:
            # plotting
            import matplotlib.pyplot as plt

            theta = np.linspace(-self.fov / 2.0, self.fov / 2.0, num=self.num_beams)
            plt.polar(theta, new_skirk[1, :], ".", lw=0)
            plt.polar(theta, self.skirk_scan[1, :], ".", lw=0)
            plt.show()

        self.assertLess(mse, 2.0)

    def test_fps(self):
        # scan fps should be greater than 500
        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        scan_sim.set_map(map_name="Skirk")

        import time

        start = time.time()
        for i in range(10000):
            x_test = i / 10000
            scan_sim.scan(pose=np.array([x_test, 0.0, 0.0]), rng=scan_rng)
        end = time.time()
        fps = 10000 / (end - start)

        self.assertGreater(fps, 500.0)
