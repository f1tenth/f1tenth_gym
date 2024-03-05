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
from f1tenth_gym.envs.laser_models import ScanSimulator2D


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
        self.sample_scans = np.load(f"{wdir}/legacy_scan.npz")

    def _test_map_scan(self, map_name: str, debug=False):
        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        new_scan = np.empty((self.num_test, self.num_beams))
        scan_sim.set_map(map=map_name)
        # scan gen loop
        for i in range(self.num_test):
            test_pose = self.test_poses[i]
            new_scan[i, :] = scan_sim.scan(pose=test_pose, rng=scan_rng)
        diff = self.sample_scans[map_name] - new_scan
        mse = np.mean(diff**2)

        if debug:
            # plotting
            import matplotlib.pyplot as plt

            theta = np.linspace(-self.fov / 2.0, self.fov / 2.0, num=self.num_beams)
            plt.polar(theta, new_scan[1, :], ".", lw=0)
            plt.polar(theta, self.sample_scans[map_name][1, :], ".", lw=0)
            plt.show()

        self.assertLess(mse, 2.0)


    def test_map_spielberg(self, debug=False):
        self._test_map_scan("Spielberg", debug=debug)

    def test_map_monza(self, debug=False):
        self._test_map_scan("Monza", debug=debug)

    def test_map_austin(self, debug=False):
        self._test_map_scan("Austin", debug=debug)

    def test_fps(self):
        # scan fps should be greater than 500
        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        scan_sim.set_map(map="Spielberg")

        import time

        start = time.time()
        for i in range(10000):
            x_test = i / 10000
            scan_sim.scan(pose=np.array([x_test, 0.0, 0.0]), rng=scan_rng)
        end = time.time()
        fps = 10000 / (end - start)

        self.assertGreater(fps, 500.0)
