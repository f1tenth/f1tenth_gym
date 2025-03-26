import unittest

import numpy as np
from f1tenth_gym.envs.track import cubic_spline


class TestCubicSpline(unittest.TestCase):
    def test_calc_curvature(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the curvature at the four corners of the circle
        # The curvature of a circle is 1/radius
        self.assertAlmostEqual(track.calc_curvature(0), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi / 2), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(3 * np.pi / 2), 1, places=3)

    def test_calc_yaw(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the yaw at the four corners of the circle
        # The yaw of a circle is s + pi/2
        self.assertAlmostEqual(track.calc_yaw(0), np.pi / 2, places=2)
        self.assertAlmostEqual(track.calc_yaw(np.pi / 2), np.pi, places=2)
        self.assertAlmostEqual(track.calc_yaw(np.pi), 3 * np.pi / 2, places=2)
        self.assertAlmostEqual(track.calc_yaw(3 * np.pi / 2), 0, places=2)

    def test_calc_position(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the position at the four corners of the circle
        # The position of a circle is (x, y) = (cos(s), sin(s))
        self.assertTrue(
            np.allclose(track.calc_position(0), np.array([1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi / 2), np.array([0, 1]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi), np.array([-1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(
                track.calc_position(3 * np.pi / 2), np.array([0, -1]), atol=1e-3
            )
        )

    def test_calc_arclength(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength(1, 0, 0)[0], 0, places=2)
        self.assertAlmostEqual(track.calc_arclength(0, 1, 0)[0], np.pi / 2, places=2)
        self.assertAlmostEqual(
            track.calc_arclength(-1, 0, np.pi / 2)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength(0, -1, np.pi)[0], 3 * np.pi / 2, places=2
        )

    def test_calc_arclength_inaccurate(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSpline2D(circle_x, circle_y)
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength_inaccurate(1, 0)[0], 0, places=2)
        self.assertAlmostEqual(
            track.calc_arclength_inaccurate(0, 1)[0], np.pi / 2, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength_inaccurate(-1, 0)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength_inaccurate(0, -1)[0], 3 * np.pi / 2, places=2
        )
