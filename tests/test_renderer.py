import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    ObservationConfig,
    RenderConfig,
    SimulationConfig,
)
from f1tenth_gym.envs.f110_env import RenderClock
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.rendering import make_lidar_scan_callback, make_renderer
import gymnasium as gym

# rgb_array/human rendering uses the GL backend, which needs an X display
# (real or a virtual one via xvfb). Skip display-bound tests when there is none.
_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))


def _rgb_env(render_fps=60, real_time_factor=1.0, timestep=0.01):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
            simulation_config=SimulationConfig(timestep=timestep, integrator_timestep=timestep, max_laps=None),
            render_config=RenderConfig(
                render_fps=render_fps,
                real_time_factor=real_time_factor,
            ),
        ),
        render_mode="rgb_array",
    )


@unittest.skipUnless(_HAS_DISPLAY, "rendering needs an X display (run under xvfb)")
class TestRenderer(unittest.TestCase):
    def test_rgb_array_render(self):
        env =  gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE)),
            render_mode="rgb_array",
        )
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            frame = env.render()

            self.assertTrue(isinstance(frame, np.ndarray), "Frame is not a numpy array")
            self.assertTrue(len(frame.shape) == 3, "Frame is not a 3D array")
            self.assertTrue(frame.shape[2] == 3, "Frame does not have 3 channels")

        env.close()

        self.assertTrue(True, "rgb_array render test failed")

    def test_rgb_array_list(self):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE)),
            render_mode="rgb_array_list",
        )
        env.reset()

        steps = 100
        for _ in range(steps):
            action = env.action_space.sample()
            env.step(action)

        frame_list = env.render()

        self.assertTrue(
            isinstance(frame_list, list), "the returned object is not a list of frames"
        )
        self.assertTrue(
            len(frame_list) == steps + 1,
            "the returned list does not have the correct number of frames",
        )
        self.assertTrue(
            all([isinstance(frame, np.ndarray) for frame in frame_list]),
            "not all frames are numpy arrays",
        )
        self.assertTrue(
            all([len(frame.shape) == 3 for frame in frame_list]),
            "not all frames are 3D arrays",
        )
        self.assertTrue(
            all([frame.shape[2] == 3 for frame in frame_list]),
            "not all frames have 3 channels",
        )

        env.close()

    def test_rgb_array_frame_is_contiguous_square_uint8(self):
        """rgb_array frames must be C-contiguous, square (window_size), uint8 RGB."""
        env = _rgb_env()
        env.reset(seed=1)
        for _ in range(30):
            env.step(np.array([[0.0, 3.0]], dtype=np.float32))
            frame = env.render()
        h, w = frame.shape[:2]
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(frame.flags["C_CONTIGUOUS"], "frame must be contiguous for video encoders")
        self.assertEqual(h, w, "frame must be square (window_size), not aspect-stretched")
        env.close()

    def test_rgb_array_is_rgb_not_bgr(self):
        """The default red car must record as red (R channel), not blue (BGR bug)."""
        env = _rgb_env()
        env.reset(seed=1)
        for _ in range(30):
            env.step(np.array([[0.0, 3.0]], dtype=np.float32))
            frame = env.render()
        r = frame[:, :, 0].astype(int)
        b = frame[:, :, 2].astype(int)
        red_px = int(((r > 180) & (b < 130) & (frame[:, :, 1].astype(int) < 130)).sum())
        blue_px = int(((b > 180) & (r < 130)).sum())
        self.assertGreater(red_px, 0, "no red car pixels found")
        self.assertGreater(red_px, blue_px, "frame looks BGR (blue car) instead of RGB")
        env.close()

    def test_metadata_render_fps_not_aliased(self):
        """metadata is per-instance; render_fps must not alias across envs (RecordVideo fps)."""
        e1 = _rgb_env(timestep=0.01)
        e2 = _rgb_env(timestep=0.02)
        self.assertEqual(e1.unwrapped.metadata["render_fps"], 100)
        self.assertEqual(e2.unwrapped.metadata["render_fps"], 50)
        self.assertIsNot(e1.unwrapped.metadata, e2.unwrapped.metadata)
        e1.close()
        e2.close()

    def test_set_real_time_factor_runtime(self):
        env = _rgb_env(real_time_factor=1.0)
        self.assertEqual(env.unwrapped.real_time_factor, 1.0)
        env.unwrapped.set_real_time_factor(5.0)
        self.assertEqual(env.unwrapped.real_time_factor, 5.0)
        env.unwrapped.set_real_time_factor(float("inf"))
        self.assertEqual(env.unwrapped.real_time_factor, float("inf"))
        with self.assertRaises(ValueError):
            env.unwrapped.set_real_time_factor(0.0)
        with self.assertRaises(ValueError):
            env.unwrapped.set_real_time_factor(-2.0)
        env.close()

    def test_frame_is_new_cadence(self):
        """At render_fps=60 over 1.0 sim-second (timestep 0.01), ~60 distinct frames emit."""
        env = _rgb_env(render_fps=60, timestep=0.01)
        env.reset(seed=1)
        count = 0
        for _ in range(100):
            env.step(np.array([[0.0, 2.0]], dtype=np.float32))
            if env.unwrapped.frame_is_new:
                count += 1
        self.assertTrue(55 <= count <= 65, f"expected ~60 distinct frames, got {count}")
        env.close()

    def test_render_fps_property_matches_config(self):
        env = _rgb_env(render_fps=45)
        self.assertEqual(env.unwrapped.render_fps, 45)
        env.close()

    def test_window_size_controls_frame_shape(self):
        """RenderConfig.window_size drives the rgb_array frame / video resolution
        (proves the RenderSpec->RenderConfig migration is wired through)."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
                simulation_config=SimulationConfig(max_laps=None),
                render_config=RenderConfig(window_size=512, focus_on=None, show_lap_info=False),
            ),
            render_mode="rgb_array",
        )
        env.reset(seed=1)
        for _ in range(3):
            env.step(np.array([[0.0, 2.0]], dtype=np.float32))
            frame = env.render()
        self.assertEqual(frame.shape, (512, 512, 3))
        env.close()


class TestRenderConfigFields(unittest.TestCase):
    def test_render_config_exposes_visual_fields(self):
        """The RenderSpec visual fields are now on RenderConfig (RenderSpec retired)."""
        fields = set(RenderConfig().__dataclass_fields__)
        for f in ("window_size", "focus_on", "vehicle_palette", "show_wheels",
                  "render_map_img", "car_thickness", "bigger_car_when_map_centered",
                  "show_lap_info"):
            self.assertIn(f, fields)
        with self.assertRaises(ValueError):
            RenderConfig(window_size=0)


class TestLidarCallback(unittest.TestCase):
    def test_points_use_active_lr_and_the_full_lidar_transform(self):
        class Capture:
            def __init__(self, points):
                self.points = points

            def update(self, points):
                self.points = points

        class FakeRenderer:
            def __init__(self):
                self.params = SimpleNamespace(lr=0.2)
                self.obs = {
                    "agent_0": {
                        "scan": np.array([1.0], dtype=np.float32),
                        "std_state": np.array(
                            [2.0, -1.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0],
                            dtype=np.float32,
                        ),
                    }
                }
                self.capture = None

            def get_points_renderer(self, points, **_kwargs):
                self.capture = Capture(points)
                return self.capture

        lidar = LiDARConfig(
            num_beams=1,
            angle_min=0.0,
            angle_max=0.5,
            base_link_to_lidar_tf=(0.3, 0.1, 0.25),
        )
        renderer = FakeRenderer()
        callback = make_lidar_scan_callback("agent_0", lidar)
        callback(renderer)

        sensor_origin = np.array([1.9, -0.9])
        heading = np.pi / 2 + 0.25
        expected = sensor_origin + np.array([np.cos(heading), np.sin(heading)])
        np.testing.assert_allclose(
            renderer.capture.points[0], expected, atol=2.0e-7
        )


class TestNoDisplayGuidance(unittest.TestCase):
    """make_renderer must fail loudly with setup guidance when no display exists."""

    def test_rgb_array_without_display_raises_with_guidance(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            with self.assertRaises(RuntimeError) as ctx:
                # params/track are unused before the display check raises
                make_renderer(None, None, ["agent_0"], render_mode="rgb_array")
            self.assertIn("xvfb", str(ctx.exception))

    def test_render_mode_none_needs_no_display(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            renderer, _ = make_renderer(None, None, ["agent_0"], render_mode=None)
            self.assertIsNone(renderer)


class TestRenderClock(unittest.TestCase):
    """Unit tests for the pacing/fixed-fps clock (no display required)."""

    def test_fixed_fps_emission(self):
        clk = RenderClock(render_fps=60, real_time_factor=float("inf"), timestep=0.01)
        clk.reset()
        n = sum(clk.advance() for _ in range(100))
        self.assertTrue(55 <= n <= 65, f"expected ~60 frames/100 steps, got {n}")

    def test_coarse_timestep_emits_every_step(self):
        # timestep (0.05) coarser than frame period (1/60) -> emit every step, no backlog
        clk = RenderClock(render_fps=60, real_time_factor=1.0, timestep=0.05)
        clk.reset()
        self.assertTrue(all(clk.advance() for _ in range(20)))

    def test_first_frame_after_reset_emits(self):
        clk = RenderClock(render_fps=60, real_time_factor=1.0, timestep=0.01)
        clk.reset()
        self.assertTrue(clk.frame_is_new)

    def test_pace_real_time(self):
        # rtf=1: pacing to sim_time should take ~ sim elapsed wall-clock
        clk = RenderClock(render_fps=60, real_time_factor=1.0, timestep=0.02)
        clk.reset()
        t0 = time.perf_counter()
        for i in range(11):
            clk.pace(i * 0.02)  # 0 .. 0.2 sim-seconds
        elapsed = time.perf_counter() - t0
        self.assertTrue(0.15 <= elapsed <= 0.45, f"rtf=1 pacing took {elapsed:.3f}s, expected ~0.2s")

    def test_pace_inf_no_sleep(self):
        clk = RenderClock(render_fps=60, real_time_factor=float("inf"), timestep=0.01)
        clk.reset()
        t0 = time.perf_counter()
        for i in range(100):
            clk.pace(i * 0.01)
        self.assertLess(time.perf_counter() - t0, 0.05, "rtf=inf must not sleep")

    def test_set_rtf_reanchors(self):
        clk = RenderClock(render_fps=60, real_time_factor=float("inf"), timestep=0.01)
        clk.reset()
        for i in range(50):
            clk.pace(i * 0.01)  # free-run, anchor unset under inf
        clk.set_rtf(1.0)
        self.assertEqual(clk.rtf, 1.0)
        # after switching to rtf=1 there must be no catch-up sleep for the elapsed sim time
        t0 = time.perf_counter()
        clk.pace(0.5)  # first call re-anchors, should not block on the 0.5s of prior sim time
        self.assertLess(time.perf_counter() - t0, 0.05, "set_rtf must re-anchor (no catch-up spike)")

    def test_display_cap_wall_clock(self):
        # display_due gates at most render_fps times per wall-second
        clk = RenderClock(render_fps=1000, real_time_factor=float("inf"), timestep=0.01)
        clk.reset()
        # first call always due
        self.assertTrue(clk.display_due())
        # immediate second call not due (1ms period not elapsed)
        self.assertFalse(clk.display_due())

