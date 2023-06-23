import unittest
from f110_gym.envs.track import Track


class TestTrack(unittest.TestCase):
    def test_loading_default_tracks(self):
        track_names = ["Berlin", "Example", "Levine", "Skirk", "StataBasement", "Vegas"]
        for track_name in track_names:
            track = Track.from_track_name(track_name)
            self.assertEqual(track.spec.name, track_name)

    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        self.assertRaises(FileNotFoundError, Track.from_track_name, wrong_track_name)