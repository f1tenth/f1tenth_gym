import pathlib
import unittest

import numpy as np

from f110_gym.envs.track import Track, find_track_dir, Raceline


class TestTrack(unittest.TestCase):
    def test_loading_default_tracks(self):
        track_names = ["Berlin", "Example", "Levine", "Skirk", "StataBasement", "Vegas"]
        for track_name in track_names:
            track = Track.from_track_name(track_name)
            self.assertEqual(track.spec.name, track_name)

    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        self.assertRaises(FileNotFoundError, Track.from_track_name, wrong_track_name)

    def test_raceline(self):
        track_name = "Example"  # Example is the only track with a raceline for now
        track = Track.from_track_name(track_name)

        # check raceline is not None
        self.assertNotEqual(track.raceline, None)

        # check loaded raceline match the one in the csv file
        track_dir = find_track_dir(track_name)
        assert track_dir is not None and track_dir.exists(), "track_dir does not exist"

        raceline = np.loadtxt(track_dir / f"{track_name}_raceline.csv", delimiter=";")
        s_idx, x_idx, y_idx, psi_idx, kappa_idx, vx_idx, ax_idx = range(7)

        self.assertTrue(np.isclose(track.raceline.ss, raceline[:, s_idx]).all())
        self.assertTrue(np.isclose(track.raceline.xs, raceline[:, x_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ys, raceline[:, y_idx]).all())
        self.assertTrue(np.isclose(track.raceline.yaws, raceline[:, psi_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ks, raceline[:, kappa_idx]).all())
        self.assertTrue(np.isclose(track.raceline.vxs, raceline[:, vx_idx]).all())
        self.assertTrue(np.isclose(track.raceline.axs, raceline[:, ax_idx]).all())

    def test_missing_raceline(self):
        track = Track.from_track_name("Vegas")
        self.assertEqual(track.raceline, None)
        self.assertEqual(track.centerline, None)

    def test_map_dir_structure(self):
        """
        Check that the map dir structure is correct:
        - maps/
            - Trackname/
                - Trackname_map.*               # map image
                - Trackname_map.yaml            # map specification
                - [Trackname_raceline.csv]      # raceline (optional)
                - [Trackname_centerline.csv]    # centerline (optional)
        """
        mapdir = pathlib.Path(__file__).parent.parent / "maps"
        for trackdir in mapdir.iterdir():
            if trackdir.is_file():
                continue

            # check subdir is capitalized (at least first letter is capitalized)
            trackdirname = trackdir.stem
            self.assertTrue(
                trackdirname[0].isupper(), f"trackdir {trackdirname} is not capitalized"
            )

            # check map spec file exists
            file_spec = trackdir / f"{trackdirname}_map.yaml"
            self.assertTrue(
                file_spec.exists(),
                f"map spec file {file_spec} does not exist in {trackdir}",
            )

            # read map image file from spec
            map_spec = Track.load_spec(track=str(trackdir), filespec=str(file_spec))
            file_image = trackdir / map_spec.image

            # check map image file exists
            self.assertTrue(
                file_image.exists(),
                f"map image file {file_image} does not exist in {trackdir}",
            )

            # check raceline and centerline files
            file_raceline = trackdir / f"{trackdir.stem}_raceline.csv"
            file_centerline = trackdir / f"{trackdir.stem}_centerline.csv"

            if file_raceline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                raceline = Raceline.from_raceline_file(file_raceline)

            if file_centerline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                centerline = Raceline.from_raceline_file(file_centerline)
