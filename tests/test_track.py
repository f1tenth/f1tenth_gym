import pathlib
import time
import unittest

import numpy as np
from f1tenth_gym.envs.track import Raceline, Track, find_track_dir


class TestTrack(unittest.TestCase):
    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        self.assertRaises(FileNotFoundError, Track.from_track_name, wrong_track_name)

    def test_raceline(self):
        track_name = "Spielberg"
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
                Raceline.from_raceline_file(file_raceline)

            if file_centerline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                Raceline.from_centerline_file(file_centerline)

    def test_download_racetrack(self):
        import shutil

        track_name = "Spielberg"
        track_backup = Track.from_track_name(track_name)

        # rename the track dir
        track_dir = find_track_dir(track_name)
        tmp_dir = track_dir.parent / f"{track_name}_tmp{int(time.time())}"
        track_dir.rename(tmp_dir)

        # download the track
        track = Track.from_track_name(track_name)

        # check the two tracks' specs are the same
        for spec_attr in [
            "name",
            "image",
            "resolution",
            "origin",
            "negate",
            "occupied_thresh",
            "free_thresh",
        ]:
            self.assertEqual(
                getattr(track.spec, spec_attr), getattr(track_backup.spec, spec_attr)
            )

        # check the two tracks' racelines are the same
        for raceline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.raceline, raceline_attr),
                    getattr(track_backup.raceline, raceline_attr),
                ).all()
            )

        # check the two tracks' centerlines are the same
        for centerline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.centerline, centerline_attr),
                    getattr(track_backup.centerline, centerline_attr),
                ).all()
            )

        # remove the newly created track dir
        track_dir = find_track_dir(track_name)
        shutil.rmtree(track_dir, ignore_errors=True)

        # rename the backup track dir to its original name
        track_backup_dir = find_track_dir(tmp_dir.stem)
        track_backup_dir.rename(track_dir)

    def test_frenet_to_cartesian(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # Check frenet to cartesian conversion
        # using the track's xs, ys
        for s, x, y in zip(
            track.centerline.ss, track.centerline.xs, track.centerline.ys
        ):
            x_, y_, _ = track.frenet_to_cartesian(s, 0, 0)
            self.assertAlmostEqual(x, x_, places=2)
            self.assertAlmostEqual(y, y_, places=2)

    def test_frenet_to_cartesian_to_frenet(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check frenet to cartesian conversion
        s_ = 0
        for s in np.linspace(0, 1, 10):
            x, y, psi = track.frenet_to_cartesian(s, 0, 0)
            s_, d, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=2)
            self.assertAlmostEqual(d, 0, places=2)

        # check frenet to cartesian conversion
        # with non-zero lateral offset
        s_ = 0
        for s in np.linspace(0, 1, 10):
            d = np.random.uniform(-1.0, 1.0)
            x, y, psi = track.frenet_to_cartesian(s, d, 0)
            s_, d_, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=2)
            self.assertAlmostEqual(d, d_, places=2)
