"""Exact LiDAR scanning against oriented wall segments."""

import numpy as np

from ..track.ray_tiles import DEFAULT_TILE_SIZE, build_for_track


class SegmentScanSimulator2D:
    """Analytic ray-segment scanner over a per-tile candidate list."""

    def __init__(self, num_beams, fov, angle_min=None, angle_max=None, std_dev=0.01,
                 min_range=0.0, max_range=30.0, tile_size=DEFAULT_TILE_SIZE,
                 device="cpu"):
        """
        Args:
            num_beams: Beams per sweep.
            fov: Total field of view in radians, used when the angles are not given.
            angle_min: First beam angle relative to heading, or None to derive it.
            angle_max: Last beam angle, or None to derive it.
            std_dev: Observation noise standard deviation, applied by ``scan``.
            min_range: Threshold below which the environment reports zero.
            max_range: Longest reported range, and the radius the tile table covers.
            tile_size: Tile side in metres for the candidate table.
            device: ``"cpu"`` or ``"gpu"`` for the jitted kernel.
        """
        self.num_beams = int(num_beams)
        self.fov = float(fov)
        self.angle_min = float(angle_min) if angle_min is not None else -self.fov / 2.0
        self.angle_max = float(angle_max) if angle_max is not None else self.fov / 2.0
        self.std_dev = float(std_dev)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.tile_size = float(tile_size)
        self.device = device

        self.angle_increment = (self.angle_max - self.angle_min) / max(self.num_beams - 1, 1)
        self._angles = (self.angle_min
                        + np.arange(self.num_beams, dtype=np.float64) * self.angle_increment
                        ).astype(np.float32)
        self.track = None
        self.map_scale = 1.0
        self.index = None
        self._scan = None

    def get_increment(self) -> float:
        """Angle between adjacent beams, in radians."""
        return self.angle_increment

    def set_map(self, track, map_scale: float = 1.0) -> None:
        """Extract walls for ``track`` and compile the kernel against them.

        Args:
            track: A ``Track``.
            map_scale: Accepted for interface parity; the segments already carry it.
        """
        import jax
        import jax.numpy as jnp
        from jax.sharding import SingleDeviceSharding

        from ..contact.adapter import resolve_device
        from f1tenth_gym.envs.lidar.kernels import ray_segment_range

        self.track = track
        self.map_scale = float(map_scale)
        walls, index = build_for_track(track, self.max_range, self.tile_size)
        self.index = index
        if index.is_empty:
            self._scan = None
            return

        dev = resolve_device(self.device)
        # One degenerate segment past the end, which the table pads with: it gives a
        # zero denominator, so padded slots need no mask.
        seg_a = jax.device_put(np.vstack([walls.a, walls.a[:1]]).astype(np.float32), dev)
        seg_b = jax.device_put(np.vstack([walls.b, walls.a[:1]]).astype(np.float32), dev)
        table = jax.device_put(index.table, dev)
        origin = jax.device_put(np.asarray(index.origin, np.float32), dev)
        angles = jax.device_put(self._angles, dev)
        tile = float(index.tile_size)
        rows, cols = int(table.shape[0]), int(table.shape[1])
        reach = float(self.max_range)

        def run(pose):
            col = jnp.clip(((pose[0] - origin[0]) / tile).astype(jnp.int32), 0, cols - 1)
            row = jnp.clip(((pose[1] - origin[1]) / tile).astype(jnp.int32), 0, rows - 1)
            cand = table[row, col]
            bearing = pose[2] + angles
            direction = jnp.stack([jnp.cos(bearing), jnp.sin(bearing)], axis=1)
            return ray_segment_range(pose[:2], direction, seg_a[cand], seg_b[cand], reach)

        self._scan = jax.jit(run, out_shardings=SingleDeviceSharding(dev))

    def scan(self, pose, rng=None):
        """Ranges for one sweep.

        Args:
            pose: (3,) sensor ``(x, y, yaw)``.
            rng: Generator for observation noise, or None for a clean scan.

        Returns:
            (num_beams,) float32 ranges in metres.

        Raises:
            ValueError: If called before ``set_map``.
        """
        if self.track is None:
            raise ValueError("Map is not set for scan simulator.")
        if self._scan is None:
            out = np.full(self.num_beams, self.max_range, dtype=np.float64)
        else:
            out = np.array(self._scan(np.asarray(pose, np.float32)), dtype=np.float64)
        if rng is not None and self.std_dev > 0.0:
            out = out + rng.normal(0.0, self.std_dev, self.num_beams).astype(np.float32)
        return out
