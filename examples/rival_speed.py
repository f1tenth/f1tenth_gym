"""Shared, precomputed periodic reference-speed planning; no controller changes.

Integration (opt in at the caller)::

    profile = SpeedProfile(line, envelope, vehicle_settings, robust_config, v_max)
    requested_speed = min(requested_speed, profile.limit(state.s, state.speed))

Construct once per reference/speed limit and share between cars. ``envelope`` is
TrackEnvelope-compatible (s, reference_offsets, reference_slope(s)); None plans
for the unshifted line. No map queries, physics, tactical offsets, or temporal
command filtering are involved. Rebuild if the reference or settings change.

Optional config attributes, with suggested initial defaults:
  speed_profile_spacing = .25       maximum additional sample spacing, metres
  speed_lateral_reserve = .9        fraction of max_lateral_accel available
  speed_slope_reserve = .8          fraction of offset_rate available
  speed_reaction_time = control_period (or vehicle.timestep)
  speed_jerk_reserve = 1.            multiplier of (max_acceleration+max_braking)/max_jerk

These attributes need not be added to RobustConfig to use its existing defaults.
They are deliberately NOT added to its strict YAML parser by this module.
Lateral reserve allows for tracking error; slope reserve leaves target-filter
headroom. The response preview assumes no useful braking during reaction and a
full acceleration-to-braking jerk ramp. This is a pragmatic reservation, NOT a
jerk-feasibility proof or a remedy for an already overspeed vehicle. Keep the
existing acceleration/jerk, footprint, opponent, and tactical safety checks.

ClosedRaceline and TrackEnvelope are polygonal/piecewise-linear approximations.
Offset curvature below is a spatial finite-difference estimate of their smooth
reference interpretation, not a continuous swept-path certificate. In particular,
linear offset interpolation has derivative jumps at nodes. Positive d and k mean
left of travel; derivatives are with respect to raceline metres, not node index.
"""

import numpy as np


def _number(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {qualifier}')
    return value


def _derivatives(nodes, values, period):
    """Periodic quadratic finite differences, including nonuniform closure."""
    forward_ds = np.diff(np.r_[nodes, period])
    backward_ds = np.roll(forward_ds, 1)
    forward = (np.roll(values, -1) - values) / forward_ds
    backward = np.roll(forward, 1)
    first = (forward_ds * backward + backward_ds * forward) / (forward_ds + backward_ds)
    second = 2 * (forward - backward) / (forward_ds + backward_ds)
    return first, second, np.maximum(np.abs(forward), np.abs(backward))


class SpeedProfile:
    """Immutable sampled plan with stateless O(log N) wrapped ``limit`` queries.

    ``config`` supplies max_lateral_accel, max_braking, max_acceleration (or
    max_accel), max_jerk, and optionally control_period/max_lateral_rate plus the
    speed_* attributes above. ``vehicle`` supplies lookahead and offset_rate.
    speed_limit is finite, nonnegative, and in m/s. Bad geometry/settings raise
    ValueError rather than silently creating an unsafe, NaN, or fallback plan.

    Public read-only arrays use ``s`` (raceline metres, no repeated endpoint).
    ``reference_curvature`` is signed offset-path curvature; ``curvature`` is the
    conservative absolute maximum of it, line curvature, and offset-reference
    three-point geometric curvature. ``segment_distances`` bounds planning
    distance conservatively by raceline ds, reference metric, and XY chord.
    ``raw_limits`` combines lateral/slope/absolute caps; ``speeds`` additionally
    satisfies the cyclic backward braking inequality. Squared speed is linearly
    interpolated. Acceleration on corner exit remains the caller's job.
    ``diagnostics`` contains construction summaries, never per-car query state.
    """

    def __init__(self, line, envelope, vehicle, config, speed_limit):
        self.length = _number(line.length, 'line.length', positive=True)
        self.speed_limit = _number(speed_limit, 'speed_limit')
        braking = _number(config.max_braking, 'max_braking', positive=True)
        acceleration = getattr(config, 'max_acceleration', None)
        if acceleration is None:
            acceleration = config.max_accel
        self._acceleration = _number(acceleration, 'max_acceleration', positive=True)
        jerk = _number(config.max_jerk, 'max_jerk', positive=True)
        lateral = _number(config.max_lateral_accel, 'max_lateral_accel', positive=True)
        self._lookahead = _number(vehicle.lookahead, 'lookahead')
        rate = _number(vehicle.offset_rate, 'offset_rate')
        if hasattr(config, 'max_lateral_rate'):
            rate = min(rate, _number(config.max_lateral_rate, 'max_lateral_rate'))
        spacing = _number(getattr(config, 'speed_profile_spacing', .25),
                          'speed_profile_spacing', positive=True)
        reserves = []
        for name, default in [('speed_lateral_reserve', .9), ('speed_slope_reserve', .8)]:
            value = _number(getattr(config, name, default), name, positive=True)
            if value > 1:
                raise ValueError(f'{name} must be <= 1')
            reserves.append(value)
        lateral_reserve, slope_reserve = reserves
        reaction = _number(getattr(config, 'speed_reaction_time',
                           getattr(config, 'control_period', getattr(vehicle, 'timestep', .02))),
                           'speed_reaction_time')
        jerk_reserve = _number(getattr(config, 'speed_jerk_reserve', 1.), 'speed_jerk_reserve')
        self._response_time = reaction + jerk_reserve * (self._acceleration + braking) / jerk
        with np.errstate(over="ignore", invalid="ignore"):
            self._response_acceleration_distance = (
                .5 * np.float64(self._acceleration) * np.float64(self._response_time) ** 2)
        if not np.isfinite(self._response_acceleration_distance):
            raise ValueError("Nonfinite reaction/jerk response distance; rescale dynamics settings")

        if envelope is None:
            nodes = np.linspace(0., self.length, max(16, int(np.ceil(self.length / spacing))),
                                endpoint=False)
            offsets = np.zeros_like(nodes)
        else:
            nodes = np.array(envelope.s, dtype=float, copy=True)
            offsets = np.array(envelope.reference_offsets, dtype=float, copy=True)
        if (nodes.ndim != 1 or len(nodes) < 3 or offsets.shape != nodes.shape
                or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(offsets))
                or nodes[0] != 0 or nodes[-1] >= self.length or np.any(np.diff(nodes) <= 0)):
            raise ValueError('envelope s/offsets must be matching finite 1D arrays, '
                             'with >=3 nodes increasing from 0 in [0,line.length)')

        # Retain all reference nodes AND raceline vertices: uniform resampling
        # alone can miss a short isolated bend on an arbitrary ClosedRaceline.
        uniform = np.linspace(0., self.length, max(16, int(np.ceil(self.length / spacing))),
                              endpoint=False)
        grids = [nodes, uniform]
        points = getattr(line, 'points', None)
        if points is not None:
            edges = np.roll(points, -1, axis=0) - points
            grids.append(np.r_[0., np.cumsum(np.linalg.norm(edges, axis=1))][:-1])
        self.s = np.unique(np.concatenate(grids))
        # Floating-point coincident vertices/nodes must not create tiny cells.
        tolerance = 32 * np.finfo(float).eps * self.length
        self.s = self.s[np.r_[True, np.diff(self.s) > tolerance]]
        self.s = self.s[self.s < self.length - tolerance]
        ds = np.diff(np.r_[self.s, self.length])

        def interpolate(values):
            return np.interp(self.s, nodes, values, period=self.length)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            first, second, slope_bound = _derivatives(nodes, offsets, self.length)
            d, d1, d2 = interpolate(offsets), interpolate(first), interpolate(second)
            self.reference_slope = interpolate(slope_bound)
            if envelope is not None:
                supplied_slope = np.array([envelope.reference_slope(s) for s in self.s], dtype=float)
                if supplied_slope.shape != self.s.shape or not np.all(np.isfinite(supplied_slope)):
                    raise ValueError('envelope.reference_slope must return finite scalars')
                self.reference_slope = np.maximum(self.reference_slope, np.abs(supplied_slope))
            k = np.array([line.curvature(s) for s in self.s], dtype=float)
            if k.shape != self.s.shape or not np.all(np.isfinite(k)):
                raise ValueError('line.curvature must return finite scalars')
            k1, _, _ = _derivatives(self.s, k, self.length)
            along = 1 - k * d
            if np.any(along <= 1e-6):
                raise ValueError('Offset reference folds or becomes singular: 1 - curvature * offset <= 0')
            metric = np.hypot(along, d1)
            # r'=(1-k*d)T+d'N; r''=(-k'*d-2*k*d')T+(k*(1-k*d)+d'')N.
            self.reference_curvature = (k * along ** 2 + along * d2
                                       + k1 * d * d1 + 2 * k * d1 ** 2) / metric ** 3
            xy = np.array([line.sample(s, offset) for s, offset in zip(self.s, d)])
            if xy.shape != (len(self.s), 2) or not np.all(np.isfinite(xy)):
                raise ValueError('line.sample must return finite XY positions')
            chords = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
            # Estimate geometric curvature on the ORIGINAL reference grid, not
            # tiny resampled polygon cells with discontinuous segment normals.
            ref_xy = np.array([line.sample(s, offset) for s, offset in zip(nodes, offsets)])
            if ref_xy.shape != (len(nodes), 2) or not np.all(np.isfinite(ref_xy)):
                raise ValueError('line.sample must return finite XY positions')
            edges = np.roll(ref_xy, -1, axis=0) - ref_xy
            previous = np.roll(edges, 1, axis=0)
            denominator = (np.linalg.norm(previous, axis=1) * np.linalg.norm(edges, axis=1)
                           * np.linalg.norm(previous + edges, axis=1))
            cross = previous[:, 0] * edges[:, 1] - previous[:, 1] * edges[:, 0]
            geometric = np.divide(2 * cross, denominator, out=np.zeros_like(cross),
                                  where=denominator > 0)
            self.curvature = np.maximum.reduce([np.abs(k), np.abs(self.reference_curvature),
                                                interpolate(np.abs(geometric))])
            self.segment_distances = np.minimum.reduce([
                ds, ds * np.minimum(metric, np.roll(metric, -1)), chords])
            self.reference_second_derivative = d2
            self.reference_metric = metric
            self.line_curvature = k
            self.lateral_limits = np.full_like(self.s, self.speed_limit)
            turning = self.curvature > 0
            self.lateral_limits[turning] = np.minimum(self.speed_limit, np.sqrt(
                lateral * lateral_reserve / self.curvature[turning]))
            self.slope_limits = np.full_like(self.s, self.speed_limit)
            changing = self.reference_slope > 0
            self.slope_limits[changing] = np.minimum(self.speed_limit,
                slope_reserve * rate * np.minimum(1., metric[changing])
                / self.reference_slope[changing])
            self.raw_limits = np.minimum(self.lateral_limits, self.slope_limits)
            squared = self.raw_limits ** 2
            braking_budget = 2 * braking * self.segment_distances

        arrays = (self.reference_curvature, self.curvature, self.reference_slope,
                  self.reference_second_derivative, self.reference_metric,
                  self.segment_distances, squared, braking_budget)
        if (not all(np.all(np.isfinite(a)) for a in arrays)
                or not np.isfinite(self._response_time)
                or np.any(self.segment_distances <= 0)):
            raise ValueError('Nonfinite or degenerate reference geometry/speed coefficients')
        # Two reverse sweeps propagate every possible limiting node across the
        # seam, independent of where the lap starts. All edge costs are positive.
        count = len(self.s)
        for _ in range(2):
            for i in range(count - 1, -1, -1):
                squared[i] = min(squared[i], squared[(i + 1) % count] + braking_budget[i])
        self.speeds = np.sqrt(squared)
        self._squared = np.r_[squared, squared[0]]
        self._wrapped_s = np.r_[self.s, self.length]
        self._doubled_s = np.r_[self.s, self.s + self.length]
        # Sparse range-minimum table: O(N log N) once, O(1) per index range.
        self._minima = [np.tile(squared, 2)]
        width = 2
        while width <= 2 * count:
            previous = self._minima[-1]
            half = width // 2
            self._minima.append(np.minimum(previous[:-half], previous[half:]))
            width *= 2
        self._global_min = float(np.min(squared))
        self.diagnostics = {
            'nodes': count, 'reference_nodes': len(nodes),
            'reference_source': 'line' if envelope is None else 'envelope',
            'min_speed': float(np.min(self.speeds)), 'max_speed': float(np.max(self.speeds)),
            'max_curvature': float(np.max(self.curvature)),
            'max_reference_slope': float(np.max(self.reference_slope)),
            'max_reference_second_derivative': float(np.max(np.abs(d2))),
            'lateral_reserve': lateral_reserve, 'slope_reserve': slope_reserve,
            'response_time': float(self._response_time), 'lookahead': self._lookahead,
            'max_braking': braking,
            'braking_limited_nodes': int(np.count_nonzero(self.speeds < self.raw_limits)),
            'max_braking_residual': float(np.max(squared - np.roll(squared, -1) - braking_budget)),
        }
        for array in (self.s, self.speeds, self.curvature, self.line_curvature,
                      self.reference_curvature, self.reference_slope, self.reference_second_derivative,
                      self.reference_metric, self.segment_distances, self.lateral_limits,
                      self.slope_limits, self.raw_limits, self._squared, self._wrapped_s,
                      self._doubled_s, *self._minima):
            array.flags.writeable = False

    def _at(self, s):
        return float(np.interp(s % self.length, self._wrapped_s, self._squared))

    def _front_minimum(self, s, distance):
        if distance >= self.length:
            return self._global_min
        end = s + distance
        result = min(self._at(s), self._at(end))
        lo = int(np.searchsorted(self._doubled_s, s, side='right'))
        hi = int(np.searchsorted(self._doubled_s, end, side='left'))
        if hi > lo:
            level = (hi - lo).bit_length() - 1
            width = 1 << level
            result = min(result, self._minima[level][lo], self._minima[level][hi - width])
        return result

    def limit(self, s, measured_speed=0.):
        """Return a front-preview cap in m/s, including the CURRENT position.

        s wraps in either direction; nonfinite inputs raise ValueError. Negative
        measured speed is treated as zero forward speed. Preview uses the larger
        of measured speed and the local planned cap, so stopped cars still see
        upcoming braking and overspeed cars reserve extra response distance.
        Minimum over the ENTIRE preview prevents jumping past an isolated corner.
        A constant flat-straight cap is unchanged, regardless of preview length.
        """
        s, measured_speed = float(s), float(measured_speed)
        if not np.isfinite(s) or not np.isfinite(measured_speed):
            raise ValueError('s and measured_speed must be finite')
        s %= self.length
        speed = max(0., measured_speed, np.sqrt(self._at(s)))
        # No useful deceleration is credited during reaction/ramp; extra travel
        # under bounded positive acceleration gets a distance reservation too.
        with np.errstate(over='ignore', invalid='ignore'):
            distance = (self._lookahead + speed * self._response_time
                        + self._response_acceleration_distance)
        return float(np.sqrt(self._front_minimum(s, distance)))
