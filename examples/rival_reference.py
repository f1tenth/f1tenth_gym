"""Bound-constrained periodic geometric references (no controller integration).

Only nodal legality is asserted. Periodic linear interpolation preserves the
linearly interpolated bands, but is not a continuous-curvature or footprint
safety certificate. No temporal filtering is performed.
"""

import numpy as np
from scipy.optimize import minimize


def smoothed_reference(nodes, lower, upper, period, smoothing_length=3.):
    """Return a deterministic reference and finite-difference diagnostics.

    Uniform nodes must start at zero and exclude the repeated endpoint. Minimize
    ds/2 * sum(d**2 + length**2 * D1(d)**2 + length**4 * D2(d)**2),
    with periodic forward D1 and centered D2, subject to lower <= d <= upper.
    The offset term makes the quadratic strictly convex, including at length=0.
    Derivatives in diagnostics include the closing neighbor and have units
    offset/metre and offset/metre**2; they are not vehicle path curvature.
    Invalid inputs, numerical overflow, or unsuccessful optimization raise
    ValueError rather than silently relaxing any legal band.
    """
    nodes, lower, upper = (np.asarray(v, dtype=float) for v in (nodes, lower, upper))
    if (nodes.ndim != 1 or len(nodes) < 3 or lower.shape != nodes.shape
            or upper.shape != nodes.shape):
        raise ValueError('nodes, lower, upper must be matching 1D arrays with at least three nodes')
    if not all(np.all(np.isfinite(v)) for v in (nodes, lower, upper)):
        raise ValueError('nodes and legal bounds must be finite')
    period, smoothing_length = float(period), float(smoothing_length)
    if not np.isfinite(period) or period <= 0:
        raise ValueError('period must be finite and positive')
    if not np.isfinite(smoothing_length) or smoothing_length < 0:
        raise ValueError('smoothing_length must be finite and nonnegative')
    ds = period / len(nodes)
    if (nodes[0] != 0 or nodes[-1] >= period or np.any(np.diff(nodes) <= 0)
            or not np.allclose(np.diff(np.r_[nodes, period]), ds, rtol=1e-9, atol=ds * 1e-12)):
        raise ValueError('nodes must be uniform, ascending from 0, including the closing interval')
    invalid = np.flatnonzero(lower > upper)
    if len(invalid):
        i = int(invalid[0])
        raise ValueError(f'Empty legal band at node {i}, s={nodes[i]:.6g}: lower > upper')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a = np.float64(smoothing_length / ds) ** 2
        b = a * a
        scale = 1. + 4. * a + 16. * b
    if not np.isfinite(scale) or ds <= 0:
        raise ValueError('smoothing_length / node spacing is too large for finite coefficients')

    def quadratic(x):
        forward = np.roll(x, -1) - x
        second = np.roll(x, -1) - 2 * x + np.roll(x, 1)
        value = .5 * (np.dot(x, x) + a * np.dot(forward, forward)
                      + b * np.dot(second, second))
        gradient = (x - a * second
                    + b * (np.roll(second, -1) - 2 * second + np.roll(second, 1)))
        # Scalar preconditioning leaves the minimizer unchanged.
        return value / scale, gradient / scale

    initial = np.clip(np.zeros_like(nodes), lower, upper)
    with np.errstate(over='ignore', invalid='ignore'):
        result = minimize(quadratic, initial, jac=True, method='L-BFGS-B',
                          bounds=list(zip(lower, upper)),
                          options={'maxiter': 20000, 'maxls': 50, 'maxcor': 20,
                                   'ftol': 1e-15, 'gtol': 1e-11})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f'Periodic reference optimization failed: {result.message}; '
                         'check band magnitudes, spacing, and smoothing_length')
    reference = np.clip(result.x, lower, upper)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        slope = (np.roll(reference, -1) - reference) / ds
        second = (np.roll(reference, -1) - 2 * reference + np.roll(reference, 1)) / ds ** 2
        objective = quadratic(reference)[0] * scale * ds
    if not (np.isfinite(objective) and np.all(np.isfinite(slope))
            and np.all(np.isfinite(second))):
        raise ValueError('Reference objective or derivatives overflowed; rescale input geometry')
    return reference, {'solver_success': True, 'max_slope': float(np.max(np.abs(slope))),
                       'max_second_derivative': float(np.max(np.abs(second))),
                       'objective': float(objective), 'iterations': int(getattr(result, 'nit', 0)),
                       'solver_message': str(result.message)}


def _reference_poses(line, nodes, reference):
    """Nodal poses using periodic centered spatial slopes, including closure."""
    ds = line.length / len(nodes)
    slope = (np.roll(reference, -1) - np.roll(reference, 1)) / (2 * ds)
    curvature = np.array([line.curvature(s) for s in nodes])
    yaw = np.array([line.heading(s) for s in nodes]) + np.arctan2(
        slope, np.maximum(.1, 1 - curvature * reference))
    xy = np.array([line.sample(s, d) for s, d in zip(nodes, reference)])
    return np.column_stack((xy, yaw))


def clearance_reference(line, track, nodes, lower, upper, length, width, margin,
                        smoothing_length=3., step=.02, iterations=2):
    """Smooth within connected, full-body-clear subsets of ORIGINAL legal bands.

    Each refinement tests a grid of spacing <= step with margin + step/2,
    covering normal interpolation displacement at the current nodal yaw. The
    nearest connected valid run is selected (widest breaks distance ties).
    Yaw changes on re-smoothing, so final poses are independently certified at
    the requested margin; failure raises rather than returning an unsafe path.
    This certifies nodes only, not between-node swept motion or tracking error.
    """
    for name, value in [('length', length), ('width', width), ('margin', margin),
                        ('step', step)]:
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if step == 0 or step > .02:
        raise ValueError('step must be positive and <= .02 metres')
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    reference, diagnostics = smoothed_reference(nodes, lower, upper, line.length, smoothing_length)
    nodes = np.asarray(nodes, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def safe(poses, required):
        # Bound temporary rectangle-grid memory on long/wide tracks.
        return np.concatenate([track.footprints_safe(poses[i:i + 1024], length, width, required)
                               for i in range(0, len(poses), 1024)])

    for refinement in range(iterations):
        poses = _reference_poses(line, nodes, reference)
        grids = [np.linspace(lo, hi, max(1, int(np.ceil((hi - lo) / step))) + 1)
                 if hi > lo else np.array([lo]) for lo, hi in zip(lower, upper)]
        counts = np.array([len(grid) for grid in grids])
        offsets = np.concatenate(grids)
        indices = np.repeat(np.arange(len(nodes)), counts)
        centres = np.array([line.sample(s) for s in nodes])
        normals = np.array([line.sample(s, 1.) for s in nodes]) - centres
        candidates = np.column_stack((centres[indices] + offsets[:, None] * normals[indices],
                                      poses[indices, 2]))
        valid = safe(candidates, margin + step / 2)
        refined_lo, refined_hi = np.empty_like(lower), np.empty_like(upper)
        start = 0
        for i, grid in enumerate(grids):
            mask = valid[start:start + len(grid)]
            start += len(grid)
            edges = np.diff(np.r_[False, mask, False].astype(int))
            runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
            if not runs:
                raise ValueError(f'Empty body-clear legal band at node {i}, s={nodes[i]:.6g}, '
                                 f'refinement {refinement + 1}, margin={margin}; '
                                 'original legal bounds were not relaxed')
            def rank(run):
                lo, hi = grid[run[0]], grid[run[1]]
                return (max(lo - reference[i], reference[i] - hi, 0.), -(hi - lo), lo)
            first, last = min(runs, key=rank)
            refined_lo[i], refined_hi[i] = grid[first], grid[last]
        reference, diagnostics = smoothed_reference(
            nodes, refined_lo, refined_hi, line.length, smoothing_length)
    final_valid = safe(_reference_poses(line, nodes, reference), margin)
    if not np.all(final_valid):
        i = int(np.flatnonzero(~final_valid)[0])
        raise ValueError(f'Final reference footprint fails margin={margin} at node {i}, '
                         f's={nodes[i]:.6g} after {iterations} refinements; '
                         'increase iterations or inspect body-clear corridor geometry')
    diagnostics.update(min_bandwidth=float(np.min(refined_hi - refined_lo)),
                       refinements=int(iterations), footprint_success=True,
                       empty_bands=0, clearance_margin=float(margin), grid_cover=float(step / 2))
    return reference, diagnostics
