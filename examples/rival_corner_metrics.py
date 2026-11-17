"""Headless, reference-relative diagnostics for completed rival action CSV rows.

Truth columns are evaluator-only: def uses ego_truth_eval_{s,d,yaw,speed} and
steering; att uses opponent_truth_eval_{s,d,yaw,speed} and attacker_steering.
Commands/targets are never substituted for measured speed or reference offset.
Distances are metres, angles radians and durations seconds.

Each pre-action observation is held over its actual action_elapsed (preferred),
otherwise the next row's time gap. No nominal timestep or extrapolated final
interval is assumed. A final row without action_elapsed is rejected explicitly.
Rows with time < start_time or time >= a car's end_time are excluded, and the
last included interval is clipped at that end_time. We do not interpolate truth
within an interval. All input numeric fields are validated even outside windows.

RMS and absolute p95 are duration-weighted; p95 is the inverse empirical CDF
(no interpolation). Max is absolute. Empty subsets have null statistics, not
perfect zero errors. Steering variation uses adjacent, temporally contiguous
rows only; corner variation additionally requires both rows to be corners.
Sign reversals ignore commands within a 0.01-rad deadband (about 0.57 degrees),
retaining the last significant sign within each contiguous segment. These are
descriptive counts, not a claim that normal cornering reversals are bad chatter.
"""

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CURVATURE_THRESHOLD = .05
REFERENCE_SLOPE_THRESHOLD = .05
SLOW_SPEED_THRESHOLD = .3
STEERING_DEADBAND = .01
_CARS = {'def': ('ego_truth_eval', 'steering'),
         'att': ('opponent_truth_eval', 'attacker_steering')}


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name}: expected a finite number, got {value!r}') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name}: expected a finite number, got {value!r}')
    return number


def _field(row, name, index):
    if name not in row:
        raise ValueError(f'row {index}: missing {name}')
    return _finite(row[name], f'row {index} {name}')


def _statistics(values, weights):
    if not values:
        return dict(rms=None, p95=None, max=None)
    absolute = [abs(value) for value in values]
    maximum = max(absolute)
    total = math.fsum(weights)
    # Scale before squaring so large, finite errors need not overflow.
    rms = (maximum * math.sqrt(math.fsum((value / maximum) ** 2 * (weight / total)
            for value, weight in zip(absolute, weights))) if maximum else 0.)
    cumulative = 0.
    percentile = maximum
    for value, weight in sorted(zip(absolute, weights)):
        cumulative += weight
        if cumulative >= .95 * total:
            percentile = value
            break
    return dict(rms=rms, p95=percentile, max=maximum)


def _contiguous(previous, current):
    return (current['index'] == previous['index'] + 1
            and math.isclose(previous['end'], current['time'], rel_tol=0., abs_tol=1e-9))


def _steering(samples):
    variation = 0.
    reversals = pairs = 0
    previous = None
    last_sign = 0
    for sample in samples:
        if previous is None or not _contiguous(previous, sample):
            last_sign = 0
        else:
            variation += abs(sample['steering'] - previous['steering'])
            pairs += 1
        command = sample['steering']
        sign = (1 if command > STEERING_DEADBAND else
                -1 if command < -STEERING_DEADBAND else 0)
        if sign:
            reversals += int(last_sign != 0 and sign != last_sign)
            last_sign = sign
        previous = sample
    return dict(total_variation=variation if samples else None,
                significant_sign_reversals=reversals, adjacent_pairs=pairs)


def _summarize(samples) -> dict[str, Any]:
    weights = [sample['duration'] for sample in samples]
    return dict(samples=len(samples), duration=math.fsum(weights),
        derror=_statistics([sample['derror'] for sample in samples], weights),
        heading_error=_statistics([sample['heading_error'] for sample in samples], weights),
        slow_stopped_duration=math.fsum(sample['duration'] for sample in samples
                                      if abs(sample['speed']) < SLOW_SPEED_THRESHOLD),
        steering=_steering(samples))


def corner_metrics(rows, line, envelope, *, start_time=2., end_times=None) -> dict[str, Any]:
    """Return a strict-JSON-compatible dict with parameters, samples, def and att.

    Each car has all-window metrics plus a nested ``corner`` subset, where
    abs(line.curvature(s)) >= .05 OR abs(envelope.reference_slope(s)) >= .05.
    derror = d - envelope.reference(s). Heading error wraps yaw minus
    [line.heading(s) + atan2(reference_slope(s), 1 - curvature(s)*reference(s))]
    into [-pi, pi). In particular, the tangent metric uses reference d, not the
    measured d or a tactical target.

    end_times is an optional mapping with def/att keys (omitted/None means no
    finish cutoff), or a two-element sequence in def, att order. start_time and
    finite finish times must be nonnegative. Input order must have strictly
    increasing times, positive durations and no overlapping action intervals.
    Missing/nonfinite truth, steering, timing or used geometry raises ValueError
    with row/field context; no invalid samples are silently filtered or zeroed.
    The input rows, geometry and envelope are not modified.
    """
    start_time = _finite(start_time, 'start_time')
    if start_time < 0:
        raise ValueError('start_time must be nonnegative')
    if end_times is None:
        end_times = {}
    elif not isinstance(end_times, Mapping):
        if len(end_times) != 2:
            raise ValueError('end_times needs def, att times')
        end_times = dict(zip(_CARS, end_times))
    if set(end_times) - set(_CARS):
        raise ValueError('end_times keys must be def and att')
    ends = {}
    for car in _CARS:
        end = end_times.get(car)
        ends[car] = None if end is None else _finite(end, f'end_times.{car}')
        if ends[car] is not None and ends[car] < 0:
            raise ValueError(f'end_times.{car} must be nonnegative')

    rows = list(rows)
    times = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f'row {index}: expected a mapping')
        t = _field(row, 'time', index)
        if t < 0 or (times and t <= times[-1]):
            raise ValueError(f'row {index} time must be nonnegative and strictly increasing')
        times.append(t)

    samples = {car: [] for car in _CARS}
    excluded = {car: dict(before_start=0, at_or_after_finish=0) for car in _CARS}
    sources = dict(action_elapsed=0, next_time_gap=0)
    for index, (row, t) in enumerate(zip(rows, times)):
        if 'action_elapsed' in row:
            duration = _field(row, 'action_elapsed', index)
            sources['action_elapsed'] += 1
        elif index + 1 < len(rows):
            duration = times[index + 1] - t
            sources['next_time_gap'] += 1
        else:
            raise ValueError(f'row {index}: missing action_elapsed and no next time gap; '
                             'final interval duration is unknown')
        end = _finite(t + duration, f'row {index} interval end')
        if duration <= 0 or end <= t:
            raise ValueError(f'row {index} action duration must be positive')
        if (index + 1 < len(rows) and end > times[index + 1]
                and not math.isclose(end, times[index + 1], rel_tol=0., abs_tol=1e-9)):
            raise ValueError(f'row {index} action_elapsed overlaps next row time')
        for car, (prefix, steering_field) in _CARS.items():
            state = {name: _field(row, f'{prefix}_{name}', index)
                     for name in ('s', 'd', 'yaw', 'speed')}
            steering = _field(row, steering_field, index)
            if t < start_time:
                excluded[car]['before_start'] += 1
                continue
            finish = ends[car]
            if finish is not None and t >= finish:
                excluded[car]['at_or_after_finish'] += 1
                continue
            stop = end if finish is None else min(end, finish)
            s = state['s']
            try:
                reference = _finite(envelope.reference(s), 'reference')
                slope = _finite(envelope.reference_slope(s), 'reference_slope')
                curvature = _finite(line.curvature(s), 'curvature')
                heading = _finite(line.heading(s), 'line heading')
                metric = _finite(1 - curvature * reference, 'reference tangent metric')
                reference_heading = heading + math.atan2(slope, metric)
                error = _finite(state['yaw'] - reference_heading, 'unwrapped heading error')
                heading_error = (error + math.pi) % (2 * math.pi) - math.pi
                derror = _finite(state['d'] - reference, 'derror')
            except ValueError as exc:
                raise ValueError(f'row {index} {car} geometry: {exc}') from exc
            samples[car].append(dict(index=index, time=t, end=stop, duration=stop - t,
                derror=derror, heading_error=heading_error, speed=state['speed'],
                steering=steering, corner=(abs(curvature) >= CURVATURE_THRESHOLD
                    or abs(slope) >= REFERENCE_SLOPE_THRESHOLD)))

    report: dict[str, Any] = dict(samples=len(rows), duration_sources=sources, parameters=dict(
        start_time=start_time, end_times=ends,
        curvature_threshold=CURVATURE_THRESHOLD,
        reference_slope_threshold=REFERENCE_SLOPE_THRESHOLD,
        slow_speed_threshold=SLOW_SPEED_THRESHOLD, steering_deadband=STEERING_DEADBAND,
        corner_rule='abs(curvature) >= threshold OR abs(reference_slope) >= threshold',
        reference_heading='line.heading(s) + atan2(reference_slope, 1-curvature*reference_d)',
        error_weighting='actual interval duration, clipped at per-car finish',
        p95='absolute-error inverse duration-weighted empirical CDF',
        time_window='start_time <= row.time < end_time; no pre-start interval carry-in',
        duration_policy='action_elapsed else next time gap; unknown final interval rejected',
        invalid_samples='reject with ValueError, including rows outside time windows',
        steering_policy='contiguous adjacent rows; deadband retains last significant sign; '
                        'reversals are descriptive, not a bad-chatter classification',
        columns={car: dict(truth_prefix=prefix, steering=steering)
                 for car, (prefix, steering) in _CARS.items()}))
    for car in _CARS:
        report[car] = _summarize(samples[car])
        report[car]['excluded_samples'] = excluded[car]
        report[car]['corner'] = _summarize([sample for sample in samples[car] if sample['corner']])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path, help='saved rival action CSV')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('rival_advanced_fast.yaml'))
    parser.add_argument('--output', type=Path, help='write JSON here instead of stdout')
    parser.add_argument('--start-time', type=float, default=2.)
    parser.add_argument('--def-end-time', type=float, help='defender finish time, exclusive')
    parser.add_argument('--att-end-time', type=float, help='attacker finish time, exclusive')
    args = parser.parse_args(argv)

    # Keep the generic evaluator stdlib-only; the CLI loads geometry, not Gym or a viewer.
    import yaml
    from rival_baseline import load
    from rival_robust import RobustConfig, TrackEnvelope

    try:
        raw = yaml.safe_load(args.config.read_text())
        cfg, line, track, _, _ = load(args.config)
        robust = dict(raw.get('robust', {}))
        if robust.get('boundaries_path'):
            robust['boundaries_path'] = str((args.config.parent / robust['boundaries_path']).resolve())
        rc = RobustConfig.from_dict(robust)
        envelope = TrackEnvelope(line, track, cfg, rc, boundaries_path=rc.boundaries_path)
        with args.csv.open(newline='') as stream:
            report = corner_metrics(csv.DictReader(stream), line, envelope,
                start_time=args.start_time,
                end_times={'def': args.def_end_time, 'att': args.att_end_time})
        report['source'] = dict(csv=str(args.csv.resolve()), config=str(args.config.resolve()),
                                configuration=raw)
        payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output is None:
            print(payload, end='')
        else:
            args.output.write_text(payload)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return report


if __name__ == '__main__':
    main()
