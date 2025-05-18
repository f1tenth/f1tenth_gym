from collections import deque
import time

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline


class Raceline:
    def __init__(self, raceline_path):

        # TODO: handle the case when there is no raceline

        data = np.genfromtxt(raceline_path, delimiter=';', dtype='<U32')
        self.n_points = len(data) - 1
        self.total_s = float(data[-1, 0])
        print(f'Number of raceline points: {self.n_points}')
        print(f'Raceline length: {self.total_s}')
        self.last_delta_s = self.total_s - float(data[-2, 0])
        print(f'Last delta s: {self.last_delta_s}')

        # The last point of the raceline is the same as the first point, with the exception of
        # the last point's length in order to obtain the last segment length between the
        # previous last and the first point.
        self.s = data[:, 0].astype(np.float32).tolist()
        self.x = data[:, 1].astype(np.float32).tolist()
        self.y = data[:, 2].astype(np.float32).tolist()
        # TODO: fix this upstream
        self.width_right = data[:, 4].astype(np.float32).tolist()
        self.width_left = data[:, 3].astype(np.float32).tolist()
        self.heading = data[:, 5].astype(np.float32).tolist()
        self.curvature = data[:, 6].astype(np.float32).tolist()
        self.speed = data[:, 7].astype(np.float32).tolist()

        # Compute the two splines of x and y based on s
        # These spline handle the periodicity of the raceline, so you can input values outside the [0, total_s] range
        self.x_spline = CubicSpline(self.s, self.x, bc_type='periodic')
        self.y_spline = CubicSpline(self.s, self.y, bc_type='periodic')
        self.width_right_spline = CubicSpline(self.s, self.width_right, bc_type='periodic')
        self.width_left_spline = CubicSpline(self.s, self.width_left, bc_type='periodic')
        self.heading_spline = CubicSpline(self.s, self.heading, bc_type='periodic')
        self.curvature_spline = CubicSpline(self.s, self.curvature, bc_type='periodic')
        self.speed_spline = CubicSpline(self.s, self.speed, bc_type='periodic')

        # Remove the last point of the raceline to avoid having two points with the same value
        self.s = self.s[:-1]
        self.x = self.x[:-1]
        self.y = self.y[:-1]
        self.width_right = self.width_right[:-1]
        self.width_left = self.width_left[:-1]
        self.heading = self.heading[:-1]
        self.curvature = self.curvature[:-1]
        self.speed = self.speed[:-1]

        self.zones_length = 2.0 # meters
        self.zones = {0: (0.0, self.zones_length),
                      1: (0.33 * self.total_s, 0.33 * self.total_s + self.zones_length),
                      2: (0.66 * self.total_s, 0.66 * self.total_s + self.zones_length)}
        self.visit_order = deque(maxlen=4)
        # Legend:
        # -1: not in any zone (unclaimed area)
        # 0: zone 0, starting zone
        # 1: zone 1, between zone 0 and zone 2
        # 2: zone 2, between zone 1 and zone 0
        self.current_zone = -1  # the zone the car is currently in
        self.start_time = None

        # Validate zones
        for zone, (start, end) in self.zones.items():
            if start < 0 or end > self.total_s:
                raise ValueError(f"Zone {zone} is out of raceline bounds: ({start}, {end})")
        for i in range(len(self.zones) - 1):
            current_zone = i
            current_start, current_end = self.zones[current_zone]
            next_zone = i + 1
            next_start, next_end = self.zones[next_zone]
            if current_end > next_start:
                raise ValueError(
                    f"Zone {current_zone} (end: {current_end}) overlaps with zone {next_zone} (start: {next_start})")

    def reset(self, s):
        # `s` is the starting position of the car in the raceline
        # TODO: update self.zones with zone 0 equal to the starting position

        self.visit_order.clear()
        self.current_zone = -1
        self.start_time = None

    def is_lap_completed(self, s):
        """
        Check if a lap is completed based on the current s coordinate.
        :param s: s coordinate
        """

        # Check if the car has completed a lap, also providing information about:
        # - time
        # - forward or backward orientation

        # Get zone the car is currently in
        car_zone = -1
        for zone, (start, end) in self.zones.items():
            if start < s < end:
                car_zone = zone
                break
        if car_zone == self.current_zone or car_zone == -1:
            # The car is in the same zone as before or not in any claimed zone
            self.current_zone = car_zone
            return {'lap_completed': False, 'lap_time': np.inf, 'lap_orientation': None}

        # The car is in a new zone and that zone is not -1
        self.current_zone = car_zone
        self.visit_order.append(car_zone)

        if len(self.visit_order) != 4:
            if car_zone == 0:
                self.start_time = time.time()

            # Not enough data to determine lap time
            return {'lap_completed': False, 'lap_time': np.inf, 'lap_orientation': None}

        if car_zone == 0:
            # Get time
            current_time = time.time()
            order = list(self.visit_order)

            # Forward orientation
            if order == [0, 1, 2, 0]:
                if self.start_time is None:
                    raise ValueError('self.start_time should not be None here')

                lap_time = current_time - self.start_time
                self.start_time = current_time

                return {'lap_completed': True, 'lap_time': lap_time, 'lap_orientation': 'forward'}

            elif order == [0, 2, 1, 0]:
                if self.start_time is None:
                    raise ValueError('self.start_time should not be None here')

                lap_time = current_time - self.start_time
                self.start_time = current_time

                return {'lap_completed': True, 'lap_time': lap_time, 'lap_orientation': 'backward'}

            self.start_time = time.time()

        return {'lap_completed': False, 'lap_time': np.inf, 'lap_orientation': None}


    def to_cartesian(self, s, d):
        """
        Convert the curvilinear coordinates (s, d) to Cartesian coordinates (x, y).
        :param s: s coordinate
        :param d: d coordinate
        :return: x and y coordinates
        """
        x_raceline = self.x_spline(s)
        y_raceline = self.y_spline(s)

        unit_normal = self.get_normal(s)

        p = np.array([x_raceline, y_raceline])
        # `d` is the distance from the raceline, so we need to move along the normal vector
        # `d` is positive on the right of the raceline
        p += d * unit_normal

        return float(p[0]), float(p[1])

    def get_nearest_index(self, x, y, previous_s, discretization_step=0.01):
        """
        Get the nearest index of the raceline to the given x and y coordinates in the curvilinear space.
        `previous_s` needs to be stored and handled by the owner of this class.
        :param x: x coordinate
        :param y: y coordinate
        :param previous_s: previous found s coordinate
        :param discretization_step: discretization step. lower values give more accuracy at the cost of performance
        :return: the nearest index of the raceline to the given x and y coordinates in the curvilinear space (s, d)
        """

        # - compute the distances between the given x and y coordinates and the raceline points
        # - find the local minima of the distances within the given radius. each local minima is a candidate (for plateaus, take the middle one, super rare)
        # - for each candidate in order of distances:
        #   - compute the `d` of (x,y) from the candidate
        #   - if the `d` is less than the maximum `d` of that candidate (no collision), then the candidate is the nearest point
        # - if no candidate is found (we are probably off track):
        #   - get the previous nearest index (`s`) and wrt that compute the closest current candidate
        #   - if no previous index is available (this should not happen, it means we are at the start of the algo execution),
        #     then return the index (`s`) of the closest candidate

        dx = [x - x_raceline for x_raceline in self.x]
        dy = [y - y_raceline for y_raceline in self.y]
        distances = np.hypot(dx, dy)

        # Find local minima
        candidates_indexes = find_local_minima(distances)
        if len(candidates_indexes) == 0:
            raise ValueError('It is not possible that a local minimum has not been found. Check the raceline/code.')

        # Order the indexes in terms of distance
        candidates_indexes.sort(key=lambda i: distances[i])
        # candidate_distances = np.empty(len(candidates_indexes), dtype=np.float64)
        # for idx in range(len(candidates_indexes)):
        #     candidate_distances[idx] = distances[candidates_indexes[idx]]
        # sorted_order = np.argsort(candidate_distances)
        # sorted_candidates_indexes = [candidates_indexes[i] for i in sorted_order]
        # candidates_indexes = sorted_candidates_indexes

        cache_results = {}
        for i in candidates_indexes:
            refined_s, _ = self.get_refined_s(x, y, i, discretization_step)

            d = self.get_refined_d(x, y, refined_s)

            cache_results[i] = (refined_s, d)

            # Check if the `d` is less than the maximum width of the raceline at that point
            if d > 0:
                if d < self.width_right_spline(refined_s):
                    return refined_s, d, 'normal'
            else:
                if -d < self.width_left_spline(refined_s):
                    return refined_s, d, 'normal'

        # If we are here, we are probably off track
        if previous_s is not None:
            # Convert all the candidates to s
            s_list = []
            for i in candidates_indexes:
                s_list.append(self.s[i])
                # Summing and subtracting the total_s to handle the periodicity of the raceline
                s_list.append(self.s[i] + self.total_s)
                s_list.append(self.s[i] - self.total_s)

            # Get the closest s to the previous s
            s_list = np.array(s_list)
            s_list_diff = np.abs(s_list - previous_s)
            closest_candidate_index = np.argmin(s_list_diff) // 3
            closest_index = candidates_indexes[closest_candidate_index]

            status = 'off_track'
        else:
            # If no previous s is available, return the s of the current closest candidate
            closest_index = candidates_indexes[0]
            status = 'no_previous_s'

        return cache_results[closest_index][0], cache_results[closest_index][1], status

    #@njit(cache=True)
    def get_normal(self, s):
        """
        Get the normal vector of the raceline at the given s coordinate.
        :param s: s coordinate
        :return: normal vector of the raceline at the given s coordinate
        """

        x_raceline = self.x_spline(s)
        y_raceline = self.y_spline(s)

        # Calculate the tangent vector for the raceline at the point s
        heading = self.heading_spline(s)
        # Move the point along the tangent vector
        tangent_x = x_raceline + np.cos(heading) * 1.0
        tangent_y = y_raceline + np.sin(heading) * 1.0
        tangent_dx = tangent_x - x_raceline
        tangent_dy = tangent_y - y_raceline
        unit_tangent = np.array([tangent_dx, tangent_dy])
        unit_tangent /= np.linalg.norm(unit_tangent)

        # Choosing the perpendicular vector to be to the right of the tangent (d positive on the right)
        # Therefore, `unit_normal` = (unit_tangent_y, -unit_tangent_x)
        # If you want the left normal, set `unit_normal` = (-unit_tangent_y, unit_tangent_x)
        unit_normal = np.array([tangent_dy, -tangent_dx])

        return unit_normal

    def get_refined_d(self, x, y, refined_s):
        """
        Get the refined `d` of the given x and y coordinates in the curvilinear space.
        :param x: x coordinate
        :param y: y coordinate
        :param refined_s: s coordinate
        :return: d of the given x and y coordinates in the curvilinear space
        """
        refined_x_raceline = self.x_spline(refined_s)
        refined_y_raceline = self.y_spline(refined_s)

        dx = x - refined_x_raceline
        dy = y - refined_y_raceline

        unit_normal = self.get_normal(refined_s)

        # Finally, get the `d` of the refined candidate, `d` = dot product of ((dx, dy), `unit_normal`)
        d = dx * unit_normal[0] + dy * unit_normal[1]

        return d

    def get_refined_s(self, x, y, candidate_index, discretization_step):
        # `previous_index` segment part
        start_idx = (candidate_index - 1) % self.n_points
        end_idx = candidate_index
        discretized_previous_segment = self.discretize_segment(start_idx, end_idx, discretization_step)

        best_s = None
        best_distance = np.inf
        for s in discretized_previous_segment:
            dx = x - self.x_spline(s)
            dy = y - self.y_spline(s)
            distance = np.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best_s = s

        # `next_index` segment part
        start_idx = candidate_index
        end_idx = (candidate_index + 1) % self.n_points
        discretized_next_segment = self.discretize_segment(start_idx, end_idx, discretization_step)

        for s in discretized_next_segment:
            dx = x - self.x_spline(s)
            dy = y - self.y_spline(s)
            distance = np.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best_s = s

        return best_s, best_distance

    def discretize_segment(self, start_index, end_index, discretization_step):
        """
        Discretize the segment between two indices of the raceline
        :param start_index: starting index
        :param end_index: ending index
        :param discretization_step: step size for discretization
        :return: discretized segment
        """
        start_s = self.s[start_index]
        if end_index == 0:
            end_s = self.total_s
        else:
            end_s = self.s[end_index]

        discretized_segment = np.arange(start_s, end_s, discretization_step)
        if discretized_segment[-1] != end_s:
            new_segment = np.empty(len(discretized_segment) + 1)
            new_segment[:-1] = discretized_segment
            new_segment[-1] = end_s
            discretized_segment = new_segment

        return discretized_segment


#@njit(cache=True)
def find_local_minima(distances):
    """
    Returns a list of indices corresponding to local minima.
    For any local minimum plateau, returns the middle index (average index rounded down).

    The signal is assumed to be circular.
    """

    if len(distances) == 0:
        raise ValueError("The input signal is empty.")

    n = len(distances)
    minima_indices = []
    i = 0

    # Helper function to get circular index
    circ = lambda idx: idx % n

    # Iterate over each index
    while i < n:
        # Get current value and its two neighbors (circular)
        prev_val = distances[circ(i - 1)]
        cur_val = distances[i]
        next_val = distances[circ(i + 1)]

        # If current value is distinctly lower than both neighbors, it's a local minimum
        if cur_val < prev_val and cur_val < next_val:
            minima_indices.append(i)
            i += 1
        # Check for a plateau: starts if current equals next and is lower than the neighbor before plateau and after plateau
        elif cur_val == next_val:
            # Identify the plateau segment. We'll scan forward while values are equal

            # Move j pointer
            j = i + 1
            iters = 1 # n points in plateau
            while j < i + n and distances[circ(j)] == cur_val:
                j += 1
                iters += 1

            # Determine the "neighbors" of the plateau:
            left_index = circ(i - 1)
            right_index = circ(j)
            left_val = distances[left_index]
            right_val = distances[right_index]

            # Plateaus count as a local minimum if cur_val < both neighbors
            if cur_val < left_val and cur_val < right_val:
                length = iters
                # Choose the middle: if even number of points, the formula picks the lower middle
                # We compute the logical middle index (mod n).
                mid_offset = (length - 1) // 2
                mid_index = circ(i + mid_offset)
                minima_indices.append(mid_index)
            elif iters == n:
                # If the plateau spans the entire signal, you might either pick one index or ignore
                minima_indices.append(i)

            # Skip the entire plateau region
            i = j
        else:
            i += 1

    return minima_indices


# TODO: is the raceline in the correct reference frame? also heading? yes, besides width left/right

# raceline = Raceline('/home/edo/unibo/driverless/repos/ubm-f1tenth/raceline/csv/TUM_raceline/TUM_raceline_250514_084549.csv')
#
# import matplotlib.pyplot as plt
#
# #plt.plot(raceline.x, raceline.y, 'o')
# #plt.plot(raceline.x_spline(raceline.s), raceline.y_spline(raceline.s), 'r-')
# #plt.scatter(raceline.x[0], raceline.y[0], c='g', s=100, label='Start')
# s_linspace = np.linspace(- raceline.total_s * 2 - 10, raceline.total_s * 2 + 10, 1000)
# plt.plot(raceline.x_spline(s_linspace), raceline.y_spline(s_linspace), 'r-')
# # plt.plot(s_linspace, raceline.curvature_spline(s_linspace), 'b-')
# plt.show()
