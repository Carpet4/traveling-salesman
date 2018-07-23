import numpy as np
from utils import time_stamp, xy_route_to_indices_route, find_segment_flip, find_pop, get_average_edge_length, calculate_journey_distance
from .tools import get_temperature


def simulated_annealing(scenario, time_limit=40000, use_flips=True, use_pops=True):
    # simulated annealing (described here: # https://en.wikipedia.org/wiki/Simulated_annealing)
    # is generally an optimization algorithm that tweaks the state by a little each time
    # until reaching optimum. in order not to get stuck in local optima, the process sometimes
    # chooses tweaks with high energy (bad score) in order to shake things up and allow for new pathways
    # as the optimization proceeds, the "temperature" lowers and high energy tweaks become less and less
    # likely.

    # the advanced version is basically an improvement over local_search, it checks for both segment
    # flips and node pops.

    # time_limit: amount of allowed computation time in milliseconds

    start_time = time_stamp()
    counter = 0
    # normalizer that makes the scale (width and height) of the scenario irrelevant
    distance_normalizer = get_average_edge_length(scenario)

    # initiate a route randomly
    route = scenario.copy()
    np.random.shuffle(route)

    while time_stamp() < start_time + time_limit:
        # figure the current temperature
        stamp = time_stamp()
        temperature = get_temperature(stamp - start_time, time_limit)

        # get a random threshold, route length changes below that threshold will be accepted
        random_threshold = get_random_threshold(temperature, distance_normalizer)

        if use_flips:
            # the two endpoints of the segment to flip
            segment_start, segment_end = find_segment_flip(route, threshold=random_threshold)

            # if succeeded to find a flip below the threshold
            if segment_start is not None:
                route[segment_start + 1: segment_end] = route[segment_end - 1: segment_start: -1]
                counter += 1

        if use_pops:
            pop_from, place_at = find_pop(route, threshold=random_threshold)
            # if succeeded to find a pop below the threshold
            if pop_from is not None:
                # simulating a python list pop() and insert() using math

                if pop_from > place_at:
                    route[place_at:pop_from + 1] = route[np.r_[pop_from, place_at:pop_from]]
                else:
                    route[pop_from:place_at] = route[np.r_[pop_from + 1:place_at, pop_from]]
                continue
    # turn the coordinates route to indices route each pointing
    # to its corresponding node in the original scenario array.
    return xy_route_to_indices_route(scenario, route)


def get_random_threshold(temperature, distance_normalizer):
    # any change to the route length below this threshold will be accepted
    # this is sort of the inverse function to "get_thresholds()" in the basic form,
    # since there the operation is done on the route mutation scores instead of the
    # random number.
    return -1 * temperature * np.log(np.random.random()) * distance_normalizer
