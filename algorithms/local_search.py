import numpy as np
from utils import time_stamp, xy_route_to_indices_route, find_segment_flip, find_pop
from .random_walk import random_walk
from .greedy import greedy


def local_search(scenario, initiate_greedy=True, use_segment_flip=True, use_pop=True, time_limit=np.inf):

    stop_time = time_stamp() + time_limit

    # initiate a route
    if initiate_greedy:
        route = scenario[greedy(scenario)]
    else:
        route = scenario[random_walk(scenario)]

    while time_stamp() < stop_time:
        if use_segment_flip:
            # the two endpoints of the segment to flip
            segment_start, segment_end = find_segment_flip(route)

            if segment_start is not None:
                route[segment_start + 1: segment_end] = route[segment_end - 1: segment_start: -1]
                continue

        if use_pop:
            pop_from, place_at = find_pop(route)
            # if found a pop that shortens the route
            if pop_from is not None:
                # simulating a python list pop() and insert() using math

                if pop_from > place_at:
                    route[place_at:pop_from + 1] = route[np.r_[pop_from, place_at:pop_from]]
                else:
                    route[pop_from:place_at] = route[np.r_[pop_from + 1:place_at, pop_from]]

                continue
        break
    return xy_route_to_indices_route(scenario, route)
