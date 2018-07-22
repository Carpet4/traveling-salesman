import numpy as np
from utils import time_stamp, xy_route_to_indices_route, random_arange
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


def find_segment_flip(route):
    # find a segment of the route to flip so the route length is shortened

    for i in random_arange(len(route) - 1):
        for j in random_arange(i + 3, len(route) + 1):
            # indices of the nodes involved
            n1, n2, n3, n4 = np.array([i, i + 1, j - 1, j]) % len(route)

            # calculate the magnitudes of both the newly formed edges and the edges removed by the segment_flip
            matrix = route[[n1, n2, n1, n3]] - route[[n3, n4, n2, n4]]
            magnitudes = np.linalg.norm(matrix, axis=1)

            # subtract the removed edges from the added ones
            distance_change = (magnitudes[:2] - magnitudes[2:]).sum()

            # annoying conditioning since np.linalg isn't 100% precise..
            # can give different results for same spots depending on order
            if distance_change < 0 and (i != 0 or j != len(route)):
                return i, j
    return None, None


def find_pop(route):
    # find a node that is better of at a different location in the route

    for i in random_arange(len(route)):
        # a list of available spots, removing the identity pop and pops that
        # are identical to segment flips.
        available_spots = np.random.permutation([x for x in range(len(route)) if 1 < (i - x) % len(route) < len(route) - 2])

        for j in available_spots:
            # indices of the nodes involved, n0 is the popped one
            n0, n1, n2, n3, n4 = np.array([i, i - 1, i + 1, j - 1, j]) % len(route)

            # calculate the magnitudes of both the newly formed edges and the edges removed by the pop
            matrix = route[[n1, n3, n4, n3, n1, n2]] - route[[n2, n0, n0, n4, n0, n0]]
            magnitudes = np.linalg.norm(matrix, axis=1)

            # subtract the removed edges from the added ones
            distance_change = (magnitudes[:3] - magnitudes[3:]).sum()

            if distance_change < 0:
                return i, j
    return None, None
