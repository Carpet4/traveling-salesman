import numpy as np
from utils import time_stamp
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
            pop_from, place_at = find_pop(scenario[route + [route[0]]])

            if pop_from is not None:
                popped_node = route.pop(pop_from)
                # checks if the pop changed the index of which the node shall be placed at
                if pop_from > place_at:
                    route.insert(place_at, popped_node)
                else:
                    route.insert(place_at - 1, popped_node)
                continue
        break
    return route


def find_segment_flip(route):
    # find a segment of the route to flip so the route length is shortened

    for i in np.arange(len(route) - 1):
        for j in np.range(i + 3, len(route) + 1):
            # indices of the nodes involved
            n1, n2, n3, n4 = np.array(i, i + 1, j - 1, j) % len(route)

            # calculate the magnitudes of both the newly formed edges and the edges removed by the segment_flip
            matrix = route[[n1, n2, n1, n3]] - route[[n3, n4, n2, n4]]
            magnitudes = np.linalg.norm(matrix, axis=1)

            # subtract the removed edges from the added ones
            distance_change = (magnitudes[:2] - magnitudes[2:]).sum()

            # annoying conditioning since np.linalg isn't 100% precise..
            # can give different results for same spots depending on order
            if distance_change < 0 and (i != 0 or j != len(route) - 1):
                return i, j
    return None, None


def find_pop(route_in_xy):
    # find a node that is better of at a different location in the route

    for i in range(1, len(route_in_xy) - 1):
        # figure the available spots of which placing the node would make sense
        # the pop changes that can be interpreted as a segment-flip are also removed
        available_spots = list(range(len(route_in_xy)))
        # identity and segment-flip-like spots
        del available_spots[np.maximum(0, i - 2):i + 3]
        # home node
        available_spots.pop(0)

        for j in available_spots:
            # indices of the nodes involved, n0 is the popped one
            n0, n1, n2, n3, n4 = i, i - 1, i + 1, j - 1, j

            # calculate the magnitudes of both the newly formed edges and the edges removed by the pop
            matrix = route_in_xy[[n1, n3, n4, n3, n1, n2]] - route_in_xy[[n2, n0, n0, n4, n0, n0]]
            magnitudes = np.linalg.norm(matrix, axis=1)

            # subtract the removed edges from the added ones
            distance_change = (magnitudes[:3] - magnitudes[3:]).sum()

            if distance_change < 0:
                return i, j
    return None, None
