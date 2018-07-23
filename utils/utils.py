import numpy as np
from datetime import datetime


def generate_scenario(num_nodes, map_size=100):
    return np.random.random((num_nodes, 2)) * map_size


def calculate_journey_distance(route_in_xy):
    return np.linalg.norm(route_in_xy - np.roll(route_in_xy, 1, axis=0), axis=1).sum()


def time_stamp():
    return datetime.now().timestamp() * 1000


# translate route between coordinates to route between indices pointing to
# these coordinates in the scenario
def xy_route_to_indices_route(scenario, xy_route):
    return [np.where((scenario == node).all(axis=1))[0][0] for node in xy_route]


def random_arange(*args):
    return np.random.permutation(np.arange(*args))


def find_segment_flip(route, threshold=0):
    # find a segment of the route to flip so the route length is shortened

    # threshold: any change in the route below that number is being accepted

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
            if distance_change < threshold and (i != 0 or j != len(route)):
                return i, j
    return None, None


def find_pop(route, threshold=0):
    # find a node that is better off at a different location in the route
    # *** pops that can be represented as a segment flip are excluded (example: 1 to 3)

    # threshold: any change in the route below that number is being accepted

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

            if distance_change < threshold:
                return i, j
    return None, None


def get_average_edge_length(scenario, num_samples=5):
    # calculate the average distance between nodes in the scenario
    # by taking random samples of it and averaging the distances between each two consecutive nodes
    # num_samples: the amount of random samples to test, the more - the more accurate

    return np.average(
        [calculate_journey_distance(np.random.permutation(scenario)) for _ in range(num_samples)]
    ) / scenario.shape[0]


def get_distance_matrix(scenario):
    # figure the distance between each two nodes in the scenario
    return np.linalg.norm(scenario[:, :] - scenario[:, None], axis=2)
