import numpy as np


def generate_scenario(num_nodes, map_size=100):
    return np.random.random((num_nodes, 2)) * map_size


def calculate_journey_distance(route_in_xy):
    return np.linalg.norm(route_in_xy - np.roll(route_in_xy, 1, axis=0), axis=1).sum()


assert(calculate_journey_distance(np.array([[0, 0], [4, -3], [8, 0]])) == 18)
