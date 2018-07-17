import numpy as np


def generate_scenario(num_nodes, map_size=100):
    return np.random.random((num_nodes, 2)) * map_size


def calculate_journey_distance(scenario, route):
    return np.linalg.norm(scenario[route] - scenario[route + [route[0]]][1:], axis=1).sum()


assert(calculate_journey_distance(np.array([[0, 0], [8, 0], [4, -3]]), [0, 2, 1]) == 18)
