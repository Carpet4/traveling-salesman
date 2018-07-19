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
