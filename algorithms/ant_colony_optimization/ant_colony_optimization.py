# https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
import numpy as np
from utils import get_distance_matrix

def ant_colony_optimization(scenario):
    pheromones = get_initial_pheromones()

    while(True):  # TODO: replace true with some condition
        ant_routes = release_ants(pheromones)
        pheromone_update_matrix = calculate_produced_pheromones(ant_routes)
        pheromones += pheromone_update_matrix
        pheromones /= pheromones.sum() * scenario.shape[0]


def get_initial_pheromones(scenario):
    # using the distances between nodes as initial pheromone values
    pheromones = get_distance_matrix(scenario)
    # emptying the lower left triangle of the array
    pheromones[np.tril_indices(scenario.shape[0], -1)] = 0

    return pheromones



def produce_ant_route(pheromones):
    route = [np.random.randint(pheromones.shape[0])]

    for _ in range(pheromones.shape[0] - 1):
        next_node = get_next_node(pheromones, route)
        route.append(next_node)

    return route


def get_next_node(pheromones, route):
    # gather all the edges from the current node
    options = np.concatenate(pheromones[:route[-1] + 1, route[-1]], pheromones[route[-1], route[-1]+1:])
    # exclude the ones leading to already visited nodes
    options[route] = 0
    # normalize to sum of 1
    probability_vector = options / options.sum()
    # choose one at random using the normalized pheromones as distributions
    return np.random.choice(options.shape[0], 1, probability_vector)[0]