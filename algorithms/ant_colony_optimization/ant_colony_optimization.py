# https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
import numpy as np
from utils import get_distance_matrix

def ant_colony_optimization(scenario, evaporation_speed = 1/10):
    #

    # evaporation_speed: the ratio in which new pheromones replace
    # old ones in each time step

    # a matrix holding the distance between each two nodes
    edge_lengths = get_distance_matrix(scenario)

    # pheromone amount in each edge
    pheromones = get_initial_pheromones(edge_lengths)

    while not did_converge(pheromones):  # TODO: replace true with some condition
        # let the ants go wild and release pheromones
        ant_routes = release_ants(pheromones)
        pheromone_update_matrix = calculate_produced_pheromones(ant_routes, edge_lengths)

        # normalize the new pheromones and add them to the total pheromones
        pheromone_update_matrix /= pheromone_update_matrix.sum() * scenario.shape[0] * evaporation_speed
        pheromones += pheromone_update_matrix

        # normalize the pheromones to avoid overflow
        pheromones /= pheromones.sum() * scenario.shape[0]

    return pheromones_to_route(pheromones)


def get_initial_pheromones(edge_lengths):
    # using the distances between nodes as initial pheromone values
    pheromones = edge_lengths.copy()
    # emptying the lower left triangle of the array
    pheromones[np.tril_indices(edge_lengths.shape[0], -1)] = 0

    return pheromones


def release_ants(pheromones, num_ants):
    # let n ants walk and produce routes
    return [produce_ant_route(pheromones) for _ in range(num_ants)]


def produce_ant_route(pheromones):
    route = [np.random.randint(pheromones.shape[0])]

    for _ in range(pheromones.shape[0] - 1):
        next_node = get_next_node(pheromones, route)
        route.append(next_node)

    return route


def calculate_produced_pheromones(ant_routes, edge_lengths):
    # array to hold the new pheromones
    produced_pheromones = np.zeros(edge_lengths.shape)
    # the length of each route taken by an ant
    route_lengths = edge_lengths[ant_routes, np.roll(ant_routes, 1, axis=1)].sum(axis=1)
    # for each route, add pheromones to the its edges.
    # the longer the route, the less pheromones each edge gets
    for ant_route, route_length in zip(ant_routes, route_lengths):
        produced_pheromones[ant_route, np.roll(ant_route, 1)] += 1 / route_length

    # since the array is a mirror of it self on the diagonal, add the lower left pheromones
    # to the upper right, and then zero-out the lower left
    right_triangle_indices = np.triu_indices(edge_lengths.shape[0], 1)
    produced_pheromones[right_triangle_indices] += produced_pheromones.T[right_triangle_indices]
    produced_pheromones.T[right_triangle_indices] = 0

    return produced_pheromones


def get_next_node(pheromones, route):
    # gather all the edges from the current node
    options = np.concatenate(pheromones[:route[-1] + 1, route[-1]], pheromones[route[-1], route[-1]+1:])
    # exclude the ones leading to already visited nodes
    options[route] = 0
    # normalize to sum of 1
    probability_vector = options / options.sum()
    # choose one at random using the normalized pheromones as distributions
    return np.random.choice(options.shape[0], 1, probability_vector)[0]