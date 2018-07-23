import numpy as np
from utils import time_stamp, xy_route_to_indices_route, get_average_edge_length
from .tools import get_temperature


def simulated_annealing(scenario, time_limit=20000):
    # simulated annealing (described here: # https://en.wikipedia.org/wiki/Simulated_annealing)
    # is generally an optimization algorithm that tweaks the state by a little each time
    # until reaching optimum. in order not to get stuck in local optima, the process sometimes
    # chooses tweaks with high energy (bad score) in order to shake things up and allow for new pathways
    # as the optimization proceeds, the "temperature" lowers and high energy tweaks become less and less
    # likely.

    # the basic version only mutates the route using swaps, AKA a swap between two consecutive nodes

    # time_limit: amount of allowed computation time in milliseconds

    start_time = time_stamp()

    scenario_len = scenario.shape[0]

    # normalizer that makes the scale (width and height) of the scenario irrelevant
    distance_normalizer = get_average_edge_length(scenario)

    # initiate a route randomly
    route = scenario.copy()
    np.random.shuffle(route)

    # score of each possible swap
    energy_deltas = get_energy_deltas(route, distance_normalizer, np.arange(scenario_len))

    while time_stamp() < start_time + time_limit:
        # figure the current temperature
        stamp = time_stamp()
        temperature = get_temperature(stamp - start_time, time_limit)

        # get the thresholds for all swaps
        thresholds = get_thresholds(energy_deltas, temperature)

        # get a random number
        random_num = np.random.random()

        # get a random swap with threshold above the random number
        available_swaps = np.arange(thresholds.shape[0])[thresholds > random_num]

        if available_swaps.size > 0:
            winner_swap = np.random.choice(available_swaps, 1)[0]

            # swap the nodes
            i, j = swap_index_to_node_indices(winner_swap, scenario_len)
            route[[i, j]] = route[[j, i]]

            # recalculate the affected energies
            affected_indices = get_indices_affected_by_swap(winner_swap, scenario_len)
            new_energy_deltas = get_energy_deltas(route, distance_normalizer, affected_indices)
            energy_deltas[affected_indices] = new_energy_deltas

    # turn the coordinates route to indices route each pointing
    # to its corresponding node in the original scenario array.
    return xy_route_to_indices_route(scenario, route)


def swap_index_to_node_indices(i, scenario_length):
    # convert swap index to node indices to swap
    return i, (i + 1) % scenario_length


def get_thresholds(energies, temperature):
    # calculate a threshold for the activation of each route mutation
    return np.minimum(
        np.exp(-1 * energies / temperature),
        1
    )


def get_indices_affected_by_swap(swap_index, scenario_length):
    # figure which edges were changed by a certain swap
    return np.arange(swap_index - 2, swap_index + 3) % scenario_length


def get_energy_deltas(route, normalizer, relevant_indices):
    # calculate the change in energy (score) for each permutation of swapping two consecutive nodes
    # in the route

    # normalizer: a scalar to make sure the size of the scenario's field doesn't affect the energies.
    
    # create four rolled versions of the graph (a swap affects edges between 4 nodes)
    matrices = [np.roll(route, 1 - n, axis=0)[relevant_indices] for n in range(4)]
    
    # calculate the vectors of the two edges added and two edges removed by each swap
    # the four vector groups are piled up for efficiency
    piled_edge_vectors = np.concatenate((
        matrices[2] - matrices[0],
        matrices[3] - matrices[1],
        matrices[1] - matrices[0],
        matrices[3] - matrices[2]
    ))

    # figure the magnitude of each vector and split them back to initial four groups
    norms = np.linalg.norm(piled_edge_vectors, axis=1)
    unpiled_norms = np.split(norms, 4)

    # subtract the magnitudes of the two removed edges from the two added ones to get the total change of
    # journey length by each swap, then divide it by the normalizer.
    energies = (unpiled_norms[0] + unpiled_norms[1] - unpiled_norms[2] - unpiled_norms[3]) / normalizer

    return energies
