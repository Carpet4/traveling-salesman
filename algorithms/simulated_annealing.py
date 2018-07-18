# https://en.wikipedia.org/wiki/Simulated_annealing
import numpy as np
from utils import calculate_journey_distance, generate_scenario
# notes to self:
# easiest way might be to calculate all swap energies each time and
# pick the worst one that has p() above np.random[0, 1)


def simulated_annealing(scenario):
    distance_normalizer = get_distance_normalizer(scenario)
    energies = get_initial_energies(scenario, distance_normalizer)



def neighbor(scenario):
    return


def p(new_energy, old_energy, temperature):
    return np.minimum(
        np.exp(-1 * (old_energy-new_energy) / temperature),
        1
    )


def temperature(time_passed, total_time):
    return


def get_distance_normalizer(scenario, num_samples=5):
    # calculate the average distance between nodes in the scenario
    # by taking random samples of it and averaging the distances between each two consecutive nodes
    # num_samples: the amount of random samples to test, the more - the more accurate

    return np.average(
        [calculate_journey_distance(np.random.permutation(scenario)) for _ in range(num_samples)]
    ) / scenario.shape[0]


def get_initial_energies(scenario, normalizer):
    # calculate the change in energy (score) for each permutation of swapping two consecutive nodes
    # in the route

    # normalizer: a scalar to make sure the size of the scenario's field doesn't affect the energies.
    
    # create four rolled versions of the graph (a swap affects edges between 4 nodes)
    matrices = [np.roll(scenario, 1 - n, axis=0) for n in range(4)]
    
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


# check that energies are calculated correctly

# generate a scenario
num_nodes = 10
scenario = generate_scenario(num_nodes)

# figure the average distance between nodes that is used to
# normalize energies.
normalizer = get_distance_normalizer(scenario)

# initial energies for the scenario
energies = get_initial_energies(scenario, normalizer)

# figure the journey distance before swapping any nodes
# divide by normalizer to match scale of energies
pre_distance = calculate_journey_distance(scenario) / normalizer

# swap nodes in various "edge cases" (pun not intended)
# and make sure the change in distance divided by the normalizer is the same as the energy assigned to the swap
for i in [4, 0, num_nodes-2, num_nodes-1]:
    # the node to swap with
    j = (i + 1) % num_nodes
    # swap'em
    temp_scenario = np.copy(scenario)
    temp_scenario[[i, j]] = temp_scenario[[j, i]]
    # compare'em
    post_distance = calculate_journey_distance(temp_scenario) / normalizer
    assert(round(pre_distance + energies[i], 3) == round(post_distance, 3))
