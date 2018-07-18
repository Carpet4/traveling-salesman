import numpy as np
from utils import calculate_journey_distance, generate_scenario
from .simulated_annealing import get_distance_normalizer, get_initial_energies, swap_nodes
# test that energies are calculated correctly


def test_tt():
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
        # swap'em
        temp_scenario = np.copy(scenario)
        swap_nodes(temp_scenario, i)
        # compare'em
        post_distance = calculate_journey_distance(temp_scenario) / normalizer
        assert(round(pre_distance + energies[i], 3) == round(post_distance, 3))
