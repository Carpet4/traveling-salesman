import numpy as np
from .ant_colony_optimization import calculate_produced_pheromones, did_converge


def test_calculate_produced_pheromones():
    # given ant routes and distance between each two nodes,
    # calculate the produced pheromones

    # the routes for the ants to take
    ant_routes = [[4, 3, 2, 1, 0], [0, 2, 1, 4, 3]]

    # distance between each two nodes
    edge_lengths = np.array([[i * j for i in range(1, 6)] for j in range(1, 6)])
    np.fill_diagonal(edge_lengths, 0)

    # expected resulted pheromones
    a, b = 1/45, 1/43
    c = a + b
    expected_result = [[0, a, b, b, a],
                       [0, 0, c, 0, b],
                       [0, 0, 0, a, 0],
                       [0, 0, 0, 0, c],
                       [0, 0, 0, 0, 0]]

    # compare the expected to the result
    produced_pheromones = calculate_produced_pheromones(ant_routes, edge_lengths)
    assert(np.allclose(expected_result, produced_pheromones))


def test_did_converge():
    # convergence occurs of each node has two connected edges
    # with over 0.5
    converged_pheromones = np.array([[0.0, 0.1, 0.6, 0.7],
                                     [0.0, 0.0, 0.8, 0.6],
                                     [0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0]])

    assert (did_converge(converged_pheromones) is True)

    non_converged_pheromones = np.array([[0.0, 0.1, 0.4, 0.7],
                                         [0.0, 0.0, 0.8, 0.6],
                                         [0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0]])

    assert (did_converge(non_converged_pheromones) is False)
