from .utils import calculate_journey_distance, xy_route_to_indices_route, generate_scenario, get_distance_matrix
import numpy as np


def test_calculate_journey_distance():
    assert (calculate_journey_distance(np.array([[0, 0], [4, -3], [8, 0]])) == 18)


def test_xy_route_to_indices_route():
    scenario = generate_scenario(10)
    route = np.random.permutation(np.arange(10))
    xy_route = scenario[route]
    recreated_route = xy_route_to_indices_route(scenario, xy_route)
    assert (np.array_equal(route, np.array(recreated_route)))


def test_get_distance_matrix():
    scenario = np.array([[0, 0], [3, 4], [0, 8], [-3, 4]])

    expected_result = [[0, 5, 8, 5],
                       [5, 0, 5, 6],
                       [8, 5, 0, 5],
                       [5, 6, 5, 0]]
    distance_matrix = get_distance_matrix(scenario)
    assert (np.allclose(distance_matrix, expected_result))