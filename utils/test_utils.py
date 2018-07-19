from .utils import calculate_journey_distance, xy_route_to_indices_route, generate_scenario
import numpy as np


def test_calculate_journey_distance():
    assert (calculate_journey_distance(np.array([[0, 0], [4, -3], [8, 0]])) == 18)


def test_xy_route_to_indicies_route():
    scenario = generate_scenario(10)
    route = np.random.permutation(np.arange(10))
    xy_route = scenario[route]
    recreated_route = xy_route_to_indices_route(scenario, xy_route)
    assert (np.array_equal(route, np.array(recreated_route)))
