import numpy as np


# an algorithm that always moves to the closest node
def greedy(scenario):
    route = [0]

    # while there are still unvisited nodes, add the nearest node to the route
    while len(route) < len(scenario):
        remaining_nodes = np.arange(len(scenario))[~np.isin(range(len(scenario)), route)]
        route.append(
            remaining_nodes[
                find_nearest_node(
                    scenario[route[-1]],
                    scenario[remaining_nodes]
                )
            ]
        )
    return route


def find_nearest_node(current_location, nodes):
    return np.argmin(np.linalg.norm(nodes - current_location, axis=1))
