import numpy as np
from utils import get_distance_matrix


def ant_colony_optimization(scenario, discount_factor=9/10, beta=1):
    # https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

    # discount_factor: the ratio in which new pheromones replace
    # old ones in each time step.
    # beta: the amount of importance the length of an edge has over the ant's decisions.

    # a matrix holding the distance between each two nodes
    edge_lengths = get_distance_matrix(scenario)

    # pheromone amount in each edge
    pheromones = get_initial_pheromones(edge_lengths)

    # variables to keep track of the pheromone convergence
    no_convergence_counter, best_convergence_score = test_convergence(pheromones, 0, 0)

    while not did_fully_converge(pheromones) and no_convergence_counter < 10:
        # let the ants go wild and release pheromones
        ant_routes = release_ants(pheromones, edge_lengths, num_ants=scenario.shape[0], beta=beta)
        pheromone_update_matrix = calculate_produced_pheromones(ant_routes, edge_lengths)

        # normalize the old and new pheromones according to the discount factor
        pheromones *= discount_factor
        pheromone_update_matrix /= (pheromone_update_matrix.sum() / (scenario.shape[0] * (1 - discount_factor)))

        # merge old pheromones with new pheromones
        pheromones += pheromone_update_matrix

        # check convergence
        no_convergence_counter, best_convergence_score = test_convergence(pheromones, no_convergence_counter, best_convergence_score)

    return pheromones_to_route(pheromones)


def pheromones_to_route(pheromones):
    # follow the strongest pheromone track to generate a route

    route = [0]

    while len(route) < pheromones.shape[0]:
        options = get_node_edges(pheromones, route[-1])
        options[route] = 0
        route.append(options.argmax())

    return route


def did_fully_converge(pheromones):
    # go over all nodes and check the their second most pheromonized edge,
    # if its lower than 0.5, there is no convergence (pheromone values generally
    # aren't supposed to go above 1 although technically possible)
    for options in [get_node_edges(pheromones, i) for i in range(pheromones.shape[0])]:
        if np.partition(options.flatten(), -2)[-2] < 0.5:
            return False
    return True


def get_initial_pheromones(edge_lengths):
    # using the distances between nodes as initial pheromone values
    pheromones = edge_lengths.copy()
    # emptying the lower left triangle of the array
    pheromones[np.tril_indices(edge_lengths.shape[0], -1)] = 0
    # normalize
    pheromones /= (pheromones.sum() / edge_lengths.shape[0])
    return pheromones


def release_ants(pheromones, edge_lengths, num_ants, beta):
    # let n ants walk and produce routes
    return [produce_ant_route(pheromones, edge_lengths, beta) for _ in range(num_ants)]


def produce_ant_route(pheromones, edge_lengths, beta):
    route = [np.random.randint(pheromones.shape[0])]

    for _ in range(pheromones.shape[0] - 1):
        next_node = get_next_node(pheromones, edge_lengths, route, beta)
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


def get_next_node(pheromones, edge_lengths, route, beta):
    # gather all edge data of current node
    edge_pheromones = get_node_edges(pheromones, route[-1])
    edge_lengths = get_node_edges(edge_lengths, route[-1])

    # generate a probability vector for the ant to choose its next edge from
    probability_vector = generate_ant_probability_vector(edge_pheromones, edge_lengths, route, beta)

    # choose one at random using the normalized pheromones as distributions
    return np.random.choice(pheromones.shape[0], 1, p=probability_vector)[0]


def generate_ant_probability_vector(pheromone_vec, length_vec, route, beta):
    # generate a probability vector for the ant to choose its next edge to walk.
    # ants are more likely to choose shorter edges with more pheromones
    probability_vector = pheromone_vec

    # exclude edges leading to already visited nodes
    probability_vector[route] = 0

    # divide the pheromones by the edge lengths
    probability_vector[probability_vector > 0] /= length_vec[probability_vector > 0] ** beta

    # normalize vector to sum of 1
    probability_vector /= probability_vector.sum()

    return probability_vector


def get_node_edges(edge_matrix, node):
    # get the pheromone values of the node's edges
    return np.concatenate((edge_matrix[:node + 1, node], edge_matrix[node, node + 1:]))


def softmax(vector):
    pre_normalized = np.exp(vector - np.max(vector))
    return pre_normalized / pre_normalized.sum()


def test_convergence(pheromones, convergence_counter, old_convergence_score):
    # calculate a score for how polarized are the pheromone values towards 0 and high numbers.
    # if the score is higher, reset the counter and keep the new highest score
    # otherwise increase the counter by 1

    new_convergence_score = np.sum(pheromones ** 2) / pheromones.shape[0]

    if new_convergence_score > old_convergence_score:
        return 0, new_convergence_score

    return convergence_counter + 1, old_convergence_score

