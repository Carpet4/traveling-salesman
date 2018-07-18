import random as rndm


def random_walk(scenario):
    # generate a route at random
    return rndm.sample(list(range(len(scenario))), len(scenario))