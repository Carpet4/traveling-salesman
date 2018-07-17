import random as rndm


# generates a route at random
def random_walk(scenario):
    return rndm.sample(list(range(len(scenario))), len(scenario))