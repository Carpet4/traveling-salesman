# https://en.wikipedia.org/wiki/Simulated_annealing
import numpy as np

# notes to self:
# easiest way might be to calculate all swap energies each time and
# pick the worst one that has p() above np.random[0, 1)
# energies should some how be normalized by SIZE OF SCENARIO/average distance between nodes?


def neighbor(scenario):
    return


def p(new_energy, old_energy, temperature):
    return np.minimum(
        np.exp(-1 * (old_energy-new_energy) / temperature),
        1
    )


def temperature(time_passed, total_time):
    return
