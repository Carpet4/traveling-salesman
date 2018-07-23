def get_temperature(time_passed, total_time, cooling_scalar=2):
    # determines the temperature AKA the relative likelihood of bad swaps
    # to be picked
    # cooling_scalar: higher values lean more towards good swaps

    return (1 - (time_passed / total_time)) ** 2 / cooling_scalar
