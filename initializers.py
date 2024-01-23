import numpy as np


def glorot_uniform(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_uniform(shape):
    fan_in, _ = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)

