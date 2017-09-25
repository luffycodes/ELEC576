import numpy as np

# CARE: z can be a matrix
# CARE: differentiating, given the value of f(z), not z


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def diff_sigmoid(fz):
    return fz * (1 - fz)


def tanh(z):
    return np.tanh(z)


def diff_tanh(fz):
    return 1 - np.power(fz, 2)


def relu(z):
    return np.maximum(z, 0)


def diff_relu(z):
    return 1 * (z > 0)
