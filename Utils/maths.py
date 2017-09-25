import numpy as np


def sigmoid(z):
    return 1 + np.exp(-z)


def diff_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)


def diff_tanh(z):
    return 1 - np.power(tanh(z), 2)


def relu(z):
    return max(0, z)


def diff_relu(z):
    if z > 0:
        return 1
    else:
        return 0
