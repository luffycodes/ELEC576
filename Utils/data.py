import numpy as np
from sklearn import datasets


def generate_moons_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def generate_blob_data():
    np.random.seed(0)
    X, y = datasets.make_blobs(n_samples=200, centers=2, n_features=2, random_state=0)
    return X, y
