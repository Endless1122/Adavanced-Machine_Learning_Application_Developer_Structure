import numpy as np

def generate_linear_data(n=100):
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def generate_xor_data(n=200):
    X = np.random.randn(n, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y