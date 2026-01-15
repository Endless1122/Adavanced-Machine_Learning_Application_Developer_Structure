import numpy as np
from sklearn.datasets import make_blobs

def generate_linear_data(n=100):
    X,y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y = np.where(y == 0, -1, 1)  # Convert labels from {0,1} to {-1,1}
    return X,y

def generate_Overlpping_data():
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=3.0)
    y = np.where(y == 0, -1, 1)  # Convert labels from {0,1} to {-1,1}
    return X, y