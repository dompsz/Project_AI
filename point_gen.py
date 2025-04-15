import numpy as np

def generate_points(num_points, size=(100, 100)):
    return np.random.rand(num_points, 2) * size
