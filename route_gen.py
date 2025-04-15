import numpy as np
from tsp_utils import calculate_distance_matrix

num_points = 25

def save_points(filename="points.npy"):
    points = generate_points(num_points)
    np.save(filename, points)
    print(f"Punkty zapisane w pliku {filename}")
    return points

def generate_points(size=(100, 100)):
    # Zwraca punkty w przestrzeni 2D w wymiarach [0, size[0]] x [0, size[1]]
    return np.random.rand(num_points, 2) * size

points = generate_points(num_points)
np.save("points.npy", points)
print("Punkty zapisane w pliku points.npy")