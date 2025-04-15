import numpy as np

def calculate_distance_matrix(points):
    num_points = len(points)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            dist_matrix[i][j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix

def evaluate_route(route, dist_matrix):
    distance = 0
    n = len(route)
    for i in range(n):
        distance += dist_matrix[route[i], route[(i + 1) % n]]  # (i+1) % n umożliwia zamknięcie pętli
    return distance
