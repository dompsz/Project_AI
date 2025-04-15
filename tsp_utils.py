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
    for i in range(len(route)):
        distance += dist_matrix[route[i - 1], route[i]]
    return distance
