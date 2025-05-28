import numpy as np
from tsp_utils import evaluate_route

class FireflyOptimizer:
    def __init__(self, dist_matrix, num_agents, num_iterations, alpha=0.3, beta0=1.0, gamma=0.05):
        self.dist_matrix = dist_matrix
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.num_points = len(dist_matrix)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.history = []

    def _distance(self, a, b):
        return np.count_nonzero(np.array(a) != np.array(b))

    def _move_firefly(self, a, b, beta):
        a = a.copy()
        for i in range(self.num_points):
            if a[i] != b[i] and np.random.rand() < beta:
                j = a.index(b[i])
                a[i], a[j] = a[j], a[i]
        if np.random.rand() < self.alpha:
            i, j = np.random.randint(0, self.num_points, size=2)
            a[i], a[j] = a[j], a[i]
        return a

    def optimize(self):
        fireflies = [list(np.random.permutation(self.num_points)) for _ in range(self.num_agents)]
        brightness = np.array([1 / evaluate_route(f, self.dist_matrix) for f in fireflies])

        for t in range(self.num_iterations):
            # Sortuj tylko raz (najlepsze osobniki będą wyżej)
            sorted_indices = np.argsort(-brightness)  # descending
            new_fireflies = fireflies.copy()
            new_brightness = brightness.copy()

            for idx_i in range(self.num_agents):
                i = sorted_indices[idx_i]
                for idx_j in range(idx_i + 1, self.num_agents):
                    j = sorted_indices[idx_j]

                    rij = self._distance(fireflies[i], fireflies[j])
                    beta = self.beta0 * np.exp(-self.gamma * rij ** 2)
                    candidate = self._move_firefly(fireflies[i], fireflies[j], beta)
                    candidate_brightness = 1 / evaluate_route(candidate, self.dist_matrix)

                    if candidate_brightness > brightness[i]:
                        new_fireflies[i] = candidate
                        new_brightness[i] = candidate_brightness

            fireflies = new_fireflies
            brightness = new_brightness

            best_idx = np.argmax(brightness)
            self.history.append(fireflies[best_idx].copy())

        best_idx = np.argmax(brightness)
        return fireflies[best_idx], 1 / brightness[best_idx], self.history
