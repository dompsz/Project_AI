import numpy as np
from tsp_utils import evaluate_route

class FireflyOptimizer:
    def __init__(self, dist_matrix, num_agents, num_iterations, alpha=0.5, beta0=1, gamma=1):
        self.history = []  # do animacji
        self.dist_matrix = dist_matrix
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.num_points = len(dist_matrix)

        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def _distance(self, a, b):
        return np.sum(np.array(a) != np.array(b))

    def _move_firefly(self, a, b, beta):
        a = a.copy()
        for i in range(len(a)):
            if a[i] != b[i] and np.random.rand() < beta:
                j = a.index(b[i])
                a[i], a[j] = a[j], a[i]
        if np.random.rand() < self.alpha:
            i, j = np.random.randint(0, len(a), 2)
            a[i], a[j] = a[j], a[i]
        return a

    def optimize(self):
        fireflies = [list(np.random.permutation(self.num_points)) for _ in range(self.num_agents)]
        brightness = [1 / evaluate_route(f, self.dist_matrix) for f in fireflies]

        for t in range(self.num_iterations):
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if brightness[j] > brightness[i]:
                        rij = self._distance(fireflies[i], fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * (rij ** 2))
                        fireflies[i] = self._move_firefly(fireflies[i], fireflies[j], beta)
                        brightness[i] = 1 / evaluate_route(fireflies[i], self.dist_matrix)
            best_idx = np.argmax(brightness)
            self.history.append(fireflies[best_idx].copy())

        best_idx = np.argmax(brightness)
        best_route = fireflies[best_idx]
        best_distance = evaluate_route(best_route, self.dist_matrix)

        return best_route, best_distance, self.history
