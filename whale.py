import numpy as np
from tsp_utils import evaluate_route

class WhaleOptimizer:
    def __init__(self, dist_matrix, num_agents, num_iterations, b=1):
        self.history = []  # do animacji
        self.dist_matrix = dist_matrix
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.num_points = len(dist_matrix)
        self.b = b  # parametr spirali

    def _distance(self, a, b):
        return np.sum(np.array(a) != np.array(b))

    def _swap_toward(self, current, target, rate):
        # Zbliżanie się do lepszego osobnika – zamiana elementów
        current = current.copy()
        for i in range(len(current)):
            if current[i] != target[i] and np.random.rand() < rate:
                j = current.index(target[i])
                current[i], current[j] = current[j], current[i]
        return current

    def _spiral_move(self, current, best, t):
        # Spiralne przyciąganie – losowe zamiany wokół najlepszego
        current = current.copy()
        d = self._distance(current, best)
        l = np.random.uniform(-1, 1)
        rate = np.exp(self.b * l) * np.cos(2 * np.pi * l)

        return self._swap_toward(current, best, min(abs(rate), 1))

    def optimize(self):
        whales = [list(np.random.permutation(self.num_points)) for _ in range(self.num_agents)]
        fitness = [evaluate_route(w, self.dist_matrix) for w in whales]
        best_idx = np.argmin(fitness)
        best_solution = whales[best_idx]
        best_distance = fitness[best_idx]

        for t in range(self.num_iterations):
            a = 2 - 2 * (t / self.num_iterations)  # zmniejszające się a

            for i in range(self.num_agents):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r

                if np.random.rand() < 0.5:
                    if abs(A) < 1:
                        whales[i] = self._swap_toward(whales[i], best_solution, 1 - abs(A))
                    else:
                        rand_whale = whales[np.random.randint(0, self.num_agents)]
                        whales[i] = self._swap_toward(whales[i], rand_whale, 1 - abs(A))
                else:
                    whales[i] = self._spiral_move(whales[i], best_solution, t)

                # Aktualizuj najlepsze
                current_distance = evaluate_route(whales[i], self.dist_matrix)
                if current_distance < best_distance:
                    best_solution = whales[i]
                    best_distance = current_distance

        return best_solution, best_distance
