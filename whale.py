import numpy as np
from tsp_utils import evaluate_route


class WhaleOptimizer:
    def __init__(self, dist_matrix, num_agents, num_iterations, b=1):
        self.dist_matrix = dist_matrix
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.num_points = len(dist_matrix)
        self.b = b

        self.history = []  # zapis najlepszej trasy z każdej iteracji

    def _distance(self, a, b):
        # Odległość permutacji: liczba różnych pozycji
        return np.sum(np.array(a) != np.array(b))

    def _swap_toward(self, current, target, rate):
        # Przemieszczenie current w stronę target poprzez zamiany
        cur = current.copy()
        for i in range(self.num_points):
            if cur[i] != target[i] and np.random.rand() < rate:
                j = cur.index(target[i])
                cur[i], cur[j] = cur[j], cur[i]
        return cur

    def _spiral_move(self, current, best):
        # Spiralne przyciąganie wokół najlepszego rozwiązania
        cur = current.copy()
        d = self._distance(cur, best)
        l = np.random.uniform(-1, 1)
        # współczynnik spiralny
        rate = np.exp(self.b * l) * np.cos(2 * np.pi * l)
        return self._swap_toward(cur, best, min(abs(rate), 1))

    def optimize(self):
        # Inicjalizacja populacji: losowe permutacje
        whales = [list(np.random.permutation(self.num_points)) for _ in range(self.num_agents)]
        # Ocena tras
        fitness = [evaluate_route(w, self.dist_matrix) for w in whales]
        best_idx = np.argmin(fitness)
        best_solution = whales[best_idx]
        best_distance = fitness[best_idx]

        for t in range(self.num_iterations):
            a = 2 - 2 * (t / (self.num_iterations - 1))  # liniowo zmniejszające się a od 2 do 0

            for i in range(self.num_agents):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r

                if np.random.rand() < 0.5:
                    # eksploracja / eksploatacja zależnie od |A|
                    if abs(A) < 1:
                        whales[i] = self._swap_toward(whales[i], best_solution, 1 - abs(A))
                    else:
                        rand_whale = whales[np.random.randint(self.num_agents)]
                        whales[i] = self._swap_toward(whales[i], rand_whale, 1 - abs(A))
                else:
                    # spiralne przyciąganie
                    whales[i] = self._spiral_move(whales[i], best_solution)

                # aktualizacja najlepszego
                current_dist = evaluate_route(whales[i], self.dist_matrix)
                if current_dist < best_distance:
                    best_solution = whales[i]
                    best_distance = current_dist

            # zapis najlepszego rozwiązania z tej iteracji
            self.history.append(best_solution.copy())

        return best_solution, best_distance, self.history
