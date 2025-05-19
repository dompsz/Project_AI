import random
import numpy as np
from tsp_utils import calculate_distance_matrix
from firefly import FireflyOptimizer
from whale import WhaleOptimizer
from plot_utils import plot_route, animate_route_to_gif

def load_points(filename="points.npy"):
    points = np.load(filename)
    # Jeśli dane są jednowymiarowe, spróbuj przekształcić je do (n,2)
    if points.ndim == 1 or points.shape[1] != 2:
        try:
            points = points.reshape(-1, 2)
        except Exception as e:
            print("Błąd przekształcania punktów:", e)
    print(f"Załadowano punkty z pliku {filename}, shape: {points.shape}")
    return points

def main():
    #
    num_iterations = 60
    num_agents = 1000
    #firefly
    alpha = 0.4
    beta = 1
    gamma = 0.1
    #whale
    b = 0.2
    #map
    points = load_points("points.npy")
    dist_matrix = calculate_distance_matrix(points)

    # alpha - eksploracja, stopień losowości
    # beta - siła przyciągania słabszych świetlików do tych jaśniejszych
    # gamma - tłumienie światła, zasięg przyciągania
    def firefly():
        fa = FireflyOptimizer(dist_matrix, num_agents, num_iterations, alpha, beta, gamma)
        fa_route, fa_distance, fa_history = fa.optimize()
        print(f"[Firefly] Najlepszy dystans: {fa_distance}")
        plot_route(points, fa_route, title="Najlepsza trasa – Firefly")
        print(f"[Firefly] Liczba iteracji w historii: {len(fa_history)}")
        animate_route_to_gif(points, fa_history,
                             filename=f"dane/firefly_{alpha},{beta},{gamma}_{int(fa_distance)}_{num_iterations}_{num_agents}.gif",
                             title="Firefly – Ewolucja trasy")

    def whale():
        for i in range(20):
            #b - kąty spirali
            c = b + random.uniform(-0.05,0.05)
            woa = WhaleOptimizer(dist_matrix, num_agents, num_iterations, c)
            woa_route, woa_distance, woa_history = woa.optimize()
            print(f"[WOA]      Najlepszy dystans: {woa_distance}")
            plot_route(points, woa_route, title="Najlepsza trasa – Whale")
            print(f"[WOA]      Liczba iteracji w historii: {len(woa_history)}")
            animate_route_to_gif(points, woa_history,
                                 filename=f"dane/whale_{c:.2f}_{int(woa_distance)}_{num_iterations}_{num_agents}.gif",
                                 title="Whale – Ewolucja trasy")

    firefly()
    print(f"Firefly Done")
    # whale()
    # print(f"Whale Done")

if __name__ == "__main__":
    main()