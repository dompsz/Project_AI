import numpy as np
from tsp_utils import calculate_distance_matrix
from firefly import FireflyOptimizer
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
    num_iterations = 100
    num_agents = 30

    # Ładujemy zapisane punkty (nie generujemy ich w main)
    points = load_points("points.npy")
    # Obliczamy macierz odległości
    dist_matrix = calculate_distance_matrix(points)

    # Użycie algorytmu świetlika na zapisanych trasach
    fa = FireflyOptimizer(dist_matrix, num_agents, num_iterations,
                          alpha=0.4, beta0=1, gamma=0.1)
    # alpha - eksploracja, stopień losowości
    # beta - siła przyciągania słabszych świetlików do tych jaśniejszych
    # gamma - tłumienie światła, zasięg przyciągania

    best_route, best_distance, history = fa.optimize()

    print(f"Najlepszy dystans: {best_distance}")
    plot_route(points, best_route, title="Najlepsza trasa - Firefly")
    print(f"Liczba iteracji w historii: {len(history)}")
    animate_route_to_gif(points, history, filename="firefly_evolution.gif")


if __name__ == "__main__":
    main()