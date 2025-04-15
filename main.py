from point_gen import generate_points
from tsp_utils import calculate_distance_matrix
from firefly import FireflyOptimizer
from plot_utils import plot_route, animate_route_to_gif

def main():
    num_points = 30
    num_iterations = 150
    num_agents = 50

    points = generate_points(num_points)
    dist_matrix = calculate_distance_matrix(points)

    fa = FireflyOptimizer(dist_matrix, num_agents, num_iterations,
                          alpha=0.6, beta0=0.8, gamma=0.3)
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
