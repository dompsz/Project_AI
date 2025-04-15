import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_route(points, route, title="Trasa"):
    # Upewnij się, że points jest macierzą 2D o kształcie (n,2)
    points = np.array(points)
    if points.ndim == 1 or points.shape[1] != 2:
        # Jeśli dane są jednowymiarowe, spróbuj przekształcić je do (n,2)
        try:
            points = points.reshape(-1, 2)
        except Exception as e:
            print("Błąd przekształcania punktów:", e)
            return

    # Tworzymy zamkniętą ścieżkę – dodajemy początek na końcu
    full_route = route + [route[0]]
    route_points = points[full_route]

    plt.figure(figsize=(8, 6))
    plt.plot(route_points[:, 0], route_points[:, 1], 'b-', lw=2, label="Trasa")
    plt.plot(points[:, 0], points[:, 1], 'ro', label="Punkty")
    for i, (x, y) in enumerate(points):
        plt.text(x + 1, y + 1, str(i), fontsize=9)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def animate_route_to_gif(points, route_history, filename="firefly_evolution.gif", interval=300, title="Firefly - Ewolucja"):
    # Upewnij się, że points jest macierzą 2D
    points = np.array(points)
    if points.ndim == 1 or points.shape[1] != 2:
        try:
            points = points.reshape(-1, 2)
        except Exception as e:
            print("Błąd przekształcania punktów:", e)
            return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:, 0], points[:, 1], 'ro')
    for idx, (x, y) in enumerate(points):
        ax.text(x + 1, y + 1, str(idx), fontsize=9)

    line, = ax.plot([], [], 'b-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        # Tworzymy zamkniętą ścieżkę dla bieżącej iteracji
        route = route_history[frame] + [route_history[frame][0]]
        route_points = points[route]
        line.set_data(route_points[:, 0], route_points[:, 1])
        ax.set_title(f"{title} - Iteracja {frame + 1}")
        return line,

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(route_history),
                                  init_func=init,
                                  interval=interval,
                                  blit=False)
    try:
        ani.save(filename, writer='pillow')
        print(f"✅ GIF zapisany jako {filename}")
    except Exception as e:
        print(f"❌ Błąd zapisu GIF-a: {e}")
    plt.close(fig)
