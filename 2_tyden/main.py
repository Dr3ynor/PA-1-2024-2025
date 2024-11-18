import math

distances = []

def print_distances(dist, n):
    for i in range(len(dist)):
        print(f"{dist[i]:.2f}", end="\t")
        if (i + 1) % n == 0:
            print()

def compute_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def read_tsp_file(fname):
    try:
        with open(fname, 'r') as file:
            xs, ys = [], []

            for _ in range(7):
                next(file)

            for line in file:
                if line.startswith('E'):
                    break

                parts = line.split()
                id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                xs.append(x)
                ys.append(y)

            n = len(xs)
            global distances
            distances = [0] * (n * n)

            for i in range(n):
                for j in range(i, n):
                    dist = compute_distance(xs[i], ys[i], xs[j], ys[j])
                    distances[i * n + j] = dist
                    distances[j * n + i] = dist

            print_distances(distances, n)
    except FileNotFoundError:
        print(f"{fname} file not open")

if __name__ == "__main__":
    read_tsp_file("ulysses22.tsp.txt")
