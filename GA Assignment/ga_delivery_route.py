"""
Genetic Algorithm for Delivery Route Optimisation
Depot = A; delivery locations = B..G. Route starts and ends at A.
"""
import random

# ---------------------------------------------------------------- Dataset
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DEPOT = "A"
STOPS = [loc for loc in LOCATIONS if loc != DEPOT]  # B..G

# Symmetric distance matrix (km); row/column order = A, B, C, D, E, F, G
DIST_MATRIX = [
    [0, 4, 8, 6, 7, 9, 5],
    [4, 0, 5, 7, 3, 8, 6],
    [8, 5, 0, 4, 6, 3, 7],
    [6, 7, 4, 0, 5, 6, 8],
    [7, 3, 6, 5, 0, 4, 6],
    [9, 8, 3, 6, 4, 0, 5],
    [5, 6, 7, 8, 6, 5, 0],
]
INDEX = {loc: i for i, loc in enumerate(LOCATIONS)}

# ------------------------------------------------------- GA parameters
POPULATION_SIZE = 30
GENERATIONS = 100          # must be >= 50
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_COUNT = 2            # best routes copied unchanged
RANDOM_SEED = 42           # fixed seed -> reproducible experiment


# ------------------------------------------------------------ Fitness
def route_distance(chromosome):
    """Total distance of A -> chromosome -> A (includes return to depot)."""
    full_route = [DEPOT] + chromosome + [DEPOT]
    total = 0
    for here, there in zip(full_route, full_route[1:]):
        total += DIST_MATRIX[INDEX[here]][INDEX[there]]
    return total


# --------------------------------------------------- Initial population
def create_individual():
    """A chromosome is a random permutation of B..G (A is implicit)."""
    route = STOPS[:]
    random.shuffle(route)
    return route


def create_population(size):
    return [create_individual() for _ in range(size)]


# ---------------------------------------------------------- Selection
def tournament_selection(population):
    """Pick a few random routes; the shortest one becomes a parent."""
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=route_distance)


# ---------------------------------------------------------- Crossover
def order_crossover(parent1, parent2):
    """Order crossover (OX): copy a slice from parent1, fill the remaining
    positions with the missing cities in the order they appear in parent2.
    The child is always a valid permutation (no repeats, none missing)."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    fill = [city for city in parent2 if city not in child]
    for i in range(size):
        if child[i] is None:
            child[i] = fill.pop(0)
    return child


# ----------------------------------------------------------- Mutation
def swap_mutation(chromosome):
    """Swap two randomly chosen locations -> still a valid route."""
    mutated = chromosome[:]
    i, j = random.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# ------------------------------------------------------------ GA loop
def run_ga():
    random.seed(RANDOM_SEED)
    population = create_population(POPULATION_SIZE)
    best = min(population, key=route_distance)

    for generation in range(1, GENERATIONS + 1):
        population.sort(key=route_distance)

        # Elitism: carry the best routes over unchanged
        next_population = [r[:] for r in population[:ELITE_COUNT]]

        # Fill the rest with children of selected parents
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_RATE else p1[:]
            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)
            next_population.append(child)

        population = next_population
        gen_best = min(population, key=route_distance)
        if route_distance(gen_best) < route_distance(best):
            best = gen_best[:]

        if generation == 1 or generation % 10 == 0:
            print(f"Generation {generation:3d} | best distance so far: {route_distance(best)} km")

    return best


if __name__ == "__main__":
    best_route = run_ga()
    print("\n=== Result ===")
    print("Best route   :", " -> ".join([DEPOT] + best_route + [DEPOT]))
    print("Total distance:", route_distance(best_route), "km")
