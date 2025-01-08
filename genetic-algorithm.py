import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function: Rastrigin function
def objective_function(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# Initialize population randomly
def initialize_population(size, x_range, y_range):
    population = np.random.uniform(low=[x_range[0], y_range[0]], 
                                    high=[x_range[1], y_range[1]], 
                                    size=(size, 2))
    return population

# Evaluate fitness of population
def evaluate_population(population):
    return np.array([objective_function(ind[0], ind[1]) for ind in population])

# Select individuals based on fitness (roulette wheel selection)
def select_population(population, fitness):
    probabilities = 1 / (fitness + 1e-6)  # Inverse fitness for minimization
    probabilities /= np.sum(probabilities)
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]

# Perform crossover between two parents
def crossover(parent1, parent2, crossover_rate=0.7):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.hstack((parent1[:point], parent2[point:]))
        child2 = np.hstack((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Perform mutation
def mutate(individual, mutation_rate=0.1, x_range=(-5.12, 5.12), y_range=(-5.12, 5.12)):
    if np.random.rand() < mutation_rate:
        idx = np.random.randint(len(individual))
        if idx == 0:
            individual[idx] = np.random.uniform(*x_range)
        else:
            individual[idx] = np.random.uniform(*y_range)
    return individual

# Genetic Algorithm visualization
def genetic_algorithm_demo():
    # Problem setup
    population_size = 30
    generations = 50
    x_range = (-5.12, 5.12)
    y_range = (-5.12, 5.12)
    crossover_rate = 0.8
    mutation_rate = 0.2

    # Initialize population
    population = initialize_population(population_size, x_range, y_range)
    fitness_history = []

    # Set up the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = objective_function(X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.2, cmap='viridis')
    scatter = ax.scatter([], [], [], color='red', s=50)

    # Evolution process
    for generation in range(generations):
        fitness = evaluate_population(population)
        fitness_history.append(np.min(fitness))
        population = select_population(population, fitness)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[(i + 1) % len(population)]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.extend([mutate(child1, mutation_rate, x_range, y_range), 
                                   mutate(child2, mutation_rate, x_range, y_range)])
        population = np.array(new_population)

        # Update visualization
        ax.cla()
        ax.plot_surface(X, Y, Z, alpha=0.2, cmap='viridis')
        scatter = ax.scatter(population[:, 0], population[:, 1], evaluate_population(population), color='red', s=50)
        plt.title(f"Generation {generation + 1}")
        plt.pause(0.1)

    plt.show()
    print(f"Best fitness over generations: {min(fitness_history)}")

# Run the demo
genetic_algorithm_demo()
