import math
import random
import matplotlib.pyplot as plt

# Define the objective function
# f(x) = x^2 + 4*sin(5x) + 2*cos(3x)
# The goal is to minimize this function
def objective_function(x):
    return x**2 + 4 * math.sin(5 * x) + 2 * math.cos(3 * x)

# Simulated Annealing Algorithm
def simulated_annealing(objective, bounds, initial_temp, cooling_rate, max_iter):
    # Step 1: Initialize the solution
    # Randomly choose a starting point within the bounds
    current_solution = random.uniform(bounds[0], bounds[1])
    current_value = objective(current_solution)
    
    # Step 2: Initialize temperature
    temperature = initial_temp
    
    # Track the best solution found
    best_solution = current_solution
    best_value = current_value

    # Lists to store solutions and values for visualization
    solutions = [current_solution]
    values = [current_value]

    # Step 3: Main optimization loop
    for i in range(max_iter):
        # Generate a new candidate solution in the neighborhood
        new_solution = current_solution + random.uniform(-1, 1) * (temperature / initial_temp)
        
        # Ensure the new solution is within the bounds
        new_solution = max(bounds[0], min(bounds[1], new_solution))
        
        # Calculate the objective value for the new solution
        new_value = objective(new_solution)
        
        # Calculate the change in the objective function
        delta = new_value - current_value
        
        # Step 4: Accept the new solution with a probability
        # If the new solution is better, accept it
        # If worse, accept it with a probability based on the current temperature
        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
            current_solution = new_solution
            current_value = new_value
        
        # Update the best solution found
        if current_value < best_value:
            best_solution = current_solution
            best_value = current_value

        # Store the current solution and value for visualization
        solutions.append(current_solution)
        values.append(current_value)
        
        # Step 5: Cool down (reduce the temperature)
        temperature *= cooling_rate

        # Terminate if temperature is close to zero
        if temperature < 1e-8:
            break

    return best_solution, best_value, solutions, values

# Parameters for Simulated Annealing
bounds = [-2, 2]               # The range of the solution (x ∈ [-2, 2])
initial_temp = 100.0           # Starting temperature
cooling_rate = 0.95            # Rate at which the temperature decreases
max_iter = 1000                # Maximum number of iterations

# Run the Simulated Annealing algorithm
best_solution, best_value, solutions, values = simulated_annealing(
    objective_function, bounds, initial_temp, cooling_rate, max_iter)

# Print the best solution and its value
print(f"Best solution: x = {best_solution:.4f}, Minimum value: f(x) = {best_value:.4f}")

# Visualization of the solution search process
# Plot the objective function curve
x = [i / 100.0 for i in range(int(bounds[0] * 100), int(bounds[1] * 100))]
y = [objective_function(i) for i in x]

plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Objective Function")
plt.scatter(solutions, values, c='red', s=10, label="Solutions Explored")
plt.scatter([best_solution], [best_value], c='green', s=100, label="Best Solution")
plt.title("Simulated Annealing Optimization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
