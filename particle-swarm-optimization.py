import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Rastrigin function (to minimize)
def rastrigin_function(x, y, A=10):
    return A*2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# PSO Parameters
n_particles = 30  # Number of particles
n_iterations = 50  # Number of iterations
bounds = [-5.12, 5.12]  # Rastrigin function bounds
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Initialize particle positions and velocities
particles = np.array([[random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])] for _ in range(n_particles)])
velocities = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(n_particles)])
personal_best_positions = particles.copy()
personal_best_values = np.array([rastrigin_function(p[0], p[1]) for p in particles])
global_best_position = personal_best_positions[np.argmin(personal_best_values)]

# Set up the 3D plot
x = np.linspace(bounds[0], bounds[1], 500)
y = np.linspace(bounds[0], bounds[1], 500)
X, Y = np.meshgrid(x, y)
Z = rastrigin_function(X, Y)

# Create the figure and axis for 3D plot
fig = plt.figure(figsize=(10, 8))  # Increase figure size
ax = fig.add_subplot(111, projection='3d')

# Plot the Rastrigin surface with increased transparency
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)  # Reduced alpha for transparency

# Scatter plot for particles
sc = ax.scatter(particles[:, 0], particles[:, 1], rastrigin_function(particles[:, 0], particles[:, 1]), color='red', label="Particles")

# Set axis limits
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[0], bounds[1])
ax.set_zlim(0, np.max(Z) * 1.2)

# Set title and labels
ax.set_title("3D Particle Swarm Optimization Visualization")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Objective function value")

ax.legend()

# Update function for animation
def update(frame):
    global particles, velocities, personal_best_positions, personal_best_values, global_best_position

    for i in range(n_particles):
        # Update velocity
        r1 = random.random()
        r2 = random.random()
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (personal_best_positions[i] - particles[i])
            + c2 * r2 * (global_best_position - particles[i])
        )
        
        # Update position
        particles[i] += velocities[i]
        particles[i] = np.clip(particles[i], bounds[0], bounds[1])  # Stay within bounds
        
        # Update personal best
        value = rastrigin_function(particles[i][0], particles[i][1])
        if value < personal_best_values[i]:
            personal_best_positions[i] = particles[i]
            personal_best_values[i] = value
            
    # Update global best
    global_best_position = personal_best_positions[np.argmin(personal_best_values)]
    
    # Update plot (update particle positions)
    sc._offsets3d = (particles[:, 0], particles[:, 1], rastrigin_function(particles[:, 0], particles[:, 1]))
    return sc,

# Create animation
ani = FuncAnimation(fig, update, frames=n_iterations, interval=100, blit=False)

plt.show()

# Final result
print(f"Best solution: x = {global_best_position[0]:.4f}, y = {global_best_position[1]:.4f}, f(x, y) = {rastrigin_function(global_best_position[0], global_best_position[1]):.4f}")
