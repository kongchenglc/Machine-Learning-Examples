import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function J(x, y)
def J(x, y):
    return 2*x**2 - 0.3*x + 3*y**2 - 0.8*y + 7

# Compute the gradient of J(x, y)
def grad_J(x, y):
    # Partial derivatives with respect to x and y
    grad_x = 4*x - 0.3
    grad_y = 6*y - 0.8
    return np.array([grad_x, grad_y])

# Gradient Descent Algorithm
def gradient_descent(learning_rate, iterations, x_init, y_init):
    # Initialize x and y
    x, y = x_init, y_init
    x_values, y_values, J_values = [], [], []

    for i in range(iterations):
        # Calculate gradient
        gradient = grad_J(x, y)
        
        # Update x and y using the gradient and learning rate
        x = x - learning_rate * gradient[0]
        y = y - learning_rate * gradient[1]

        # Store values for plotting
        x_values.append(x)
        y_values.append(y)
        J_values.append(J(x, y))
        
        # Optionally print the progress
        if i % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {i}, x = {x:.4f}, y = {y:.4f}, J(x, y) = {J(x, y):.4f}")
        
    return x_values, y_values, J_values

# Parameters
learning_rate = 0.1
iterations = 100
x_init, y_init = 5, 5  # Initial values for x and y

# Run Gradient Descent
x_values, y_values, J_values = gradient_descent(learning_rate, iterations, x_init, y_init)

# Plotting the 3D surface and gradient descent path
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = J(X, Y)

fig = plt.figure(figsize=(10, 6))

# 3D plot of the objective function surface
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot the gradient descent path in 3D
ax.plot(x_values, y_values, J_values, color='r', marker='o', markersize=5, label="Gradient Descent Path")
ax.set_title('Gradient Descent on J(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('J(x, y)')
ax.legend()

plt.tight_layout()
plt.show()
