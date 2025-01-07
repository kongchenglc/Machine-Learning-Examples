import numpy as np
import matplotlib.pyplot as plt

# Define the objective function (J) and its gradient (derivative)
def J(x):
    return x**2 - 4*x + 4

def grad_J(x):
    return 2*x - 4
# This function computes the gradient of the objective function 𝐽(𝑥) at a given point 𝑥.
# The gradient tells you the direction and rate at which the function increases, which is used to update 𝑥 in the opposite direction (to minimize the function).

# Gradient Descent Parameters

learning_rate = 0.1 
# This is a scalar value that determines the step size in each iteration of gradient descent
# The gradient tells you the direction and rate at which the function increases, which is used to update 𝑥 in the opposite direction (to minimize the function).

x_init = 0.0  # Starting point
iterations = 50
epsilon = 1e-6  # Convergence tolerance

# Array to store the results for plotting
x_values = np.zeros(iterations)
J_values = np.zeros(iterations)

# Gradient Descent Algorithm
x = x_init
for i in range(iterations):
    # Compute the gradient at the current point
    gradient = grad_J(x)
    
    # Update the current point (x) using the gradient and learning rate
    x = x - learning_rate * gradient
    
    # Store the values for plotting
    x_values[i] = x
    J_values[i] = J(x)
    
    # Stopping condition: if the change in J is small enough, stop the loop
    if i > 0 and abs(J_values[i] - J_values[i-1]) < epsilon:
        print(f"Converged at iteration {i} with x = {x:.4f} and J(x) = {J_values[i]:.4f}")
        break

# Plotting the objective function and the gradient descent steps
x_plot = np.linspace(-1, 5, 400)
J_plot = J(x_plot)

plt.plot(x_plot, J_plot, label="Objective Function J(x)", color='r')
plt.scatter(x_values, J_values, color='blue', label="Gradient Descent Steps")
plt.xlabel('x')
plt.ylabel('J(x)')
plt.title('Gradient Descent Example')
plt.legend()
plt.grid(True)
plt.show()
