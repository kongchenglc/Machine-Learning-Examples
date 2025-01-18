import numpy as np
import matplotlib.pyplot as plt

# For Plotting the Sigmoid with w0 and w1 values
x_sig = np.arange(550, 750, 1)
x_sig = x_sig / 711  # Normalize input for plotting

# Light Bulb Example Data Set
x_data = np.arange(600, 711, 10)
x_data = x_data / 711  # Normalize input for x_data
target = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

# Gradient descent method Parameters
w0 = -100      # Initial Guess
w1 = 100       # Initial Guess
lr = 2         # Learning Rate

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(2000):
    # Calculate predictions using current weights
    predictions = sigmoid(w0 + w1 * x_data)
    
    # Log-Likelihood Calculation
    LogLikelyhood = np.sum(target * np.log(predictions) + (1 - target) * np.log(1 - predictions))
    
    # Calculate gradients
    G0 = np.sum((target - predictions) * 1)  # Gradient with respect to w0
    G1 = np.sum((target - predictions) * x_data)  # Gradient with respect to w1
    
    # Update weights
    w0 = w0 + lr * G0
    w1 = w1 + lr * G1

    # Plotting the sigmoid curve after each iteration
    estimate = sigmoid(w0 + w1 * x_sig)
    plt.plot(x_sig, estimate, color='grey')

# Plotting final result
print("Log Likelihood:", LogLikelyhood, "w0:", w0, "w1:", w1)
plt.plot(x_sig, estimate, color='red')  # Final sigmoid curve
plt.scatter(x_data, target)  # Scatter plot of data points
plt.grid()
plt.show()
