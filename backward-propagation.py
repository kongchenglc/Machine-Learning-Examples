import numpy as np

# Activation function: sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define network structure
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
expected_output = np.array([[0], [1], [1], [0]])  # Expected output

# Initialize weights and biases
input_layer_size = 2
hidden_layer_size = 4
output_layer_size = 1

np.random.seed(1)
weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)  # 输入层到隐藏层的权重
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)  # 隐藏层到输出层的权重
bias_hidden = np.random.rand(1, hidden_layer_size)  # 隐藏层的偏置
bias_output = np.random.rand(1, output_layer_size)  # 输出层的偏置

# Learning rate
learning_rate = 0.1

# Train the network
for epoch in range(10000):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate error (loss function)
    error = expected_output - predicted_output
    
    # Backpropagation
    # Output layer error
    output_layer_delta = error * sigmoid_derivative(predicted_output)
    
    # Hidden layer error
    hidden_layer_delta = output_layer_delta.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    weights_input_hidden += input_data.T.dot(hidden_layer_delta) * learning_rate
    bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))  # Mean squared error
        print(f"Epoch {epoch}, Loss: {loss}")

print("Final output:")
print(predicted_output)
