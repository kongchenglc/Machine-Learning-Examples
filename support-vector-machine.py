import numpy as np
import matplotlib.pyplot as plt

# Generate linearly separable sample data
np.random.seed(42)
X = np.array([[1, 2], [2, 3], [2, 1], [3, 2], 
              [6, 5], [7, 7], [6, 7], [7, 5]])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])  # 使用±1标签更方便数学推导

# Data normalization (SVM is sensitive to feature scales)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

class LinearSVM:
    def __init__(self, C=1.0, lr=0.01, max_iter=1000):
        """
        Initialize SVM parameters
        C: Regularization parameter (trade-off between margin width and classification error)
        lr: Learning rate
        max_iter: Maximum number of iterations
        """
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
    
    def fit(self, X, y):
        """
        Core method for training the SVM model
        Mathematical objective: minimize (1/2)||w||² + C∑max(0, 1 - y_i(w·x_i + b))
        Optimize using subgradient descent
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters: weights w and bias b
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Gradient descent optimization loop
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                y_i = y[idx]
                
                # Calculate the predicted value for the current sample
                decision = y_i * (np.dot(x_i, self.w) + self.b)
                
                # Hinge Loss subgradient calculation
                if decision >= 1:
                    # Correctly classified and outside the margin, no contribution to gradient
                    dw = 0
                    db = 0
                else:
                    # Inside the margin or misclassified, contributes linear gradient
                    dw = self.w - self.C * y_i * x_i
                    db = -self.C * y_i
                
                # Update weights and bias
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
    def decision_function(self, X):
        """Calculate decision function value w·x + b"""
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """Predict class (±1)"""
        return np.sign(self.decision_function(X))

# Train the model
svm = LinearSVM(C=1000, lr=0.001, max_iter=5000)
svm.fit(X, y)

# Visualize results
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=70, edgecolors='k')

# Draw decision boundary w·x + b = 0
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Generate grid points
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX, YY = np.meshgrid(xx, yy)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Draw decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
           linestyles=['--', '-', '--'], linewidths=2)

# Mark support vectors (manual calculation)
support_vectors = []
for i in range(len(X)):
    if np.abs(svm.decision_function(X[i])) <= 1.0 + 1e-5:
        support_vectors.append(X[i])
support_vectors = np.array(support_vectors)

# Ensure support_vectors is a 2D array
if support_vectors.ndim == 1:
    support_vectors = support_vectors[np.newaxis, :]

# Add check to ensure support_vectors is not empty
if support_vectors.size > 0:
    plt.scatter(support_vectors[:,0], support_vectors[:,1], 
                s=150, facecolors='none', edgecolors='red', linewidths=2)
else:
    print("No support vectors found.")

# Add mathematical formula annotation
plt.title("Manual Linear SVM Implementation\n"
          r"Minimize $\frac{1}{2}||\mathbf{w}||^2 + C\sum \max(0, 1-y_i(\mathbf{w}\cdot\mathbf{x_i}+b))$",
          pad=20)
plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.show()

# Output model parameters
print(f"Learned weights: {svm.w}")
print(f"Bias term: {svm.b}")
print(f"Margin width: {2/np.linalg.norm(svm.w):.4f}")