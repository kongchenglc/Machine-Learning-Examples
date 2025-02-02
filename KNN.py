import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class MyKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calculate Euclidean distance
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Get the indices of the k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        # Get the corresponding labels
        k_labels = [self.y_train[i] for i in k_indices]
        # Majority voting
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Generate example data (apples and bananas)
np.random.seed(42)
# Apples: red (0), round (0)
apples = np.random.randn(20, 2) * 0.5 + [0, 0]
# Bananas: yellow (2), long (1)
bananas = np.random.randn(20, 2) * 0.5 + [2, 1]
X = np.vstack([apples, bananas])
y = np.array([0]*20 + [1]*20)  # 0=apples, 1=bananas

# Train KNN
knn = MyKNN(k=3)
knn.fit(X, y)

# Predict new sample (yellow round)
new_sample = np.array([[1.5, 0.5]])
prediction = knn.predict(new_sample)
print(f"Prediction result: {'banana' if prediction[0]==1 else 'apple'}")

# Visualization
plt.scatter(apples[:,0], apples[:,1], c='red', label='apples')
plt.scatter(bananas[:,0], bananas[:,1], c='yellow', label='bananas')
plt.scatter(new_sample[0,0], new_sample[0,1], c='blue', marker='x', s=100, label='new sample')
plt.xlabel('Color (0=red, 2=yellow)')
plt.ylabel('Shape (0=round, 1=long)')
plt.legend()
plt.show()