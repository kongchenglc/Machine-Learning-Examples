import numpy as np
import matplotlib.pyplot as plt

class MyKMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign samples to the nearest centroid
            clusters = self._create_clusters(X)
            # Save old centroids for comparison
            old_centroids = self.centroids.copy()
            # Update centroids
            self.centroids = self._update_centroids(X, clusters)
            # If centroids do not change, stop early
            if np.allclose(old_centroids, self.centroids):
                break
    
    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for x in X:
            distances = [np.sqrt(np.sum((x - c)**2)) for c in self.centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x)
        return clusters
    
    def _update_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                centroids[i] = self.centroids[i]  # If cluster is empty, keep original centroid
            else:
                centroids[i] = np.mean(cluster, axis=0)
        return centroids

# Generate example data (two clusters)
np.random.seed(42)
cluster1 = np.random.randn(50, 2) * 0.5 + [0, 0]  # Center at (0,0)
cluster2 = np.random.randn(50, 2) * 0.5 + [3, 3]  # Center at (3,3)
X = np.vstack([cluster1, cluster2])

# Train K-Means
kmeans = MyKMeans(k=2)
kmeans.fit(X)

# Visualization
plt.scatter(X[:,0], X[:,1], c='gray', alpha=0.5, label='Original data')
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], 
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()