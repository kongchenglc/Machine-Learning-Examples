import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate three-dimensional data
X, y = make_classification(n_samples=100, n_features=3, 
                           n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1)
model = SVC(kernel='rbf', C=1).fit(X, y)

# Create grid
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
z_min, z_max = X[:,2].min()-1, X[:,2].max()+1
xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 10),
                         np.linspace(y_min, y_max, 10),
                         np.linspace(z_min, z_max, 10))

# Predict class of grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Draw 3D plot
fig = go.Figure(data=[
    go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], 
                 mode='markers', marker=dict(color=y, size=5)),
    go.Volume(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
              value=Z.flatten(), isomin=0.5, isomax=1.5, 
              opacity=0.1, surface_count=2, colorscale='Viridis')
])

fig.update_layout(title='3D SVM with RBF Kernel')
fig.show()