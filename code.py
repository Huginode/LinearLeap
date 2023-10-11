import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#Random regression Graph using sklearn
x, y = make_regression(n_samples= 100, n_features=1, noise=10)
plt.scatter(x, y)

# Of course reshaping y dimensions
y = y.reshape(y.shape[0], 1)

# Create X matrice
X = np.hstack((x, np.ones(x.shape)))

# Initialize Theta parameter
theta = np.random.randn(2, 1)

#Def the mode
def model(X, theta):
    return X.dot(theta)
