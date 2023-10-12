import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#Random regression Graph using sklearn
x, y = make_regression(n_samples= 100, n_features=1, noise=10)

# Of course reshaping y dimensions
y = y.reshape(y.shape[0], 1)

# Create X matrice
X = np.hstack((x, np.ones(x.shape)))

# Initialize Theta parameter
theta = np.random.randn(2, 1)

#Def the mode
def model(X, theta):
    return X.dot(theta)

#Plot the dataset
plt.scatter(x, y)
plt.plot(x, model(X, theta), c='r')

# Cost funciton
def costFunction(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

# Gradiant Descent
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


def gradientDescent(X, y ,theta, learningRate, nIterations):

    for i in range(nIterations):
        theta = theta - learningRate * grad(X, y , theta)

    return theta

thetaFinal = gradientDescent(X, y, theta, learningRate=0.01, nIterations=1000)

# Verifying the parameters
prediction = model(X, thetaFinal)
plt.plot(x, prediction, c='g')
plt.show()
