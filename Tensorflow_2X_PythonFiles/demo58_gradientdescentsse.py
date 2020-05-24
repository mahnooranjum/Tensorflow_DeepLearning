# -*- coding: utf-8 -*-
"""Demo58_GradientDescentSSE.ipynb


# **Delve Deeper**

We need sound conceptual foundation to be good Machine Learning Artists

## Leggo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_regression

# Regression Dataset
n = 1000
X, y = make_regression(n_samples=n, n_features=1, noise=30)
X = X.reshape(n)
datadict = {'feature': X, 'target': y}
data = pd.DataFrame(data=datadict)
plt.scatter(X,y, color= 'blue')
plt.grid()
plt.show()

dof = 2
# Initialize weights
weights = np.random.normal(scale= 1/dof**.5, size=dof)
print(weights)
epochs = 2000
lr = 0.2

for e in range(epochs):
    delta_w = np.array([0], dtype='float64')
    delta_b = np.array([0], dtype='float64')
    for x, y in zip(X, data.target):
        y_pred_current = weights[0] * x + weights[1]
        error = (y - y_pred_current)**2

        delta_w += (-2) * x * (y - y_pred_current)
        delta_b += (-2) * (y - y_pred_current)

    weights[0] -= lr * delta_w / (n*2)
    weights[1] -= lr * delta_b / (n*2)

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        pred= X.T * weights[0] + weights[1]
        loss = np.mean((pred -  data.target) ** 2)
        print("Train loss: ", loss)

y_pred = X.T * weights[0] + weights[1]

plt.scatter(X,data.target, color= 'blue')
plt.scatter(X,y_pred, color= 'red')
plt.grid()
plt.show()