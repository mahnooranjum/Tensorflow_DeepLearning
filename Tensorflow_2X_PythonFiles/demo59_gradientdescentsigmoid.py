# -*- coding: utf-8 -*-
"""Demo59_GradientDescentSigmoid.ipynb

# **Delve Deeper**

We need sound conceptual foundation to be good Machine Learning Artists

## Leggo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_hat(x):
    return sigmoid(x) * (1 - sigmoid(x))

from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
n = 500
X, y = make_moons(n_samples=n, noise=0.1)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

datadict = {'X1': X[:,0],'X2' : X[:,1], 'target': y}
data = pd.DataFrame(data=datadict)

dof = 2
# Initialize weights
weights = np.random.normal(scale= 1/dof**.5, size=dof)
print(weights)
epochs = 2000
lr = 0.2

X = data.iloc[:, [0,1]].values
Y = data.iloc[:, 2].values

for e in range(epochs):
    delta_w = np.zeros(weights.shape)
    for x, y in zip(X, Y):
        pred = sigmoid(np.dot(x, weights))
        error = y - pred

        sigma = error * pred * (1 - pred)

        # error x gradient x inputs
        delta_w += sigma * x

    weights += lr * delta_w / n


    if e % (epochs / 20) == 0:
        Y_pred = sigmoid(np.dot(X, weights))
        loss = np.mean((Y_pred - Y) ** 2)
        print("Train loss: ", loss)

Y_pred = sigmoid(np.dot(X, weights))
Y_pred = Y_pred > 0.5

from matplotlib.colors import ListedColormap
X_set, y_set = X, Y_pred
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                  np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Output')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()