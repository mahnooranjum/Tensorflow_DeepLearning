# -*- coding: utf-8 -*-
"""Demo74_ANN_AND.ipynb



# **Spit some [tensor] flow**

Practice makes perfect

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

data = np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,1]])
print(data)
print(data.shape)

X = data[:,0:2]
y = data[:,2]
print(X.shape)
print(y.shape)

N, D = X.shape

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

i_layer = Input(shape = (D,))
h_layer = Dense(8, activation='relu')(i_layer)
h_layer = Dense(4, activation='relu')(h_layer)
o_layer = Dense(1, activation='sigmoid')(h_layer)
model = Model(i_layer, o_layer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='binary_crossentropy',
    metrics = ['accuracy'])

report = model.fit(X, y, epochs = 100)

plt.plot(report.history['loss'], label="loss")
plt.legend()

from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                  np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape) > 0.5,
          alpha = 0.75, cmap = ListedColormap(('pink', 'cyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title("AND")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()