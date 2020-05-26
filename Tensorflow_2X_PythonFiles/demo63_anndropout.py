# -*- coding: utf-8 -*-
"""Demo63_ANNDropout.ipynb
# **Spit some [tensor] flow**

Practice makes perfect

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Let's load the dataset using keras datasets"""

from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

"""# We'll do PCA and reduce dimensionality"""

# # Applying PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# # Visualising the dataset
# plt.scatter(X_train[:,0], X_train[:,1], color = 'red')
# plt.title('Scatter Plot PCA')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.show()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

N, D = X_train.shape

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (D,)),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(1)
])

#custom_opt = tf.keras.optimizers.Adam(0.01)
model.compile(
    optimizer='adam',
    loss='mse')

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 500, verbose=False)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()

"""## Great, the model is overfitting, let's try dropout"""

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (D,)),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(1)
])

#custom_opt = tf.keras.optimizers.Adam(0.01)
model.compile(
    optimizer='adam',
    loss='mse')
report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 500, verbose=1)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()