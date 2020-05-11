

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## We want to try some non linear curvy datasets"""

n = 10000
std = 6
X = np.random.random((n,2)) * (2*std) - (std)
y = np.cos(X[:,0]) + np.sin(X[:,1])

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Visualising the dataset
plt.scatter(X_train[:,0], y_train, color = 'red')
plt.title('X1 Plot')
plt.xlabel('X1')
plt.ylabel('y')
plt.show()

# Visualising the dataset
plt.scatter(X_train[:,1], y_train, color = 'red')
plt.title('X2 Plot')
plt.xlabel('X2')
plt.ylabel('y')
plt.show()

# Visualising the dataset with the target variable
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X_train[:,0], X_train[:,1], y_train)
plt.title('Scatter Plot')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

N, D = X_train.shape

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (D,)),
                                    tf.keras.layers.Dense(40, activation='relu'),
                                    tf.keras.layers.Dense(1)
])

#custom_opt = tf.keras.optimizers.Adam(0.01)
model.compile(
    optimizer='adam',
    loss='mse')

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 100)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))

y_pred = model.predict(X_test)[:,0]

print(y_test.shape)
print(y_pred.shape)

plt.scatter(y_test, y_pred, color = 'b')
plt.title('y_test vs y_pred')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# Visualising the dataset
plt.scatter(X_test[:,0], y_test, color = 'red')
plt.scatter(X_test[:,0], y_pred, color = 'b')
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 1')
plt.ylabel('y')
plt.show()

# Visualising the dataset
plt.scatter(X_test[:,1], y_test, color = 'red')
plt.scatter(X_test[:,1], y_pred, color = 'b')
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 2')
plt.ylabel('y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)

y_pred = model.predict(X).flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y_pred)
plt.show()