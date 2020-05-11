

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Let's load the dataset using keras datasets"""

dataset = tf.keras.datasets.boston_housing

(X_train, y_train), (X_test, y_test) = dataset.load_data()

"""# We'll do PCA and reduce dimensionality"""

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Visualising the dataset
plt.scatter(X_train[:,0], X_train[:,1], color = 'red')
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Visualising the dataset with the target variable
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X_train[:,0], X_train[:,1], y_train)
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

N, D = X_train.shape

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (D,)),
                                    tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dense(40, activation='relu'),
                                    tf.keras.layers.Dense(20, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='relu'),
                                    tf.keras.layers.Dense(1)
])

#custom_opt = tf.keras.optimizers.Adam(0.01)
model.compile(
    optimizer='adam',
    loss='mse')

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 500)

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



# Visualising the dataset with the target variable
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X_test[:,0], X_test[:,1], y_test)
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X_test[:,0], X_test[:,1], y_test)
plt.title('Scatter Plot PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()