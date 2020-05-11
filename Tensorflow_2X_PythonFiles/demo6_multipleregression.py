
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_regression
#MULTIPLE LINEAR REGRESSION
n = 10000
X = np.random.randn(n)
Z = np.random.randn(n)
randomize = np.random.randint(-50,50, size = n)
y = []
for i in range(n):
    y.append((2*X[i])-(40*Z[i])+24)

for i in range(n):
    y[i] = y[i] + randomize[i]
    
datadict = {'X1': X, 'X2': Z, 'target': y}
data = pd.DataFrame(data=datadict)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Z, y, zdir='z', color="green", s=20, c=None, depthshade=True)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")

X = data.iloc[:,[0, 1]].values
type(X)

y = data.target.values

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

N, D = X_train.shape

from sklearn.preprocessing import StandardScaler 
scaleObj = StandardScaler()
X_train = scaleObj.fit_transform(X_train)
X_test = scaleObj.transform(X_test)

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

input_layer = Input(shape=(D,))
dense_layer_1 = Dense(20, activation='relu')(input_layer)
dense_layer_2 = Dense(20, activation='relu')(input_layer)
output = Dense(1)(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mse']
)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

plt.plot(report.history['loss'], label="loss = SGD")

print(X.shape)
print(y.shape)

y_predicted = model.predict(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, zdir='z', color="blue", s=1, c=None, depthshade=True)
ax.scatter(X[:,0], X[:,1], y_predicted, zdir='z', color="red", s=40, c=None, depthshade=True)

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")

y_predicted = model.predict(X)
plt.scatter(X[:,0],y, color='b')
plt.scatter(X[:,0],y_predicted, color='r')
plt.show()

plt.scatter(X[:,1],y, color='b')
plt.scatter(X[:,1],y_predicted, color='r')
plt.show()

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))