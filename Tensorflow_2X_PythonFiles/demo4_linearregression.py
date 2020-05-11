
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_regression

# Regression Dataset
n = 10000
X, y = make_regression(n_samples=n, n_features=1, noise=70)
X = X.reshape(n)
datadict = {'data': X, 'target': y}
data = pd.DataFrame(data=datadict)
plt.scatter(X,y)
plt.show()

X = data.iloc[:,[0]].values
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

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Input(shape=(D,)),
                                    tf.keras.layers.Dense(1)
])


model.compile(
    optimizer = tf.keras.optimizers.SGD(0.001, 0.99),
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

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

plt.plot(report.history['loss'], label="loss = SGD")

w = model.layers[0].get_weights()[0][0]
print(w)

print(X.shape)
print(y.shape)
print(w.shape)

print(type(X))
print(type(w))

y_predicted = w.dot(X.transpose())
plt.scatter(X,y)
plt.plot(X,y_predicted, 'r', 5)
plt.show()

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))