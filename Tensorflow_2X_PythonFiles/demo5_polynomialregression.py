
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_regression

# the equation 2 * x^3 - 40 * x^2 + 9 * x + 24
n = 10000
X = np.random.randn(n)
randomize = np.random.randint(-100,100, size = n)
y = []
for i in range(n):
    y.append((2*X[i]**3)-(40*X[i]**2)+(9*X[i])+24)

for i in range(n):
    y[i] = y[i] + randomize[i]
    
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

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200)

plt.plot(report.history['loss'], label="loss = SGD")

print(X.shape)
print(y.shape)

y_predicted = model.predict(X)
plt.scatter(X,y, color='b')
plt.scatter(X,y_predicted, color='r')
plt.show()

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))