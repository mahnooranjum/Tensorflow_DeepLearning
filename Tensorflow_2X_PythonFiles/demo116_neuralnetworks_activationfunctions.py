# -*- coding: utf-8 -*-
"""Demo116_NeuralNetworks_ActivationFunctions.ipynb


# **Tame Your Python**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
tf.keras.backend.set_floatx('float64')

def plot(X, out):
  plt.plot(np.linspace(-2, 2, 100), X, 'r', label = 'X')
  plt.plot(np.linspace(-2, 2, 100), out,'b', label = 'Activation')
  plt.legend()
  plt.grid()
  plt.show()

X = np.linspace(-4, 4, 100)

print(X.shape)

layer = tf.keras.layers.Activation('relu')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('sigmoid')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('tanh')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('elu')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('exponential')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('hard_sigmoid')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('linear')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('selu')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('swish')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('softsign')
out = layer(X)
plot(X, out)

layer = tf.keras.layers.Activation('softplus')
out = layer(X)
plot(X, out)