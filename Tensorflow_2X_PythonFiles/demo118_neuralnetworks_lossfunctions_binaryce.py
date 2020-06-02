# -*- coding: utf-8 -*-
"""Demo118_NeuralNetworks_LossFunctions_BinaryCE.ipynb


# **Tame Your Python**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

def plot(X, out):
  plt.plot(np.linspace(-2, 2, 100), X, 'r', label = 'X')
  plt.plot(np.linspace(-2, 2, 100), out,'b', label = 'Activation')
  plt.legend()
  plt.grid()
  plt.show()

y_true = [1.0]
y_pred = [0.0]

layer = tf.keras.losses.BinaryCrossentropy()
output = layer(y_true,y_pred)
print(output)

y_true = [1.0]
y_pred = [0.5]

layer = tf.keras.losses.BinaryCrossentropy()
output = layer(y_true,y_pred)
print(output)

y_true = [1.0]
y_pred = [1.0]

layer = tf.keras.losses.BinaryCrossentropy()
output = layer(y_true,y_pred)
print(output)

