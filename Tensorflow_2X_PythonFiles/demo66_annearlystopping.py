# -*- coding: utf-8 -*-
"""Demo66_ANNEarlyStopping.ipynb

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

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

N, D = X_train.shape

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

i_layer = Input(shape = (D,))
h_layer = Dense(128, activation='relu')(i_layer)
h_layer = Dense(256, activation='relu')(h_layer)
h_layer = Dense(256, activation='relu')(h_layer)
o_layer = Dense(1, activation='relu')(h_layer)
model = Model(i_layer, o_layer)

#custom_opt = tf.keras.optimizers.Adam(0.01)
model.compile(
    optimizer='adam',
    loss='mse')

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs =300, verbose=False)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()

"""## Great, the model is overfitting, let's try early stopping"""

i_layer = Input(shape = (D,))
h_layer = Dense(128, activation='relu')(i_layer)
h_layer = Dense(256, activation='relu')(h_layer)
h_layer = Dense(256, activation='relu')(h_layer)
o_layer = Dense(1, activation='relu')(h_layer)
model = Model(i_layer, o_layer)

#custom_opt = tf.keras.optimizers.Adam(0.01)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
model.compile(
    optimizer='adam',
    loss='mse')
report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 300, verbose=False, callbacks = [callback])

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()