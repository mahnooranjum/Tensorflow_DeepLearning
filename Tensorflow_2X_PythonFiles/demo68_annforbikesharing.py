# -*- coding: utf-8 -*-
"""Demo68_ANNforBikeSharing.ipynb

# **Spit some [tensor] flow**

Practice makes perfect

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

!wget --passive-ftp --prefer-family=ipv4 https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

!ls

!unzip Bike-Sharing-Dataset.zip

!rm Bike-Sharing-Dataset.zip

!ls

data = pd.read_csv('hour.csv')
data.head()

categoricals = ['weathersit', 'season', 'mnth', 'hr', 'weekday']
for col in categoricals:
  dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
  data = pd.concat([data, dummies], axis = 1)

data = data.drop(categoricals, axis=1)
data.head()

drop_cols = "instant,dteday,workingday,atemp".split(",")
data = data.drop(drop_cols, axis=1)
data.head()

from sklearn.preprocessing import StandardScaler
numericals = ['temp', 'hum', 'windspeed', 'registered', 'cnt','casual']
scaler = StandardScaler()
data[numericals] = scaler.fit_transform(data[numericals])

targets = ['cnt', 'casual', 'registered']
y = data[targets]

X = data.drop(targets, axis=1)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

N, D = X_train.shape

Y = y_train.shape[1]

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

i_layer = Input(shape=(D,))
h_layer = Dense(10, activation='relu')(i_layer)
h_layer = Dense(10, activation='relu')(h_layer)
o_layer = Dense(Y, activation='relu')(h_layer)
model = Model(i_layer, o_layer)

model.compile(
    optimizer='adam',
    loss='mse')

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 100)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()