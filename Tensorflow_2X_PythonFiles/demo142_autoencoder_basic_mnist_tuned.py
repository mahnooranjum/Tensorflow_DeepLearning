# -*- coding: utf-8 -*-
"""Demo142_Autoencoder_Basic_MNIST_Tuned.ipynb
# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0 , X_test / 255.0 
print(X_train.shape)
print(X_test.shape)

# X_train = np.expand_dims(X_train, -1)
# X_test = np.expand_dims(X_test, -1)
print(X_train.shape)
print(X_test.shape)
# SHAPE 
# N x H x W x Colors 
# Colors = 1 for grayscale 
# Fashion MNIST is grayscale

X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)
print(X_train.shape)
print(X_test.shape)

classes = len(set(y_train))
print(classes)

X_train[0].shape

input_shape = X_train[0].shape

i_layer = Input(shape = input_shape)
h_layer = Dense(512, activation='relu')(i_layer)
h_layer = Dense(254, activation='relu')(h_layer)
h_layer = Dense(128, activation='relu')(h_layer)
h_layer = Dense(254, activation='relu')(h_layer)
h_layer = Dense(512, activation='relu')(h_layer)
o_layer = Dense(X_train[0].shape[0], activation=None)(h_layer)
model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = "mse")

report = model.fit(X_train, X_train, epochs=40, batch_size=200)

idx = np.random.randint(0, len(X_train))
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].imshow(X_train[idx].reshape(28,28), cmap='gray')
X_decoded = model.predict(X_train[[idx]])
ax[1].imshow(X_decoded.reshape(28,28), cmap='gray')

