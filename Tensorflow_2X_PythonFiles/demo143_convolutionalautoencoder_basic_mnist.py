# -*- coding: utf-8 -*-
"""Demo143_ConvolutionalAutoencoder_Basic_MNIST.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten, MaxPool2D, UpSampling2D, Reshape
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0 , X_test / 255.0 
print(X_train.shape)
print(X_test.shape)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print(X_train.shape)
print(X_test.shape)
# SHAPE 
# N x H x W x Colors 
# Colors = 1 for grayscale 
# Fashion MNIST is grayscale

classes = len(set(y_train))
print(classes)

X_train[0].shape

input_shape = X_train[0].shape

i_layer = Input(shape = input_shape)
h_layer = Conv2D(16, (3,3), activation='relu', padding='same')(i_layer)
h_layer = MaxPool2D((2,2), padding = 'same')(h_layer)
h_layer = Conv2D(8, (3,3), activation='relu', padding='same')(h_layer)
h_layer = MaxPool2D((2,2), padding = 'same')(h_layer)
h_layer = UpSampling2D((2,2))(h_layer)
h_layer = Conv2D(8, (3,3), activation='relu', padding='same')(h_layer)
h_layer = UpSampling2D((2,2))(h_layer)
h_layer = Conv2D(16, (3,3), activation='relu', padding='same')(h_layer)
o_layer = Conv2D(1, (3,3), activation=None, padding='same')(h_layer)
model = Model(i_layer, o_layer)

model.summary()

model.compile(optimizer='adam', 
              loss = "mse")

report = model.fit(X_train, X_train, epochs=10, batch_size=200)

idx = np.random.randint(0, len(X_train))
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].imshow(X_train[idx].reshape(28,28), cmap='gray')
X_decoded = model.predict(X_train[[idx]])
ax[1].imshow(X_decoded.reshape(28,28), cmap='gray')

