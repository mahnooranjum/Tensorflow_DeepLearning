# -*- coding: utf-8 -*-
"""Demo125_Pooling_Visualization.ipynb
# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from glob import glob
import sys, os
import cv2

!wget https://www.theluxecafe.com/wp-content/uploads/2014/07/ferrari-spider-indian-theluxecafe.jpg

!ls

X = cv2.imread('ferrari-spider-indian-theluxecafe.jpg')
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
plt.imshow(X)

print(X.shape)

IMAGE_SIZE = X.shape

X = np.expand_dims(X, axis=0)

print(X.shape)

y = np.ndarray([1])
print(y.shape)

i_layer = Input(shape = IMAGE_SIZE)
h_layer = MaxPool2D((2,2), padding='same')(i_layer)
h_layer = Flatten()(h_layer)
o_layer = Dense(1, activation='sigmoid')(h_layer)

model = Model(i_layer, o_layer)

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

report = model.fit(X, y, epochs = 10)

model.layers

layer = model.layers[1]
print(layer)

model_visual = Model(inputs=model.inputs, outputs=layer.output)

model_visual.summary()

maps = model_visual(X)
print(maps.shape)

size = tuple(maps.shape)

maps.shape[3]

size

size = [size[1], size[2], size[3]]

size = tuple(size)
print(size)

ax = plt.subplot()
plt.imshow(np.array(maps, dtype="int64").reshape(size))
plt.show()

plt.imshow(X[0,:,:,:])

