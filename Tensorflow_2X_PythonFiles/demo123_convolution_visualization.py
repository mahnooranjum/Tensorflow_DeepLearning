# -*- coding: utf-8 -*-
"""Demo123_Convolution_Visualization.ipynb
# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Reference MachineLearningMastery.com"""

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
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
h_layer = Conv2D(8, (3,3), strides = 1, activation='relu', padding='same')(i_layer)
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

conv_layer = model.layers[1]
print(conv_layer)

filters, biases = conv_layer.get_weights()
print(conv_layer.name, filters.shape)

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

plt.figure(figsize=(20,10))
n_filters, idx = 8, 1
for i in range(n_filters):
	# get filter
	f = filters[:, :, :, i]
	for j in range(3):
		ax = plt.subplot(n_filters, 3, idx)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(f[:, :, j], cmap='gray')
		idx += 1
plt.show()

model_visual = Model(inputs=model.inputs, outputs=conv_layer.output)

model_visual.summary()

maps = model_visual(X)
print(maps.shape)

plt.figure(figsize=(20,10))
square = 4
idx = 1
for _ in range(square):
  for _ in range(square):
    if (idx > square * 2):
      break
    # specify subplot and turn of axis
    ax = plt.subplot(square, square, idx)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(maps[0, :, :, idx-1], cmap='gray')
    idx += 1

plt.show()

maps.shape[3]

for i in range(maps.shape[3]):
  ax = plt.subplot()
  plt.imshow(maps[0, :, :, i], cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()

