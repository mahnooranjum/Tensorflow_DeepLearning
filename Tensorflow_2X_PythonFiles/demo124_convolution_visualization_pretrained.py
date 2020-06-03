# -*- coding: utf-8 -*-
"""Demo124_Convolution_Visualization_Pretrained.ipynb


# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from glob import glob
import sys, os
import cv2

from tensorflow.keras.applications.vgg16 import VGG16 as pretrained, preprocess_input

!wget https://www.theluxecafe.com/wp-content/uploads/2014/07/ferrari-spider-indian-theluxecafe.jpg

!ls

X = cv2.imread('ferrari-spider-indian-theluxecafe.jpg')
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
plt.imshow(X)

print(X.shape)

IMAGE_SIZE = X.shape

IMAGE_SIZE

X = np.expand_dims(X, axis=0)

print(X.shape)

C = 3
pretrained_model = pretrained(input_shape = IMAGE_SIZE,
                              weights = 'imagenet',
                              include_top = False)

model = Model(pretrained_model.input, pretrained_model.output)

model.summary()

model.layers

conv_layers = []
for i in model.layers:
  if 'conv' in i.name:
    conv_layers.append(i)

conv_layers

layer = 0

conv_layer = conv_layers[layer]
print(conv_layer)

filters, biases = conv_layer.get_weights()
print(conv_layer.name, filters.shape)

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

plt.figure(figsize=(6,64))
n_filters, idx = filters.shape[3], 1
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

maps.shape[3]

for i in range(maps.shape[3]):
  ax = plt.subplot()
  plt.imshow(maps[0, :, :, i], cmap='gray')
  ax.set_xticks([])
  ax.set_yticks([])
  plt.show()

