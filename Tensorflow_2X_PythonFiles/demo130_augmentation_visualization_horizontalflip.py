# -*- coding: utf-8 -*-
"""Demo130_Augmentation_Visualization_HorizontalFlip.ipynb

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

def show(X_batch, batch_size):
  for k in range(batch_size):
    i = X_batch.next()
    plt.imshow(np.array(i[0,:,:,:], dtype="int64"))
    plt.show()

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip = True)

datagen.fit(X)
batch_size = 10
X_batch = datagen.flow(X, batch_size=batch_size)

print(X_batch)

show(X_batch, batch_size)

