# -*- coding: utf-8 -*-
"""Demo76_ANN_MovieReviews_DatasetTuning.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, SimpleRNN, LSTM, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""## Let's import the dataset"""

def evaluation_tf(report, y_test, y_pred, classes):
  plt.plot(report.history['loss'], label = 'training_loss')
  plt.plot(report.history['val_loss'], label = 'validation_loss')
  plt.legend()
  plt.show()
  plt.plot(report.history['accuracy'], label = 'training_accuracy')
  plt.plot(report.history['val_accuracy'], label = 'validation_accuracy')
  plt.legend()
  plt.show()

from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                     num_words=None,
                                                     skip_top=0,
                                                     maxlen=None)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train[np.random.randint(0,len(X_train))])

print(set(y_train))

V = 5000
tokenizer = Tokenizer(num_words=V)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print(X_train[np.random.randint(0,len(X_train))])

print(X_train.shape)
print(X_test.shape)

classes = len(set(y_train))
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)
print(y_train.shape)
print(y_test.shape)

print("tokens = " + str(V))

i_layer = Input(shape = (V,))
h_layer = Dense(254, activation='relu')(i_layer)
h_layer = Dropout(0.9)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, epochs = 30, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

evaluation_tf(report, y_test, y_pred, classes)