# -*- coding: utf-8 -*-
"""Demo53_CNN_MovieReviews.ipynb


`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## We can use Convolutions to get features from text

In Images we have H, W and Input, Output nodes for convolution layers 

e.g., 3 x 3 x 5 x 64

We can migrate this idea to text by using T, input and output nodes as the filter dimensions
"""

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout
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

  from sklearn.metrics import confusion_matrix
  import itertools
  cm = confusion_matrix(y_test, y_pred)

  plt.figure(figsize=(10,10))
  plt.imshow(cm, cmap=plt.cm.Blues)
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i,j], 'd'),
            horizontalalignment = 'center',
            color='black')
  plt.xlabel("Predicted labels")
  plt.ylabel("True labels")
  plt.xticks(range(0,classes))
  plt.yticks(range(0,classes))
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.show()

data = pd.read_csv("sample_data/train.tsv", delimiter='\t')
data.head()

Y = len(data.Sentiment.unique())
print(Y)

X = data.iloc[:, 2]
y = data.iloc[:, -1].values

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

MAX_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_SIZE)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

word2index = tokenizer.word_index
V = len(word2index)
print("tokens = " + str(V))

X_train = pad_sequences(sequences_train)

X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

N, T = X_train.shape

# Let's talk about D, what is it and how do we set it? 
# This is the dimensionality of the embedding layer, essentially the vector that each word becomes 

D = 40

i_layer = Input(shape = (T,))
h_layer = Embedding(V+1, D)(i_layer)
# V+1 because https://github.com/tensorflow/tensorflow/issues/38619
h_layer = Conv1D(128, 3, activation='relu')(h_layer)
h_layer = MaxPooling1D(3)(h_layer)
h_layer = Conv1D(254, 3, activation='relu')(h_layer)
h_layer = GlobalMaxPooling1D()(h_layer)
h_layer = Dense(20, activation='relu')(h_layer)
h_layer = Dropout(0.5)(h_layer)
o_layer = Dense(Y, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, epochs = 10, validation_data=(X_test, y_test))

"""## Seems like the model is overfitting, let's see the results"""

y_pred = model.predict(X_test).argmax(axis = 1)

evaluation_tf(report, y_test, y_pred, Y)