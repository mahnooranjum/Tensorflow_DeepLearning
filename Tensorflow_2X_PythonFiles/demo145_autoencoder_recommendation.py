# -*- coding: utf-8 -*-
"""Demo145_Autoencoder_Recommendation.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

!wget http://files.grouplens.org/datasets/movielens/ml-1m.zip

!ls

!unzip ml-1m.zip

!ls

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

!wget http://files.grouplens.org/datasets/movielens/ml-100k.zip

!unzip ml-100k.zip

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

training_set.shape

test_set.shape

num_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
num_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

def transform(data):
    transformed = []
    for user_id in range(1, num_users + 1):
        id_movies = data[:,1][data[:,0] == user_id]
        id_ratings = data[:,2][data[:,0] == user_id]
        ratings = np.zeros(num_movies)
        ratings[id_movies - 1] = id_ratings
        transformed.append(list(ratings))
    return transformed

training_set = transform(training_set)
test_set = transform(test_set)

len(training_set)

len(training_set[0])

num_movies

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model

X_train = np.array(training_set)
X_test = np.array(test_set)

print(X_train.shape)
print(X_test.shape)

num_users

X_train[0].shape

set(X_train[0])

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

report = model.fit(X_train, X_train, epochs=100, batch_size=10)

X_decoded = model.predict(X_test)

X_decoded.shape

X_test.shape

def map(_temp):
  for i in range(len(_temp)):
    interval = (_temp.max()-_temp.min())/5
    if _temp[i]>=_temp.min() and _temp[i]<(_temp.min()+(1* interval)):
      _temp[i] = 1
    elif (_temp[i]>=(_temp.min()+(1* interval)) and (_temp[i]<(_temp.min()+(2* interval)))):
      _temp[i] = 2
    elif (_temp[i]>=(_temp.min()+(2* interval)) and (_temp[i]<(_temp.min()+(3* interval)))):
      _temp[i] = 3
    elif (_temp[i]>=(_temp.min()+(3* interval)) and (_temp[i]<(_temp.min()+(4* interval)))):
      _temp[i] = 4
    elif (_temp[i]>=(_temp.min()+(4* interval)) and (_temp[i]<(_temp.min()+(5* interval)))):
      _temp[i] = 5
    else:
      pass
  return _temp

idx = np.random.randint(0, len(X_test))
for i in range(len(X_test[idx])):
  if X_test[idx][i] != 0:
    print("Rating by User {u} for Movie {a} = {b}".format(u = idx, a=i, b=X_test[idx][i]))
    print("Prediction by User {u} for Movie {a} = {b}".format(u = idx, a=i, b=map(X_decoded[idx])[i]))

