# -*- coding: utf-8 -*-
"""Demo57_ANN_Recommendations.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Recommender systems use the concepts of Embedding layers of NLP"""

from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn

"""## Let's import the dataset"""

# Get dataset https://www.kaggle.com/grouplens/movielens-20m-dataset?select=rating.csv

data = pd.read_csv('rating.csv')
data.head()

data.userId = pd.Categorical(data.userId)
data["processed_userId"] = data.userId.cat.codes
data.head()

data.movieId = pd.Categorical(data.movieId)
data["processed_movieId"] = data.movieId.cat.codes
data.head()

userId = data['processed_userId']
movieId = data['processed_movieId']
ratings = data['rating']

N = len(set(userId))
D = len(set(movieId))
E = 15

u_layer = Input(shape=(1,))
m_layer = Input(shape=(1,))
u_embedding = Embedding(N, E)(u_layer)
m_embedding = Embedding(D, E)(m_layer)

# The output for embeddings is samples x T x E === samples x 1 x E migrated from NLP

u_embedding = Flatten()(u_embedding)
m_embedding = Flatten()(m_embedding)
# The output for flatten is samples x E === samples x E 

i_layer = Concatenate()([u_embedding, m_embedding])
h_layer = Dense(512, activation = 'relu')(i_layer)
h_layer = Dense(512, activation = 'relu')(h_layer)
o_layer = Dense(1)(h_layer)

model = Model([u_layer, m_layer], o_layer)

model.compile(loss = 'mse',
             optimizer = SGD(lr=0.1, momentum = 0.8))

# TRAIN TEST SPLIT
userId, movieId, ratings = sklearn.utils.shuffle(userId, movieId, ratings)
split = int(0.2 * len(ratings))
train_userId = userId[split:]
train_movieId = movieId[split:]
train_ratings = ratings[split:]

test_userId = userId[:split]
test_movieId = movieId[:split]
test_ratings = ratings[:split]

print(train_userId.shape)
print(train_movieId.shape)
print(train_ratings.shape)
print(test_userId.shape)
print(test_movieId.shape)
print(test_ratings.shape)

normalizer_ratings = train_ratings.mean()
train_ratings = train_ratings - normalizer_ratings
test_ratings = test_ratings - normalizer_ratings

report = model.fit(x = [train_userId, train_movieId],
                   y = train_ratings,
                   epochs = 20, 
                   batch_size = 2048, 
                   validation_data = ([test_userId, test_movieId], test_ratings))

"""## let's see the results"""

plt.plot(report.history['loss'], label="train_loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()
plt.show()

