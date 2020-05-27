# -*- coding: utf-8 -*-
"""Demo70_GAN_MNIST.ipynb

# **Spit some [tensor] flow**

Practice makes perfect

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Generative Adversarial Networks 

GANs are one of the most interesting applications of deep learning. They work in pairs; they have two sub-networks working against each other. A generative network generates fake data to fool the discriminating network. E.g., a generative network makes faces, and the discriminating network classifies if they are real people or not. 

They are called adversarial because their subunits work against each other. Let's get into the concepts of GANs. 

What loss function would the discriminator use? It has to check if the generated data is real or fake, so we'll use binary crossentropy. 

What about the generator loss? We'll just freeze the discriminator! and use the binary crossentropy loss function with labels reversed. So the generative network fools the discriminator and we ourselves fool the generative network !


## Image ====> Discriminator ====> Real or Fake? 

## Noise ====> Generator ====> Fake Image

## Let's load the dataset using keras datasets
"""

from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import os
import sys

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0 * 2 - 1 , X_test / 255.0 * 2 -1  
# SCALING BETWEEN -1 AND 1

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

N, H, W = X_train.shape
D = H * W

X_train = X_train.reshape(-1, D)
X_test = X_test.reshape(-1, D)

print(X_train.shape)
print(X_test.shape)

# Dimensionality of the latent space
LD = 200
# Reference https://www.tensorflow.org/tutorials/generative/dcgan

def make_generator(LD, D):
  i_layer = Input(shape=(LD,))
  h_layer = Dense(128, activation=LeakyReLU(alpha=0.2))(i_layer)
  h_layer = BatchNormalization(momentum = 0.7)(h_layer)
  h_layer = Dense(256, activation=LeakyReLU(alpha=0.2))(h_layer)
  h_layer = BatchNormalization(momentum = 0.7)(h_layer)
  h_layer = Dense(512, activation=LeakyReLU(alpha=0.2))(h_layer)
  h_layer = BatchNormalization(momentum = 0.7)(h_layer)
  h_layer = Dense(1024, activation=LeakyReLU(alpha=0.2))(h_layer)
  h_layer = BatchNormalization(momentum = 0.7)(h_layer)
  o_layer = Dense(D, activation='tanh')(h_layer)
  return Model(i_layer, o_layer)

def make_discriminator(D):
  i_layer = Input(shape=(D,))
  h_layer = Dense(512, activation=LeakyReLU(alpha=0.2))(i_layer)
  h_layer = Dense(1024, activation=LeakyReLU(alpha=0.2))(h_layer)
  o_layer = Dense(1, activation='sigmoid')(h_layer)
  return Model(i_layer, o_layer)

d = make_discriminator(D)

d.compile(loss='binary_crossentropy',
          optimizer = Adam(0.0001),
          metrics = ['accuracy'])

g = make_generator(LD, D)

noise_i_layer = Input(shape=(LD,))
image = g(noise_i_layer, D)
d.trainable = False
prediction = d(image)
gan = Model(noise_i_layer, prediction)

gan.compile(loss='binary_crossentropy',
          optimizer = Adam(0.0001))

batch_size = 64
epochs = 20000
export_period = 1000

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

d_loss_list = []
g_loss_list = []

if not os.path.exists('GANS'):
  os.makedirs('GANS')

def get_images(epochs, rows, columns, LD, g):
  noise = np.random.randn(rows * columns, LD)
  images_by_generator = g.predict(noise)
  images_by_generator = 0.5 * images_by_generator + 0.5

  fig, ax = plt.subplots(rows, columns)
  index = 0
  for i in range(rows):
    for j in range(columns):
      ax[i,j].imshow(images_by_generator[index].reshape(H,W), cmap='gray')
      ax[i,j].axis('off')
      index+=1
  fig.savefig("GANS/%d.png" % epochs)
  plt.close

for e in range(epochs):
  index = np.random.randint(0, X_train.shape[0], batch_size)
  real = X_train[index]

  noise = np.random.randn(batch_size, LD)
  fake = g.predict(noise)

  d_loss_real, d_accuracy_real = d.train_on_batch(real, ones)
  d_loss_fake, d_accuracy_fake= d.train_on_batch(fake, zeros)
  d_loss = (d_loss_real + d_loss_fake) * 1/2 
  d_accuracy = (d_loss_real + d_loss_fake) * 1/2


  noise = np.random.randn(batch_size, LD)
  g_loss = gan.train_on_batch(noise, ones)

  d_loss_list.append(d_loss)
  g_loss_list.append(g_loss)

  if e % 200 == 0:
    print("epoch = {} d_loss = {} and g_loss = {}".format(e, d_loss, g_loss))
  if e % export_period == 0:
    get_images(e , 4,4, LD, g)

for i in range(epochs):
  if i % 1000 == 0:
    img = plt.imread('GANS/' + str(i) + ".png" )
    plt.figure()
    plt.imshow(img)

plt.plot(d_loss_list)
plt.plot(g_loss_list)
plt.show()