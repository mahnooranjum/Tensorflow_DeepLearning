# -*- coding: utf-8 -*-
"""Demo121_CNNforNotMNIST.ipynb


# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`

## We'll take over the NotMNIST dataset and use a basic ANN to classify the digits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import sys, os

!wget http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
!wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz

!ls

!tar -xzf notMNIST_large.tar.gz
!tar -xzf notMNIST_small.tar.gz

!ls

!ls notMNIST_large

!ls notMNIST_small

path = 'notMNIST_large'
path_val = 'notMNIST_small'

plt.imshow(image.load_img(path + '/A/ZGVhckpvZSBJdGFsaWMudHRm.png'))
plt.show()

IMAGE_SIZE = [28,28]
train_images = glob(path + '/*/*.png')
validation_images = glob(path_val + '/*/*.png')

# Number of classes 
classes = glob(path + '/*')

print(classes)

plt.imshow(image.load_img(np.random.choice(train_images)))
plt.show()

plt.imshow(image.load_img(np.random.choice(validation_images)))
plt.show()

sample = image.load_img(np.random.choice(validation_images))

type(sample)

sample.size

Y = len(classes)

Y

print(len(train_images))
print(len(validation_images))

batch_size = 254

gen_object = ImageDataGenerator(rescale=1.0/255.0)

I_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]

print(I_SIZE)

train_generator = gen_object.flow_from_directory(path, target_size=IMAGE_SIZE, batch_size=1, color_mode="grayscale", shuffle=True)
validation_generator = gen_object.flow_from_directory(path_val, target_size=IMAGE_SIZE, batch_size=1,color_mode="grayscale",  shuffle=True)

N = len(train_images)
N_val = len(validation_images)

train_generator[1][0].shape

plt.imshow(train_generator[10][0].reshape(28,28))

train_generator[1][1].shape

len(train_generator)

X_train = np.zeros((N, 28, 28))
y_train = np.zeros((N, Y))

X_train.shape

k = 0
while k != N:
  try:
    i = np.random.randint(0, N)
    X_train[k,:] = train_generator[i][0].reshape(28,28)
    y_train[k,:] = train_generator[i][1]
    k+=1
  except:
    pass

y_train[0]

X_train[0, 0]

X_train.shape

y_train.shape

plt.imshow(X_train[10].reshape(28,28))

X_train.shape

X_test = np.zeros((N_val, 28, 28))
y_test = np.zeros((N_val, Y))

k = 0
while k != N_val:
  try:
    i = np.random.randint(0, N_val)
    X_test[k,:] = validation_generator[i][0].reshape(28,28)
    y_test[k,:] = validation_generator[i][1]
    k+=1
  except:
    pass

X_test.shape

X_train.shape

X_test.shape

y_train.shape

y_test.shape

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

input_shape = X_train[0].shape

print(input_shape)

i_layer = Input(shape = input_shape)
h_layer = Conv2D(32, (3,3), strides = 2, activation='relu')(i_layer)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu')(h_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu')(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(Y, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 1)

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))

y_pred = model.predict(X_test).argmax(axis=1)

print(y_test.shape)
print(y_pred.shape)

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
plt.xticks(list(range(10)))
plt.yticks(list(range(10)))
plt.title('Confusion matrix')
plt.colorbar()
plt.show()

labels = "A,B,C,D,E,F,G,H,I,J".split(",")

misshits = np.where(y_pred!=y_test)[0]
index = np.random.choice(misshits)
plt.imshow(X_test[index].reshape(28,28), cmap='gray')
plt.title("Predicted = " + str(labels[y_pred[index]]) + ", Real = " + str(labels[y_test[index]]))