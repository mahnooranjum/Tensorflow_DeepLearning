# -*- coding: utf-8 -*-
"""Demo140_CNN_Weights.ipynb


# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0 , X_test / 255.0 
print(X_train.shape)
print(X_test.shape)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print(X_train.shape)
print(X_test.shape)
# SHAPE 
# N x H x W x Colors 
# Colors = 1 for grayscale 
# Fashion MNIST is grayscale

classes = len(set(y_train))
print(classes)

input_shape = X_train[0].shape

"""## ZEROS"""

i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=tf.zeros_initializer())(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer=tf.zeros_initializer())(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

"""## ONES"""

i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=tf.ones_initializer())(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= tf.ones_initializer())(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

"""## UNIFORM"""

weights_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)



"""## ORTHOGONAL"""

weights_init = tf.keras.initializers.Orthogonal()

i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

"""## XAVIER UNIFORM"""

weights_init = tf.keras.initializers.GlorotUniform()


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)



"""## XAVIER NORMAL"""

weights_init = tf.keras.initializers.GlorotNormal()


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)



"""## CONSTANT"""

weights_init = tf.constant_initializer(5)


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)



"""## VARIANCE SCALING"""

weights_init = tf.keras.initializers.VarianceScaling()


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

"""## TRUNCATED NORMAL"""

weights_init = tf.keras.initializers.TruncatedNormal()


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

"""## RANDOM NORMAL"""

weights_init = tf.random_normal_initializer()


i_layer = Input(shape = input_shape)
h_layer = Conv2D(64, (3,3), strides = 2, activation='relu', 
                 kernel_initializer=weights_init)(i_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.4)(h_layer)
h_layer = Dense(128, activation='relu',kernel_initializer= weights_init)(h_layer)
h_layer = Dropout(0.4)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

