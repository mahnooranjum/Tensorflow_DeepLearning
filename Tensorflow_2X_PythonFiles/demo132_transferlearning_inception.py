# -*- coding: utf-8 -*-
"""Demo132_TransferLearning_Inception.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## The basic idea is that the CNNs learn things in a hierarchical manner

Thus, if we have a great classifier that can classify 10k objects with 99% accuracy, it must have great primary layers that learn the basic features perfectly. 

So we can use the primary layers for other things, like classifying 11k objects !

## Let's talk about some pretrained models

- VGG16, VGG19 having 16 and 19 layers respectively 

- ResNet is larger than VGG, with different branches that learn something different; ResNet50, ResNet151, are variations

- Inception has multiple convolutions in parallel branches, different sizes of filters are tried and then their results are appended 


## Rule of Thumb

Computations take time, we must work around them. As a rule of thumb, if you're using data augmentation, put CNN computations inside the training loop.
"""

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import sys, os

from tensorflow.keras.applications.inception_v3 import InceptionV3 as pretrained, preprocess_input

"""## Let's import the dataset"""

!wget --passive-ftp --prefer-family=ipv4 https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz

!ls

!tar -xzvf imagenette2-320.tgz

!ls imagenette2-320/train

!ls imagenette2-320/val

path = 'imagenette2-320/train'
path_val = 'imagenette2-320/val'

plt.imshow(image.load_img(path + '/n01440764/n01440764_1514.JPEG'))
plt.show()

IMAGE_SIZE = [200,200]
train_images = glob(path + '/*/*.JPEG')
validation_images = glob(path_val + '/*/*.JPEG')

# Number of classes 
classes = glob(path + '/*')

classes

plt.imshow(image.load_img(np.random.choice(train_images)))
plt.show()

plt.imshow(image.load_img(np.random.choice(validation_images)))
plt.show()

C = 3
pretrained_model = pretrained(input_shape = IMAGE_SIZE + [C],
                              weights = 'imagenet',
                              include_top = False)
pretrained_model.trainable = False

Y = len(classes)
h_layer = Flatten()(pretrained_model.output)
o_layer = Dense(Y, activation= 'softmax')(h_layer)

model = Model(pretrained_model.input, o_layer)

model.summary()

gen_object = ImageDataGenerator(rotation_range = 10,
                                width_shift_range = 0.1, 
                                height_shift_range = 0.1, 
                                zoom_range = 0.2,
                                horizontal_flip = True,
                                preprocessing_function = preprocess_input)

len(train_images)

batch_size = 254

train_generator = gen_object.flow_from_directory(path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size)
validation_generator = gen_object.flow_from_directory(path_val, target_size=IMAGE_SIZE, batch_size=batch_size)

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

report = model.fit_generator(train_generator, 
                             validation_data=validation_generator, 
                             epochs=10,
                             steps_per_epoch = int(np.ceil(len(train_images)/batch_size)),
                             validation_steps = int(np.ceil(len(validation_images)/batch_size)))

plt.plot(report.history['loss'], label = 'training_loss')
  plt.plot(report.history['val_loss'], label = 'validation_loss')
  plt.legend()
  plt.show()
  plt.plot(report.history['accuracy'], label = 'training_accuracy')
  plt.plot(report.history['val_accuracy'], label = 'validation_accuracy')
  plt.legend()
  plt.show()

