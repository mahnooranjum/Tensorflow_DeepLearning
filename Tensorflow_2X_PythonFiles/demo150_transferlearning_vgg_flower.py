# -*- coding: utf-8 -*-
"""Demo150_TransferLearning_VGG_Flower.ipynb

# **Spit some [tensor] flow**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## The basic idea is that the CNNs learn things in a hierarchical manner

Thus, if we have a great classifier that can classify 10k objects with 99% accuracy, it must have great primary layers that learn the basic features perfectly. 

So we can use the primary layers for other things, like classifying 11k objects !
"""

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import sys, os

from tensorflow.keras.applications.vgg16 import VGG16 as pretrained, preprocess_input

"""## Let's import the dataset"""

!wget --passive-ftp --prefer-family=ipv4 https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz

!ls

!tar -xzvf 17flowers.tgz

!ls jpg/ -1 | wc -l

!find jpg/ -type f ! -name '*.jpg'

!rm jpg/files.txt jpg/files.txt~

!find jpg/ -type f ! -name '*.jpg'

!ls jpg/ -1 | wc -l

#Forming data directories:
import shutil
j = 1
total = 1360
for i  in range(1, total):
    fpath = f"jpg/image_{str(i).zfill(4)}.jpg"
    destPath = 'flower17_dataset/'+str(j)
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    shutil.copy(fpath, destPath)

    if i%80==0:
        j+=1

!ls flower17_dataset

path = 'flower17_dataset'

IMAGE_SIZE = [200,200]
train_images = glob(path + '/*/*.jpg')

# Number of classes 
classes = glob(path + '/*')

classes

plt.imshow(image.load_img(np.random.choice(train_images)))
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
                                preprocessing_function = preprocess_input, 
                                validation_split=0.3)

len(train_images)

n_trains = int(np.ceil(1359*0.7))
print(n_trains)

n_vals = int(np.floor(1359*0.3))
print(n_vals)

batch_size = 254

train_generator = gen_object.flow_from_directory(path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size, subset='training')
validation_generator = gen_object.flow_from_directory(path, target_size=IMAGE_SIZE, batch_size=batch_size, subset='validation')

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['accuracy'])

report = model.fit_generator(train_generator, 
                             validation_data=validation_generator, 
                             epochs=10,
                             steps_per_epoch = int(np.ceil(n_trains/batch_size)),
                             validation_steps = int(np.ceil(n_vals/batch_size)))

plt.plot(report.history['loss'], label = 'training_loss')
  plt.plot(report.history['val_loss'], label = 'validation_loss')
  plt.legend()
  plt.show()
  plt.plot(report.history['accuracy'], label = 'training_accuracy')
  plt.plot(report.history['val_accuracy'], label = 'validation_accuracy')
  plt.legend()
  plt.show()

validation_generator[0][0][253].shape

validation_generator[0][1][253].shape

y_pred = model.predict_generator(validation_generator, steps=np.ceil(n_vals/254))

y_pred = y_pred.argmax(axis=1)

y_test = validation_generator.classes

y_test.shape

y_pred.shape

import cv2
from google.colab.patches import cv2_imshow

index = np.random.randint(0,407)
y_test_idx = y_test[index]
X_test_idx = validation_generator[0][0][index] 
y_pred_idx = y_pred[index]
print("Predicted = " + str(y_pred[index]) + ", Real = " + str(y_test[index]))
cv2_imshow((X_test_idx+(255/2)))
#plt.imshow((X_test_idx+(255/2))/255)
#plt.title("Predicted = " + str(y_pred[index]) + ", Real = " + str(y_test[index]))