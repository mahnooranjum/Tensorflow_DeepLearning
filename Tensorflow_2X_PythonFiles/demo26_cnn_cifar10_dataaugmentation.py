
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

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

# Taken from https://www.cs.toronto.edu/~kriz/cifar.html
labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")

"""## As a rule of thumb

Remember that the pooling operation decreases the size of the image, and we lose information.

However, the number of features generally increases and we get more features extracted from the images.

The choices of hyperparams bother us sometimes, because DL has a lot of trial and error involved, we can choose the 

- learning rate

- number of layers

- number of neurons per layer 

- feature size 

- feature number 

- pooling size 

- stride 

On a side note, if you use strided convolution layers, they will decrease the size of the image as well


If we have images with different sizes as inputs; for example; H1 x W1 x 3 and H2 x W2 x 3, then the output will be flatten-ed to different sizes, this won't work for DENSE layers as they do not have change-able input sizes, so we use global max pooling to make a vector of size 1 x 1 x (#_Of_Feature_Maps_)
"""

from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_test = X_train / 255.0 , X_test / 255.0 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train, y_test = y_train.flatten(), y_test.flatten() 
print(y_train.shape)
print(y_test.shape)

classes = len(set(y_train))
print(classes)

input_shape = X_train[0].shape

i_layer = Input(shape = input_shape)
h_layer = Conv2D(32, (3,3),activation='relu', padding='same')(i_layer)
h_layer = BatchNormalization()(h_layer)
h_layer = Conv2D(64, (3,3), activation='relu', padding='same')(h_layer)
h_layer = BatchNormalization()(h_layer)
h_layer = Conv2D(128, (3,3), activation='relu', padding='same')(h_layer)
h_layer = BatchNormalization()(h_layer)
h_layer = MaxPooling2D((2,2))(h_layer)
h_layer = Conv2D(128, (3,3), activation='relu', padding='same')(h_layer)
h_layer = BatchNormalization()(h_layer)
h_layer = MaxPooling2D((2,2))(h_layer)
h_layer = Flatten()(h_layer)
h_layer = Dropout(0.5)(h_layer)
h_layer = Dense(512, activation='relu')(h_layer)
h_layer = Dropout(0.5)(h_layer)
o_layer = Dense(classes, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

y_pred = model.predict(X_test).argmax(axis=1) 
# only for sparse categorical crossentropy

evaluation_tf(report, y_test, y_pred, classes)

misshits = np.where(y_pred!=y_test)[0]
print("total Mishits = " + str(len(misshits)))
index = np.random.choice(misshits)
plt.imshow(X_test[index])
plt.title("Predicted = " + str(labels[y_pred[index]]) + ", Real = " + str(labels[y_test[index]]))

"""## LET'S ADD SOME DATA AUGMENTATION FROM KERAS 

taken from https://keras.io/api/preprocessing/image/
"""

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.1, 
                                                                 height_shift_range = 0.1, 
                                                                 horizontal_flip=True)

model_dg = Model(i_layer, o_layer)
model_dg.compile(optimizer='adam', 
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])
train_data_generator = data_generator.flow(X_train, y_train, batch_size)
spe = X_train.shape[0] // batch_size

report = model_dg.fit_generator(train_data_generator, validation_data=(X_test, y_test), steps_per_epoch=spe, epochs=50)

y_pred = model.predict(X_test).argmax(axis=1) 
# only for sparse categorical crossentropy

evaluation_tf(report, y_test, y_pred, classes)

misshits = np.where(y_pred!=y_test)[0]
print("total Mishits = " + str(len(misshits)))
index = np.random.choice(misshits)
plt.imshow(X_test[index])
plt.title("Predicted = " + str(labels[y_pred[index]]) + ", Real = " + str(labels[y_test[index]]))