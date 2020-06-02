# -*- coding: utf-8 -*-
"""Demo122_ANNforMNIST.ipynb
# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`

## We'll take over the MNIST dataset and use a basic ANN to classify the digits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

"""## Let's load the dataset using keras datasets"""

dataset = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = dataset.load_data()
X_train, X_test = X_train/255.0 , X_test/255.0

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (X_train.shape[1], X_train.shape[2])),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 30)

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()

plt.plot(report.history['accuracy'], label="accuracy")
plt.plot(report.history['val_accuracy'], label="validation_accuracy")
plt.legend()

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

misshits = np.where(y_pred!=y_test)[0]
index = np.random.choice(misshits)
plt.imshow(X_test[index], cmap='gray')
plt.title("Predicted = " + str(y_pred[index]) + ", Real = " + str(y_test[index]))







from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='mnist.best.hdf5', save_best_only = True)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (X_train.shape[1], X_train.shape[2])),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 30, callbacks = [checkpointer])

plt.plot(report.history['loss'], label="loss")
plt.plot(report.history['val_loss'], label="validation_loss")
plt.legend()

plt.plot(report.history['accuracy'], label="accuracy")
plt.plot(report.history['val_accuracy'], label="validation_accuracy")
plt.legend()

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))

model_saved = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape = (X_train.shape[1], X_train.shape[2])),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

model_saved.load_weights('mnist.best.hdf5')

model_saved.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("Train eval: ", model_saved.evaluate(X_train, y_train))
print("Test eval: ", model_saved.evaluate(X_test, y_test))

