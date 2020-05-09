
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
type(data)
print(data.keys())

print(data.target_names)
print("============================================")
print(data.feature_names)

X = data.data

y = data.target

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

N, D = X_train.shape

from sklearn.preprocessing import StandardScaler 
scaleObj = StandardScaler()
X_train = scaleObj.fit_transform(X_train)
X_test = scaleObj.transform(X_test)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Input(shape=(D,)),
                                    tf.keras.layers.Dense(1,activation='sigmoid')
])


model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

report = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print("Train eval: ", model.evaluate(X_train, y_train))
print("Test eval: ", model.evaluate(X_test, y_test))