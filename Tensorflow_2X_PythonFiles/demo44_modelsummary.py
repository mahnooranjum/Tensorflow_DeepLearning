
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Adamax

"""## Let's talk about weights in NN 

We will use the following notation 

N = Number of samples in the dataset 

D = Dimensions of the feature space 

T = Sequence length or window length 

U = Number of units in the layer 

O = Output feature space

## Let's get some synthetic data
"""

N = 1
T = 5
D = 3
Y = 2
X = np.random.randn(N, T, D)

print(X.shape)

U = 4
i_layer = Input(shape = (T,D))
h_layer = SimpleRNN(U, activation='relu')(i_layer)
o_layer = Dense(Y)(h_layer)

model = Model(i_layer, o_layer)

model.summary()

"""## We have 42 trainable parameters? Let's see:"""

len(model.layers[1].get_weights())

# So we have three matrices for weights in the first hidden layer, cool!
W1, W2, W3 = model.layers[1].get_weights()
print(W1.shape)
print(W2.shape)
print(W3.shape)

# Go into the details; we are supposed to have three weights: 
# Input to hidden = D x U 
# Hidden to hidden  = U x U 
# Bias term = Vector (U)

print("D*U = " + str(D*U))
print("U*U = " + str(U*U))
print("U = " + str(U))
print(D*U + U*U + U)

len(model.layers[2].get_weights())

# So we have two matrices for weights in the last layer, cool!
W1, W2 = model.layers[2].get_weights()
print(W1.shape)
print(W2.shape)

print("U*Y = " + str(U*Y))
print("Y = " + str(Y))
print(U*Y + Y)

# Go into the details; we are supposed to have three weights: 
# Hidden to output = U x Y 
# Output bias  = Y