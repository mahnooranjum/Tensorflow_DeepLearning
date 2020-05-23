# -*- coding: utf-8 -*-
"""Demo52_RNN_MovieReviews.ipynb


# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

"""## Look at the following equations 

Well now we're going to go into the details of the implementation; 

```
x_pred_t = w0 + w1 * x_t-1 + w2 * x_t-2 + w3 * x_t-3 + w4 * x_t-4

x_pred_t+1 = w0 + w1 * x_pred_t + w2 * x_t-1 + w3 * x_t-2 + w4 * x_t-3
```

This is an autoregressive model now. 

```
x_pred_5 = w0 + w1 * x_4 + w2 * x_3 + w3 * x_2 + w4 * x_1

x_pred_6 = w0 + w1 * x_pred_5 + w2 * x_4 + w3 * x_3 + w4 * x_2

```

We know that the predictions at time T in an RNN depend on all the previous times. Thus, indirectly, y(t) depends on x(t), x(t-1), x(t-2),....., x(2), x(1)

To optimize the weights, we must take the derivative of the equation containing the weights; however, there will be numerous W_input_to_hidden weights in the network. 


The RNN keeps nesting the older timestamps; the derivatives use multiplication and chain rule in composite functions, thus, the more older the timestamp, the more its gradient vanishes.

## So how do GRU's work?

Throwback to the RNN equation: 

h(t) = activation(W(x,h) x(t) + W(h,h) h(t-1) + b(h))

GRU's calulate two other things to calculate h(t):

- update gate vector z(t)

- reset gate vector r(t)

z(t) = sigmoid(W(x,z) x(t) + W(h,z) h(t-1) + b(z))

r(t) = sigmoid(W(x,r) x(t) + W(h,r) h(t-1) + b(r))

```
h(t) = (1 - z(t)) * h(t-1)  +

        z(t) tanh(W(x,h) x(t) + W(h,h) (r(t) * h(t-1)) + b(h))
```

(*) ==== element wise multiplication

z(t), r(t), h(t) ===== Size U

## Z(t) 

Should we take the new value of h(t)? or keep h(t-1)? 

- z(t) close to 0, take h(t-1)

- z(t) close to 1, take h(t)

So: 

```
h(t) = (1 - z(t)) * h(t-1)  +

        z(t) tanh(W(x,h) x(t) + W(h,h) (r(t) * h(t-1)) + b(h))
```

Becomes 

```
h(t) = (factor keep h(t-1)) * h(t-1) + 

       (discard h(t-1)) * RNN(x(t), h(t-1))
```

## R(t) 

Change the value of h(t-1) 

- r(t) close to 0, zero value h(t-1)

- r(t) close to 1, keep value h(t-1)


## So how do LSTM's work?

### https://towardsdatascience.com/

We add another state to the mix, the cell state c(t)

we add three different neurons:

forget neuron = f(t) 

This gate decides which information should be thrown away or kept. Input from h(t-1) and x(t) is passed through this gate; and it uses sigmoid to either forget (0) or remember (1) it.


input gate neuron = i(t)

We use this to update the cell state. We pass the h(t-1) and x(t) to the sigmoid function. This will decide which values will be updated in the cell state.

output gate neuron = o(t) 

The output gate decides what the next hidden state h(t) should be. Remember that the hidden state contains information on previous inputs. The hidden state is also used for predictions. 

First, we pass the previous hidden state and the current input into a sigmoid function.

Then we pass the newly modified cell state to the tanh function. 

We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. 

The output is the hidden state. 

The new cell state and the new hidden is then carried over to the next time step.

```

f(t) = sigmoid ( W(x,f) x(t) + W(h, f)h(t-1) + b(f) )

i(t) = sigmoid ( W(x,i) x(t) + W(h, i)h(t-1) + b(i) ) 

o(t) = sigmoid ( W(x,o) x(t) + W(h, o)h(t-1) + b(o) ) 


c(t) = f(t) * c(t-1) + 

       i(t) * tanh ( W(x,c) x(t) + W(h,c) h(t-1) + b(c) ) 


h(t) = o(t) * tanh( c(t) )

```

## So we OHE the last NLP problem, why not do the same and feed it to the neural network? Well because, features in a language, are not independent. 


Let's explore this: 

The quick brown fox jumps over __________________

See you know the end of this sentence because you know the words right? 

well wb this: 

over _____________________

Now we don't know the end of this sentence. 

So in tensorflow, to save computations, we have the embedding layer: 

### Step 1: Words to ints

Nothing deep about deep learning ----> 13 43 32 43 98

### Step 2: Ints to word vector 

13 43 32 43 98 ------> [0.9, 1.2] [-0.4, 0.2] [0.3, 0.3] [-0.4, 0.2] [0.2, 0.5] 

T -----> T x D


### We can use word2vec to make sure the embedding layer has similar words close to each other
"""

from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, SimpleRNN, LSTM, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""## Let's import the dataset"""

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

data = pd.read_csv("sample_data/train.tsv", delimiter='\t')
data.head()

Y = len(data.Sentiment.unique())
print(Y)

X = data.iloc[:, 2]
y = data.iloc[:, -1].values

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

MAX_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_SIZE)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

word2index = tokenizer.word_index
V = len(word2index)
print("tokens = " + str(V))

X_train = pad_sequences(sequences_train)

X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

N, T = X_train.shape

# Let's talk about D, what is it and how do we set it? 
# This is the dimensionality of the embedding layer, essentially the vector that each word becomes 

D = 10 
# hidden units
U = 20

i_layer = Input(shape = (T,))
h_layer = Embedding(V+1, D)(i_layer)
# V+1 because https://github.com/tensorflow/tensorflow/issues/38619
h_layer = LSTM(U,return_sequences=True)(h_layer)
h_layer = GlobalMaxPooling1D()(h_layer)
o_layer = Dense(Y, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, epochs = 20, validation_data=(X_test, y_test))

plt.plot(report.history['loss'], label='training_loss')
plt.plot(report.history['val_loss'], label='validation_loss')
plt.legend()

"""## Seems like the model is overfitting, let's tune it"""

# Let's talk about D, what is it and how do we set it? 
# This is the dimensionality of the embedding layer, essentially the vector that each word becomes 

D = 15
# hidden units
U = 30

i_layer = Input(shape = (T,))
h_layer = Embedding(V+1, D)(i_layer)
# V+1 because https://github.com/tensorflow/tensorflow/issues/38619

h_layer = LSTM(U,return_sequences=True)(h_layer)
h_layer = GlobalMaxPooling1D()(h_layer)
h_layer = Dense(5, activation='relu')(h_layer)
h_layer = Dropout(0.3)(h_layer)
o_layer = Dense(Y, activation='softmax')(h_layer)

model = Model(i_layer, o_layer)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

report = model.fit(X_train, y_train, epochs = 10, validation_data=(X_test, y_test))

plt.plot(report.history['loss'], label='training_loss')
plt.plot(report.history['val_loss'], label='validation_loss')
plt.legend()

print(X_train.shape)
print(X_test.shape)

y_pred = model.predict(X_test).argmax(axis=1)

evaluation_tf(report, y_test, y_pred, Y)