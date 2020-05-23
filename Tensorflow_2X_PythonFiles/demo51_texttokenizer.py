# -*- coding: utf-8 -*-
"""Demo51_TextTokenizer.ipynb

# **Spit some [tensor] flow**

We need to learn the intricacies of tensorflow to master deep learning

`Let's get this over with`

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
print(tf.__version__)

from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, SimpleRNN, GRU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get the dataset 
data = ["Hello world", "I ain't saying hello to you", "what's up with all the hellos"]

MAX_SIZE = 10000
tokenizer = Tokenizer(num_words=MAX_SIZE)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

print(sequences)

print(tokenizer.word_index)

T = 4

data = pad_sequences(sequences, 
                    maxlen = T, 
                    padding = 'post')

print(data)

data = pad_sequences(sequences, 
                    maxlen = T, 
                    padding = 'pre')

print(data)

data = pad_sequences(sequences, 
                    maxlen = T, 
                    truncating = 'pre',
                    padding = 'post')

print(data)

data = pad_sequences(sequences, 
                    maxlen = T, 
                    truncating = 'post',
                    padding = 'post')

print(data)