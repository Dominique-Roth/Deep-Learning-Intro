#!/usr/bin/env python
# coding: utf-8

# In[5]:
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from datetime import datetime


NAME = f'Cats-vs-Dogs-cnn-64x2-{int(time.time())}'
tensorboard_callback = TensorBoard(log_dir=f'logs/{NAME}')

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X=np.array(X/255.0)

y=np.array(y)

# Model Type
model = Sequential()

# First we define 2 Convolutional Layers with 256 nodes
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, validation_data=(X, y), callbacks=[tensorboard_callback],)


# In[ ]:




