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

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
epochs = [3, 5, 10]

#NAME = f'Cats-vs-Dogs-cnn-64x2-{int(time.time())}'

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X=np.array(X/255.0)

y=np.array(y)

for epoch in epochs:
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = f"{conv_layer} conv - {layer_size} nodes - {dense_layer} dense - {epoch} epoch || {time.time()}"
                tensorboard_callback = TensorBoard(log_dir=f'logs/{NAME}')
                print(NAME)

                # Model Type
                model = Sequential()

                # First we define 2 Convolutional Layers with 256 nodes
                model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
                for l in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                model.fit(X, y, batch_size=32, epochs=epoch, validation_split=0.3, validation_data=(X, y), callbacks=[tensorboard_callback],)


                # In[ ]:




