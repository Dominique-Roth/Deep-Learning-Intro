#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import tensorflow as tf

CATEGORIES = ['Cat','Dog']

def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("models/32x1-3e-CNN.model")


# In[11]:


prediction = model.predict([prepare('cat.jpeg')]) # ALWAYS PREDICT A LIST!!!

print(prediction)
print(CATEGORIES[int(prediction[0][0])])


# In[12]:


prediction = model.predict([prepare('dog.jpeg')]) # ALWAYS PREDICT A LIST!!!

print(prediction)
print(CATEGORIES[int(prediction[0][0])])


# In[ ]:




