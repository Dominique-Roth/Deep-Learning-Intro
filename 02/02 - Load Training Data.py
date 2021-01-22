#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import cv2
import numpy as np
from tqdm import tqdm

DATADIR = f'{os.path.abspath(os.getcwd())}/PetImages'
CATEGORIES = ['Dog', 'Cat']

IMG_SIZE = 50

print(DATADIR)


# In[2]:


# Preperation of data

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        print(path)
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                
                # Debug
                # plt.imshow(new_array, cmap='gray')
                # plt.show()
                # break
                # print(f'{new_array} - {class_num}')
                
                training_data.append([new_array, class_num])  # add this to our training_data
            except:
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
        
create_training_data()


# In[3]:


print(len(training_data))


# In[11]:


import random
random.shuffle(training_data)


# In[12]:


for sample in training_data[:10]:
    print(sample[1])


# In[13]:


X = []
y = []


# In[14]:


for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[15]:


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[16]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# In[17]:


X[1]


# In[ ]:




