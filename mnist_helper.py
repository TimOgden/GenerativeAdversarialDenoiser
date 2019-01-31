
# coding: utf-8

# In[2]:


from keras.datasets import mnist


# In[51]:


import matplotlib.pyplot as plt
import numpy as np


# In[65]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[60]:


# Function that adds noise to any array of images
def addNoise(images, multiplier=2):
    for image in images:
        noise = np.random.normal(0,multiplier,784)
        noise = np.reshape(noise, (28,28))
        for r in range(28):
            for c in range(28):
                image[r][c] += noise[r][c]


# In[64]:


def import_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

def getNoisyData():
    addNoise(x_train)
    addNoise(x_test)
    return (x_train, x_test)
    
def getRegularData():
    return (x_train, x_test)

