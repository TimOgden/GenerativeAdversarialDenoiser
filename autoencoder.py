
# coding: utf-8

# In[ ]:


import keras
from keras.layers import Dense, Flatten, Input
from keras.datasets import mnist
import numpy as np


# In[ ]:


def build_model(middle_nodes = 3):
    model = keras.models.Sequential([
        Dense(784, activation='relu', input_shape=(784,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(middle_nodes, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(784, activation='sigmoid')
    ])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    


# In[ ]:


def addNoise(images, multiplier=2):
    new_images = []
    for image in images:
        new_image = np.array(image)
        noise = np.random.normal(0,multiplier,784)
        noise = np.reshape(noise, (28,28))
        for r in range(28):
            for c in range(28):
                new_image[r][c] += noise[r][c]
        new_images.append(new_image)
    return new_images

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def import_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

def getNoisyData():
    noisy_x_train = addNoise(x_train)
    noisy_x_test = addNoise(x_test)
    noisy_x_train = [np.reshape(val,(784,)) for val in noisy_x_train]
    noisy_x_test = [np.reshape(val,(784,)) for val in noisy_x_test]
    return (noisy_x_train, noisy_x_test)
    
def getRegularData():
    x_train = [np.reshape(val,(784,)) for val in x_train]
    x_test = [np.reshape(val,(784,)) for val in x_test]
    return (x_train, x_test)


# In[ ]:


model = build_model(middle_nodes=28)
noisy_x_train, noisy_x_test = getNoisyData()
x_train, x_test = getRegularData()
try:
    model.fit(x=noisy_x_train, y=x_train, epochs=10, verbose=2, validation_data=(noisy_x_test, x_test))
except Exception as e:
    print(e)
    pass

