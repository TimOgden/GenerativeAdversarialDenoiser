
# coding: utf-8

# In[ ]:


import keras
from keras.layers import Dense, Flatten, Input
from keras.datasets import mnist
import numpy as np
import os

# In[ ]:


def build_model(middle_nodes = 3):
    model = keras.models.Sequential([
        Dense(784, activation='relu', input_shape=(28,28)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(middle_nodes, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(784, activation='sigmoid')
    ])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    

def import_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

def getNoisyData(x_train, x_test):
    noisy_x_train = addNoise(x_train)
    noisy_x_test = addNoise(x_test)
    #noisy_x_train = [np.reshape(val,(784,)) for val in noisy_x_train]
    #noisy_x_test = [np.reshape(val,(784,)) for val in noisy_x_test]
    return (noisy_x_train, noisy_x_test)
    
def getRegularData():
    #x_train = [np.reshape(val,(784,)) for val in x_train]
    #x_test = [np.reshape(val,(784,)) for val in x_test]
    (x_train, _), (x_test, _) = mnist.load_data()
    return (x_train[:int(.5*len(x_train))], x_test)


def addNoise(images, multiplier=2):
    new_images = []
    i = 0
    for image in images:
        new_image = np.array(image)
        noise = np.random.normal(0,multiplier,784)
        
        noise = np.reshape(noise, (28,28))
        for r in range(28):
            for c in range(28):
                new_image[r][c] += noise[r][c]
        new_images.append(new_image)
        i+=1
        if i % 2000 == 0:
            print(i,'/',len(images))
    return new_images






# In[ ]:


model = build_model(middle_nodes=28)
print('model built')

x_train = None
x_test = None
noisy_x_train = None
noisy_x_test = None
if os.path.exists('noisy_x_train.npy') and os.path.exists('noisy_x_test.npy') and os.path.exists('x_train.npy') and os.path.exists('x_test.npy'):
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    noisy_x_train = np.load('noisy_x_train.npy')
    noisy_x_test = np.load('noisy_x_test.npy')
else:
    (x_train, x_test) = getRegularData()
    print('did it just become 0?', len(x_train))
    print('regular data aquired')

    (noisy_x_train, noisy_x_test) = getNoisyData(x_train, x_test)
    print('noisy data aquired')
    print('noisy data len:', len(noisy_x_train))
    print('reg data len:', len(x_train))
    noisy_x_train = np.array(noisy_x_train)
    noisy_x_test = np.array(noisy_x_test)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    np.save('noisy_x_train.npy', noisy_x_train)
    np.save('noisy_x_test.npy', noisy_x_test)
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)

x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
noisy_x_train = np.reshape(noisy_x_train, (-1, 28, 28, 1))
noisy_x_test = np.reshape(noisy_x_test, (-1, 28, 28, 1))
model.fit(x=noisy_x_train, y=x_train, batch_size=500, epochs=10, verbose=2, validation_data=(noisy_x_test, x_test))


