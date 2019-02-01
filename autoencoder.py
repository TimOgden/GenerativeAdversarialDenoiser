
# coding: utf-8

# In[ ]:


import keras
from keras.layers import Dense, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import os

# In[ ]:

class AutoEncoder:
    

    def step_function(x):
        if x<.5:
            return 0
        else:
            return 1

    def build_model(self):
        #get_custom_objects().update({'step': Activation()})
        model = keras.models.Sequential([
            Dense(784, activation='relu', input_shape=(784,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.encoder_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(784, activation='sigmoid')
        ])
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        

    def getNoisyData(self):
        if os.path.exists('noisy_x_train.npy') and os.path.exists('noisy_x_test.npy') and os.path.exists('x_train.npy') and os.path.exists('x_test.npy'):
            self.noisy_x_train = np.load('noisy_x_train.npy')
            self.noisy_x_test = np.load('noisy_x_test.npy')
        else:
            self.noisy_x_train = addNoise(self.x_train)
            self.noisy_x_test = addNoise(self.x_test)
        


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


    def preprocess(self, x):
        x = x.astype('float32') / 255.
        return x.reshape(-1, np.prod(x.shape[1:])) # flatten



    # In[ ]:

    def constructAndTrain(self):
        self.model = self.build_model()
        print('model built')


        self.noisy_x_train = np.array(self.noisy_x_train)
        self.noisy_x_test = np.array(self.noisy_x_test)
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        np.save('noisy_x_train.npy', self.noisy_x_train)
        np.save('noisy_x_test.npy', self.noisy_x_test)
        np.save('x_train.npy', self.x_train)
        np.save('x_test.npy', self.x_test)

        self.x_train = self.preprocess(self.x_train)
        self.x_test = self.preprocess(self.x_test)
        self.noisy_x_train = self.preprocess(self.noisy_x_train)
        self.noisy_x_test = self.preprocess(self.noisy_x_test)
        self.model.fit(x=self.noisy_x_train, y=self.x_train, batch_size=500, epochs=10, verbose=1, validation_data=(self.noisy_x_test, self.x_test))
        self.model.save('model.h5')

    def constructAndLoad(self, filename):
        self.model = self.build_model()
        self.model = load_model(filename)
        self.x_train = np.load('x_train.npy')
        self.x_test = np.load('x_test.npy')
        self.noisy_x_train = np.load('noisy_x_train.npy')
        self.noisy_x_test = np.load('noisy_x_test.npy')
        print('model built')

    def retrieveTestData(self):
        return (self.x_test, self.noisy_x_test)


    def __init__(self, encoder_size=28):
        (x_train, _), (x_test, _) = mnist.load_data()
        self.x_train = x_train
        self.x_test = x_test
        self.encoder_size = encoder_size
        self.getNoisyData()