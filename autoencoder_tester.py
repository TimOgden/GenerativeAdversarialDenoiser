import keras
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
import autoencoder
import numpy as np
import os

def restoreImage(image):
	image *= 255
	return image.reshape(28,28)

def flattenImage(image):
	return image.reshape(784)
def step(image):
	for r in range(len(image)):
		for c in range(len(image[0])):
			if image[r][c] < .5:
				image[r][c] = 0
			else:
				image[r][c] = 1
	return image

#model = build_model()
#model = load_model('model.h5')
encoder_size = 32
ae = autoencoder.AutoEncoder(encoder_size=encoder_size)



filename = str(encoder_size) + '-encoder'
epochs = 10
full_filename = filename + '-' + str(epochs) + '.h5'
if os.path.exists(full_filename):
	ae.constructAndLoad(full_filename)
else:
	ae.constructAndTrain(full_filename, epochs=epochs)


noisy_x_test = ae.noisy_x_test
y_test = ae.y_test
#print(len(noisy_x_test))
c = 0
for arg in sys.argv:
	if not c==0:
		im_num = int(arg)

		noisy_image = noisy_x_test[im_num]
		noisy_image = flattenImage(noisy_image)
		#print(im_num)
		predicted = ae.model.predict(np.array([noisy_image]))
		predicted = step(predicted)
		noisy_image = restoreImage(noisy_image)
		predicted = restoreImage(predicted)
		plt.imshow(noisy_image, cmap=plt.cm.binary)
		plt.show()
		print('Should be a', y_test[im_num])
		plt.imshow(predicted, cmap=plt.cm.binary)
		plt.show()
	c+=1

