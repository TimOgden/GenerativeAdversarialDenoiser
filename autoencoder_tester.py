import keras
from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
import autoencoder
import numpy as np

def restoreImage(image):
	image *= 255
	return image.reshape(28,28)

def flattenImage(image):
	return image.reshape(784)
def step(image):
	for r in range(len(image)):
		if image[r].any() < .5:
			image[r] = 0
		else:
			image[r] = 1
	return image

#model = build_model()
#model = load_model('model.h5')
ae = autoencoder.AutoEncoder()
ae.constructAndLoad('model.h5')
noisy_x_test = ae.noisy_x_test
for arg, c in enumerate(sys.argv):
	if c==0:
		pass
	im_num = int(arg)

	noisy_image = noisy_x_test[im_num]
	noisy_image = flattenImage(noisy_image)
	print('noisy_image shape', noisy_image.shape)
	predicted = ae.model.predict(np.array([noisy_image]))
	noisy_image = restoreImage(noisy_image)
	predicted = restoreImage(predicted)
	plt.imshow(noisy_image, cmap=plt.cm.binary)
	plt.show()

	plt.imshow(predicted, cmap=plt.cm.binary)
	plt.show()

