from fashionmnist_autoencoder import build_decoder, build_autoencoder
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.datasets import fashion_mnist

(_,_), (x_test, _) = fashion_mnist.load_data()

model = load_model('fashion_autoencoder_best.h5')

decoder = build_decoder()
decoder.load_weights('fashion_autoencoder_best.h5', by_name=True)


get_med_layer_output = K.function([model.layers[0].input],
										[model.layers[8].output])

def get_img(weights):
	weights = np.reshape(weights, (1,8))
	img = decoder.predict(weights)
	img = np.reshape(img, (28,28))
	return img

def lerp(a,b,f):
	return a + f * (b - a)

def get_weights(img_index):
	img = x_test[img_index]
	img = np.reshape(img,(1,28,28))
	layer_output = get_med_layer_output([img])[0]
	#print(layer_output[0])
	return layer_output[0]

orig_weights = get_weights(5)
end_weights = get_weights(1)

for index in range(1,11):
	plt.subplot(1,11,index)
	img_weights = []
	for i in range(len(orig_weights)):
		img_weights.append(lerp(orig_weights[i], end_weights[i], index/10))
	np_im_weights = np.array(img_weights)
	img = get_img(np_im_weights)
	plt.imshow(img)
	plt.axis('off')
plt.show()