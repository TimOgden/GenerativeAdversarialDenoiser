from fashionmnist_autoencoder import build_decoder
import numpy as np
import cv2
import matplotlib.pyplot as plt
model = build_decoder()
model.load_weights('fashion_autoencoder_best.h5', by_name=True)

# Initial weights set as all midpoint of 0 and 1
orig_weights = [15.668705, 0, 15.38014, 18.648127, 0, 0, 0, 0]
end_weights = [7.966931, 0, 6.1681437, 8.659827, 0, 10.987252, 0, 0]

def get_img(weights):
	weights = np.reshape(weights, (1,8))
	img = model.predict(weights)
	img = np.reshape(img, (28,28))
	return img

def lerp(a,b,f):
	return a + f * (b - a)


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