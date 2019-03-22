from keras.layers import *
from keras.models import Sequential, load_model
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

(x_train, _), (x_test, _) = fashion_mnist.load_data()

def build_autoencoder(input_shape=(28,28)):
	model = Sequential([
		Flatten(input_shape=input_shape),
		Dense(784, activation='relu', name='dense1'),
		Dense(512, activation='relu', name='dense2'),
		Dense(256, activation='relu', name='dense3'),
		Dense(128, activation='relu', name='dense4'),
		Dense(64, activation='relu', name='dense5'),
		Dense(32, activation='relu', name='dense6'),
		Dense(16, activation='relu', name='dense7'),
		Dense(8, activation='relu', name='dense8'),
		Dense(16, activation='relu', name='dense9'),
		Dense(32, activation='relu', name='dense10'),
		Dense(64, activation='relu', name='dense11'),
		Dense(128, activation='relu', name='dense12'),
		Dense(256, activation='relu', name='dense13'),
		Dense(512, activation='relu', name='dense14'),
		Dense(784, activation='sigmoid', name='dense15'),
		Reshape((28,28))
	])

	model.compile(optimizer=Adam(lr=1e-3), loss='mean_absolute_error')
	print(model.summary())
	return model

def build_decoder():
	model = Sequential([
		Dense(16, activation='relu', name='dense9', input_shape=(8,)),
		Dense(32, activation='relu', name='dense10'),
		Dense(64, activation='relu', name='dense11'),
		Dense(128, activation='relu', name='dense12'),
		Dense(256, activation='relu', name='dense13'),
		Dense(512, activation='relu', name='dense14'),
		Dense(784, activation='sigmoid', name='dense15'),
		Reshape((28,28))
	])

	model.compile(optimizer=Adam(lr=1e-3), loss='mean_absolute_error')
	print(model.summary())
	return model


def train_model(model):
	save_best = ModelCheckpoint('fashion_autoencoder_best.h5', save_best_only=True)
	model.fit(x=x_train, y=x_train, epochs=30, batch_size=64, validation_data=[x_test,x_test], verbose=1, callbacks=[save_best])


if __name__=='__main__':
	model = load_model('fashion_autoencoder_best.h5')
	x_train = x_train/255.
	x_test = x_test/255.
	get_med_layer_output = K.function([model.layers[0].input],
										[model.layers[8].output])

	#model = build_autoencoder()
	#train_model(model)

	img = x_test[0]
	plt.subplot(1,2,1)
	plt.imshow(img)
	img = np.reshape(img,(1,28,28))
	yhat = model.predict(img)
	layer_output = get_med_layer_output([img])[0]
	print(layer_output)
	yhat = np.reshape(yhat, (28,28))

	plt.subplot(1,2,2)
	plt.imshow(yhat)

	plt.show()