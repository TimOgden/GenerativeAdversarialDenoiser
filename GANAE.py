import keras
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras import initializers
class GANAE:
	def constructAndTrain(self, filename):
		print('Fitting')
		(train_x, noisy_train_x), (test_x, noisy_test_x) = self.loadInData()
		train_x = self.preprocess(train_x)
		noisy_train_x = self.preprocess(noisy_train_x)
		test_x = self.preprocess(test_x)
		noisy_test_x = self.preprocess(noisy_test_x)
		#train_x = train_x.reshape(28,28,-1)
		#noisy_train_x = noisy_train_x.reshape(28,28,-1)
		#test_x = test_x.reshape(28,28,-1)
		#noisy_test_x = noisy_test_x.reshape(28,28,-1)
		self.gan.fit(x=noisy_train_x, y=train_x, epochs=10, verbose=1, batch_size=500, validation_data=[noisy_test_x, test_x])
		self.gan.save(filename)

	def loadInData(self):
		train_x = np.load('x_train.npy')
		noisy_train_x = np.load('noisy_x_train.npy')
		test_x = np.load('x_test.npy')
		noisy_test_x = np.load('noisy_x_test.npy')

		#train_x.reshape(60000,28,28)
		#noisy_train_x.reshape(60000,28,28)
		#test_x.reshape(30000,28,28)
		#noisy_test_x.reshape(30000,28,28)
		return (train_x, noisy_train_x), (test_x, noisy_test_x)

	def preprocess(self, x):
		x = x.astype('float32') / 255.
		return x.reshape(-1, np.prod(x.shape[1:])) # flatten

	def build_generator(self):
		ae = keras.models.Sequential([
			Dense(784, activation='relu', input_shape=(784,)),
			Dense(128, activation='relu'),
			Dense(100, activation='relu'),
			Dense(64, activation='relu'),
			Dense(self.encoder_size, activation='relu'),
			Dense(64, activation='relu'),
			Dense(100, activation='relu'),
			Dense(128, activation='relu'),
			Dense(784, activation='sigmoid')
		])
		return ae

	def build_discriminator(self):
		discriminator = keras.models.Sequential([
			Dense(784, activation='relu', input_shape=(784,)),
			Dense(256, activation='relu'),
			Dense(128, activation='relu'),
			Dense(64, activation='relu'),
			Dense(32, activation='relu'),
			Dense(8, activation='relu'),
			Dense(1, activation='sigmoid')
		])
		return discriminator
	
	def build_gan(self, trainable=True):
		gan = keras.models.Sequential([
			self.generator,
			self.discriminator
		])

		return gan

	def __init__(self, encoder_size=50):
		self.encoder_size = encoder_size
		self.optimizer = keras.optimizers.Adam(lr=.0001)
		self.generator = self.build_generator()
		self.generator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		self.gan = self.build_gan()
		self.gan.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		

		