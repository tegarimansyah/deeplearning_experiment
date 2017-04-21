'''
Tegar Imansyah
MNIST with tensorflow and keras
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class LeNet:
	@staticmethod
	def build(input_shape, classes, weightsPath=None):
		# initialize the model
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes, activation='softmax'))

		if weightsPath is not None:
			weightsPath = 'output/' + weightsPath
			model.load_weights(weightsPath)
 
		# return the constructed network architecture
		return model