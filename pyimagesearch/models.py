# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:38:31 2019

@author: Colin Cumming
"""
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def multi_layer_percepton(dim, regress=False):
    # starting model for data which will then be combined with image CNN
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    
    if regress:
        model.add(Dense(1, activation="linear"))
        
    # return multilayer percepton with regression on data    
    return model

def create_cnn(width, height, depth, filters = (16,32,64), regress = False):
	inputShape = (height, width, depth)
	chanDim = -1

	inputs = Input(shape=inputShape)

	for (i,f) in enumerate(filters):

		if i == 0:
			x = inputs

		x = Conv2D(f,(3,3), padding ="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis = chanDim)(x)
		x = MaxPooling2D(pool_size = (2,2))(x)

		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis = chanDim)(x)
		x = Dropout(0.5)(x)

		x = Dense(4)(x)
		x = Activation("relu")(x)

		if regress:
			x = Dense(1,activation = "linear")(x)

		model = Model(inputs,x)

    return model