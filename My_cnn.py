# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:17:05 2019

@author: Abhishek Goel
"""
# Here we will built our convolutional neural network

# importing the librarires of Keras for CNN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initialising the Graph of our model
classifier = Sequential()

# Adding first convolutional layer
classifier.add(Convolution2D(32,(3,3),strides=(1,1),activation='relu', use_bias=True, bias_initializer='ones', input_shape=(64,64,3)))
# Adding the pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Adding Second convolution layer
classifier.add(Convolution2D(32,(3,3), activation='relu'))
# Second pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening the output image
classifier.add(Flatten())

# Applying the fully connected Dense Layer
classifier.add(Dense(output_dim= 128 , activation='relu'))
classifier.add(Dense(output_dim= 1 , activation='sigmoid'))

# Compiling our CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN into our images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                              'dataset/training_set',
                              target_size=(64, 64),
                              batch_size=32,
                              class_mode='binary')

test_set = test_datagen.flow_from_directory(
                              'dataset/test_set',
                              target_size=(64, 64),
                              batch_size=32,
                              class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)