# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:23:59 2019

@author: Abhishek Goel
"""
# Lets make a ANN today
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# make the matrices
X = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
Y = dataset['Exited']

from sklearn.preprocessing import LabelEncoder
label_encoder_1 = LabelEncoder()
X.iloc[:,2] = label_encoder_1.fit_transform(X.iloc[:,2])
label_encoder_2 = LabelEncoder()
X.iloc[:,1] = label_encoder_2.fit_transform(X.iloc[:,1])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

Y = Y.values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=44)

#features scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Data Pre-processing completed

# Now, we will proceed to make our ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the first hidden layer
classifier.add(Dense(use_bias=True, bias_initializer='ones',activation='relu', kernel_initializer='uniform',input_dim=11, output_dim=6))

# Adding the second Hidden Layer
classifier.add(Dense(activation='relu',use_bias=True,bias_initializer='ones', kernel_initializer='uniform', output_dim = 6))

#extra maje
classifier.add(Dense(activation='relu',use_bias=True,bias_initializer='ones', kernel_initializer='uniform', output_dim = 12))
classifier.add(Dense(activation='relu',use_bias=True,bias_initializer='ones', kernel_initializer='uniform', output_dim = 12))
classifier.add(Dense(activation='relu',use_bias=True,bias_initializer='ones', kernel_initializer='uniform', output_dim = 12))
classifier.add(Dense(activation='relu',use_bias=True,bias_initializer='ones', kernel_initializer='uniform', output_dim = 12))


# Adding the output layer
classifier.add(Dense(output_dim=1, activation='sigmoid', kernel_initializer='uniform'))

# compile and built the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the ANN
classifier.fit(X_train, y_train, batch_size=1, epochs=500, shuffle=True)

# # predicting the outputs
y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

# calculate confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_s = accuracy_score(y_test, y_pred)