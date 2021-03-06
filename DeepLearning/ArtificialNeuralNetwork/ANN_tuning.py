# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings
'''
These warnings filter are to unclutter output and easy-to-read purposes only!
Remember to lookup replacements. Uncomment for info
Reminder: some functions in sckitlearn have been/will be depreceated, 

'''
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

### ----------------------- Data-Preprocessing ---------------------------- ###
dataset = pd.read_csv('../_data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# dealing with categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# splitting into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### ------------------------------- Building ------------------------------ ###
# initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(
        units = 6, 
        kernel_initializer = 'uniform', 
        activation = 'relu',
        input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(
        units = 6,
        kernel_initializer = 'uniform', 
        activation = 'relu'))
# Adding the output layer
classifier.add(Dense(
        units = 1,
        kernel_initializer = 'uniform',
        activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

### --------------------- Evaluation/Tuning/Improvement --------------------###

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(
            units = 6, 
            kernel_initializer = 'uniform', 
            activation = 'relu', 
            input_dim = 11))
    classifier.add(Dense(
            units = 6,
            kernel_initializer = 'uniform',
            activation = 'relu'))
    classifier.add(Dense(
            units = 1, 
            kernel_initializer = 'uniform', 
            activation = 'sigmoid'))
    classifier.compile(
            optimizer = 'adam', 
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving and Tuning the ANN
# Dropout Regularization to reduce overfitting if needed

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform',
                         activation = 'relu', 
                         input_dim = 11))
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu'))
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_